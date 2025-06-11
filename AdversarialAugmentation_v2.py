import argparse
import copy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from pathlib import Path

import pandas as pd
import os
from typing import List, Tuple, Optional
from AdversarialAugmentation import RegionReconstructor, CriticalMaskGenerator, evaluate_detector_performance, evaluate_mask_generator_performance, find_adversarial_mask_gradient_based, train_epoch_critical_mask_generator, evaluate_adversarial_refiner_performance, visualize_masked_pair
from DataBalancingDeepSeek import train_speaker, test_speaker
from DeepLearningModel import DeepfakeClassifier, AudioConfig, AudioProcessor, DeepfakeDataset, collate_fn_skip_none


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdversarialRefiner(nn.Module):
    """
    Orchestrates the mask generation, region reconstruction, and combines them.
    Aims to transform fake samples into 'real' samples as perceived by the detector.
    """
    def __init__(self, detector_model: nn.Module, region_reconstructor_model: nn.Module, critical_mask_generator_fn,
                 mask_threshold: float = 0.5, epsilon_clip: Optional[float] = None):
        super().__init__()
        self.detector = detector_model # The target deepfake detector f_theta
        self.region_reconstructor = region_reconstructor_model # The r_xi model
        self.critical_mask_generator_fn = critical_mask_generator_fn # Function or nn.Module g_psi
        self.mask_threshold = mask_threshold
        self.epsilon_clip = epsilon_clip

    def forward(self, x_fake_original_mel: torch.Tensor) -> torch.Tensor:
        """
        x_fake_original_mel: batch of mel spectrograms (B, H, W) or (B, C, H, W)
        Returns: refined mel spectrograms (B, H, W) or (B, C, H, W) (denoted as X_tilde)
        """


        # Ensure input has channel dim: B, C, H, W for internal processing
        input_was_3d = False
        if x_fake_original_mel.dim() == 3:
            x_mel_bchw = x_fake_original_mel.unsqueeze(1)
            input_was_3d = True
        else: # Assuming B,C,H,W
            x_mel_bchw = x_fake_original_mel

        # 1. Generate Critical Mask (M)
        # Gradients for mask generation are typically not backpropagated through the reconstructor when training reconstructor
        # However, if critical_mask_generator_fn is a model being co-trained, this might change in its own training step.
        # For refiner training, mask is considered fixed or from a detached part of the graph.
        # i think its better to use stochastic masks (Bernoulli sampling from g_psi output)
        # For now, let's assume critical_mask_generator_fn provides mask_prob as in original code.
        with torch.no_grad(): # Keep mask generation non-differentiable w.r.t refiner's parameters
            if isinstance(self.critical_mask_generator_fn, nn.Module):
                mask_prob = self.critical_mask_generator_fn(x_fake_original_mel) # Pass original shape
                if mask_prob.dim() == 3 and x_mel_bchw.shape[1] == 1:
                    mask_prob = mask_prob.unsqueeze(1)
                elif mask_prob.dim() != 4 or mask_prob.shape[0] != x_mel_bchw.shape[0]:
                     raise ValueError(f"Mask generator output unexpected shape: {mask_prob.shape} for input {x_fake_original_mel.shape}")
            else: # Gradient-based function
                # Ensure input for grad-based mask has grad enabled and is detached
                temp_input_for_mask = x_mel_bchw.clone().detach().requires_grad_(True)
                mask_prob = self.critical_mask_generator_fn(self.detector, temp_input_for_mask)
                mask_prob = mask_prob.detach()

            if mask_prob.shape[1] != x_mel_bchw.shape[1]:
                if mask_prob.shape[1] == 1: # Single channel mask for multi-channel input
                    mask_prob = mask_prob.repeat(1, x_mel_bchw.shape[1], 1, 1)
                else:
                    raise ValueError(f"Channel mismatch: Input {x_mel_bchw.shape}, Mask {mask_prob.shape}")
            
            # mask should be stochastic
            # M_ij ~ Bernoulli(P_M_ij) where P_M is mask_prob
            # This is a change from simple thresholding.
            # For now, sticking to original code's thresholding for direct comparison,
            # stochasticity can be added later.
            binary_mask = (mask_prob > self.mask_threshold).float()

        # 2. Get Reconstructed/Modified version from RegionReconstructor (r_xi(X_fake))
        reconstructed_mel_bchw = self.region_reconstructor(x_mel_bchw)

        # Optional: Clip the perturbation if epsilon_clip is defined (L_inf bound on perturbation itself)
        if self.epsilon_clip is not None:
            perturbation = reconstructed_mel_bchw - x_mel_bchw
            perturbation = torch.clamp(perturbation, -self.epsilon_clip, self.epsilon_clip)
            reconstructed_mel_bchw_clipped = x_mel_bchw + perturbation # Apply clipped perturbation
        else:
            reconstructed_mel_bchw_clipped = reconstructed_mel_bchw

        # 3. Combine original and reconstructed parts using the binary mask
        # X_tilde = X_orig * (1-M) + X_reconstructed_by_model * M
        refined_x_bchw = x_mel_bchw * (1 - binary_mask) + reconstructed_mel_bchw_clipped * binary_mask
        
        # Clip to valid spectrogram range (e.g., original input's dynamic range or a fixed known range)
        min_val_spec, max_val_spec = x_mel_bchw.min(), x_mel_bchw.max() # Or use known global min/max
        refined_x_bchw = torch.clamp(refined_x_bchw, min_val_spec, max_val_spec)
        
        if input_was_3d and refined_x_bchw.shape[1] == 1:
            return refined_x_bchw.squeeze(1)
        return refined_x_bchw


# --- New: Patch Discriminator (D_phi) ---
class PatchDiscriminator(nn.Module):
    """
    A PatchGAN-style discriminator to judge the realism of spectrogram patches.
    Outputs a map of scores, where each score corresponds to a patch.
    """
    def __init__(self, n_channels_in=1, base_c=64, n_layers=3):
        super().__init__()
        
        layers = [
            nn.Conv2d(n_channels_in, base_c, kernel_size=4, stride=2, padding=1), # No norm on first layer
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers += [
                nn.Conv2d(base_c * mult_prev, base_c * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_c * mult), # Or BatchNorm2d
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(base_c * mult_prev, base_c * mult, kernel_size=4, stride=1, padding=1), # Stride 1 for last conv before output
            nn.InstanceNorm2d(base_c * mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        layers += [nn.Conv2d(base_c * mult, 1, kernel_size=4, stride=1, padding=1)] # Output 1 channel (logits)
        
        self.model = nn.Sequential(*layers)
        logger.info(f"PatchDiscriminator initialized: in_channels={n_channels_in}, base_channels={base_c}, n_layers={n_layers}")

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        """
        x_mel: mel spectrogram (B, C, H, W)
        Returns: map of patch scores (logits) (B, 1, H_patch, W_patch)
        """
        return self.model(x_mel)


# --- Modified Training Function for Adversarial Refiner ---
def train_epoch_adversarial_refiner_gan(
    refiner_model: AdversarialRefiner, # Contains r_xi (region_reconstructor)
    detector_model: nn.Module, # f_theta
    patch_discriminator_model: PatchDiscriminator, # D_phi
    train_loader: DataLoader,
    optimizer_reconstructor: optim.Optimizer, # Optimizer for r_xi
    optimizer_discriminator: optim.Optimizer, # Optimizer for D_phi
    criterion_detector_adv: nn.Module, # e.g., CrossEntropyLoss for f_theta
    device: torch.device,
    current_epoch: int,
    num_epochs: int,
    lambda_gan: float = 1.0, # Weight for GAN loss for refiner
    realism_loss_l1_weight: float = 0.0 # Weight for L1 fidelity loss (was realism_loss_weight)
):
    """
    Trains the AdversarialRefiner (specifically its region_reconstructor r_xi)
    and the PatchDiscriminator D_phi in a GAN setup.
    r_xi tries to fool f_theta and D_phi.
    D_phi tries to distinguish real (original fakes) from r_xi's output.
    """
    refiner_model.train() # Sets r_xi (region_reconstructor) to train mode
    patch_discriminator_model.train()
    detector_model.eval() # f_theta is fixed during this training phase

    total_loss_refiner_adv_f_theta = 0.0
    total_loss_refiner_adv_d_phi = 0.0
    total_loss_refiner_l1 = 0.0
    total_loss_discriminator = 0.0
    num_processed_samples = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None: continue
        features_dict, labels = batch_data
        
        # We only refine 'fake' samples (label 0 for f_theta)
        # These 'fake_mel_spectrograms' are considered "real" examples for D_phi
        original_fake_mels = features_dict["mel"][labels == 0].to(device) # Shape: (B_fake, H, W)
        if original_fake_mels.size(0) == 0: continue

        # Ensure input to refiner has channel dim if needed by its components
        # AdversarialRefiner's forward handles this internally to produce B,C,H,W
        
        # --- Train Patch Discriminator (D_phi) ---
        optimizer_discriminator.zero_grad()

        # D_phi expects B,C,H,W. Ensure original_fake_mels has channel dim.
        if original_fake_mels.dim() == 3:
            original_fake_mels_bchw = original_fake_mels.unsqueeze(1)
        else:
            original_fake_mels_bchw = original_fake_mels

        # Real samples for D_phi (original fakes)
        d_phi_logits_real = patch_discriminator_model(original_fake_mels_bchw)
        loss_d_real = F.binary_cross_entropy_with_logits(
            d_phi_logits_real, torch.ones_like(d_phi_logits_real, device=device)
        )

        # Fake samples for D_phi (output of r_xi, detached)
        # refined_output_r_xi is X_tilde, shape (B_fake, H, W) or (B_fake, C, H, W)
        refined_output_r_xi = refiner_model(original_fake_mels) # Pass original shape (B,H,W)
        
        # Ensure refined_output_r_xi has channel dim for D_phi
        if refined_output_r_xi.dim() == 3:
            refined_output_r_xi_bchw = refined_output_r_xi.unsqueeze(1)
        else:
            refined_output_r_xi_bchw = refined_output_r_xi
            
        d_phi_logits_fake = patch_discriminator_model(refined_output_r_xi_bchw.detach()) # Detach from r_xi's graph
        loss_d_fake = F.binary_cross_entropy_with_logits(
            d_phi_logits_fake, torch.zeros_like(d_phi_logits_fake, device=device)
        )
        
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_discriminator.step()
        total_loss_discriminator += loss_d.item() * original_fake_mels.size(0)

        # --- Train Refiner (r_xi part of refiner_model) ---
        optimizer_reconstructor.zero_grad()

        # 1. Adversarial Loss against f_theta (target detector)
        # Ensure refined_output_r_xi_bchw is suitable for detector_model
        # detector_model might expect B,1,H,W or B,H,W (if it unsqueezes internally)
        # Assuming detector_model handles B,C,H,W or B,1,H,W
        refined_logits_f_theta = detector_model(refined_output_r_xi_bchw) # Use the non-detached version for graph
        target_labels_real_for_f_theta = torch.ones(refined_logits_f_theta.size(0), dtype=torch.long, device=device)
        loss_adv_f_theta = criterion_detector_adv(refined_logits_f_theta, target_labels_real_for_f_theta)

        # 2. Adversarial Loss against D_phi (patch discriminator)
        # r_xi wants D_phi to classify its output as real
        d_phi_logits_refined_for_r_xi = patch_discriminator_model(refined_output_r_xi_bchw)
        loss_adv_d_phi = F.binary_cross_entropy_with_logits(
            d_phi_logits_refined_for_r_xi, torch.ones_like(d_phi_logits_refined_for_r_xi, device=device)
        )

        # 3. Fidelity/Realism Loss (L1)
        loss_l1_fidelity = torch.tensor(0.0, device=device)
        if realism_loss_l1_weight > 0:
            # Compare refined_output_r_xi (B,H,W or B,C,H,W) with original_fake_mels (B,H,W)
            # Ensure shapes match for L1 loss. If refined is B,C,H,W and C=1, squeeze it.
            refined_for_l1 = refined_output_r_xi_bchw
            if refined_for_l1.shape[1] == 1 and original_fake_mels.dim() == 3: # B,1,H,W vs B,H,W
                refined_for_l1 = refined_for_l1.squeeze(1)
            
            if refined_for_l1.shape == original_fake_mels.shape:
                loss_l1_fidelity = F.l1_loss(refined_for_l1, original_fake_mels)
                total_loss_refiner_l1 += loss_l1_fidelity.item() * original_fake_mels.size(0)
            else:
                logger.warning(f"Skipping L1 fidelity loss due to shape mismatch: refined {refined_for_l1.shape}, original {original_fake_mels.shape}")
        
        # Total loss for refiner's reconstructor (r_xi)
        total_loss_r_xi = loss_adv_f_theta + \
                          lambda_gan * loss_adv_d_phi + \
                          realism_loss_l1_weight * loss_l1_fidelity
        
        total_loss_r_xi.backward()
        optimizer_reconstructor.step()

        total_loss_refiner_adv_f_theta += loss_adv_f_theta.item() * original_fake_mels.size(0)
        total_loss_refiner_adv_d_phi += loss_adv_d_phi.item() * original_fake_mels.size(0)
        num_processed_samples += original_fake_mels.size(0)

        if batch_idx % 20 == 0: # Log progress
            log_msg = (
                f"RefinerGAN Epoch {current_epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss_D: {loss_d.item():.4f} | "
                f"Loss_R_fTheta: {loss_adv_f_theta.item():.4f} | "
                f"Loss_R_DPhi: {loss_adv_d_phi.item():.4f}"
            )
            if realism_loss_l1_weight > 0 and loss_l1_fidelity.item() > 0:
                log_msg += f" | Loss_R_L1: {loss_l1_fidelity.item():.4f}"
            logger.info(log_msg)
            
        if batch_idx == 20 and original_fake_mels.size(0) > 0:
                visualize_masked_pair(
                    x_orig=original_fake_mels[0].cpu(),
                    x_masked=refined_output_r_xi[0].cpu().detach(), # Use the output from r_xi
                    # mask_prob is not directly available here unless recomputed or passed from AdversarialRefiner
                    # For simplicity, can pass None or a dummy mask for visualization
                    masks_prob=None, 
                    sample_idx=0
                )

    avg_loss_d = total_loss_discriminator / num_processed_samples if num_processed_samples > 0 else 0
    avg_loss_r_ftheta = total_loss_refiner_adv_f_theta / num_processed_samples if num_processed_samples > 0 else 0
    avg_loss_r_dphi = total_loss_refiner_adv_d_phi / num_processed_samples if num_processed_samples > 0 else 0
    avg_loss_r_l1 = total_loss_refiner_l1 / num_processed_samples if num_processed_samples > 0 and realism_loss_l1_weight > 0 else 0
    
    log_summary = (
        f"RefinerGAN Epoch {current_epoch}/{num_epochs} Summary | "
        f"Avg Loss_D: {avg_loss_d:.4f} | "
        f"Avg Loss_R_fTheta: {avg_loss_r_ftheta:.4f} | "
        f"Avg Loss_R_DPhi: {avg_loss_r_dphi:.4f}"
    )
    if realism_loss_l1_weight > 0:
        log_summary += f" | Avg Loss_R_L1: {avg_loss_r_l1:.4f}"
    logger.info(log_summary)
    
    return avg_loss_r_ftheta, avg_loss_r_dphi, avg_loss_r_l1, avg_loss_d



def setup_environment_and_config(args):
    """Sets up device, multiprocessing, and audio configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    if args.num_workers > 0 and device.type == 'cuda':
        current_start_method = torch.multiprocessing.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            logger.info("Setting multiprocessing start method to 'spawn' for CUDA compatibility.")
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError as e:
                logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might cause issues.")

    audio_conf = AudioConfig(n_mels=args.n_mels, max_duration=args.max_duration)
    return device, audio_conf


def load_and_prepare_data(args, audio_conf):
    """Loads metadata, creates speaker-disjoint splits, and prepares DataLoaders."""
    logger.info("--- Loading and Splitting Metadata ---")
    metadata_path = Path(args.metadata_path)
    data_root_dir = Path(args.data_dir)

    if not metadata_path.exists():
        logger.error(f"Metadata CSV file not found at {args.metadata_path}. Exiting.")
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_path}")

    full_metadata_df = pd.read_csv(metadata_path)
    if full_metadata_df.empty:
        logger.error(f"Metadata CSV file at {args.metadata_path} is empty. Exiting.")
        raise ValueError(f"Metadata CSV is empty: {args.metadata_path}")

    if 'label' not in full_metadata_df.columns or (not all(label in ['bona-fide', 'spoof'] for label in full_metadata_df['label'].unique())):
        logger.error("Metadata CSV must contain a 'label' column with 'bona-fide' or 'spoof'. Exiting.")
        raise ValueError("Invalid 'label' column in metadata.")
    
    if 'speaker' not in full_metadata_df.columns:
        logger.error("Metadata CSV must contain a 'speaker' column for speaker-disjoint splits. Exiting.")
        raise ValueError("Missing 'speaker' column in metadata.")



    train_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(train_speaker)].reset_index(drop=True)
    val_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(test_speaker)].reset_index(drop=True)

    logger.info(f"Total training samples from metadata: {len(train_metadata_df)}")
    logger.info(f"Total validation samples from metadata: {len(val_metadata_df)}")
    if not train_metadata_df.empty:
        logger.info(f"Training label balance: {train_metadata_df['label'].value_counts().to_dict()}")
    if not val_metadata_df.empty:
        logger.info(f"Validation label balance: {val_metadata_df['label'].value_counts().to_dict()}")

    processor = AudioProcessor(audio_conf)
    train_dataset = DeepfakeDataset(data_root_dir, train_metadata_df, processor, augment=True)
    val_dataset = DeepfakeDataset(data_root_dir, val_metadata_df, processor, augment=False)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Check data and metadata. Exiting.")
        raise ValueError("Training dataset is empty.")
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_skip_none, num_workers=args.num_workers, pin_memory=True if torch.device.type == 'cuda' else False
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn_skip_none, num_workers=args.num_workers, pin_memory=True if torch.device.type == 'cuda' else False
        )
    else:
        logger.warning("Validation dataset is empty or could not be created.")
        # Depending on strictness, might raise ValueError here or allow running without validation for some components
        # For now, allowing it to be None and subsequent functions should handle it.

    logger.info("DataLoaders created.")
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset) if val_dataset else 0}")
    
    return train_loader, val_loader, val_metadata_df, processor, data_root_dir




def initialize_detector(args, audio_conf, device, pretrained_path="best_model_deepfake.pth"):
    """Initializes the DeepfakeClassifier model and loads pretrained weights."""
    logger.info(f"--- Initializing Detector from {pretrained_path} ---")
    detector = DeepfakeClassifier(
        num_classes=2, config=audio_conf, dropout=args.dropout,
        initial_channels=args.initial_channels, resnet_channels=args.resnet_channels,
        resnet_blocks=args.resnet_blocks, classifier_hidden_dim=args.classifier_hidden_dim
    ).to(device)

    if Path(pretrained_path).exists():
        try:
            detector.load_state_dict(torch.load(pretrained_path, map_location=device))
            logger.info(f"Loaded pretrained detector weights from {pretrained_path}")
        except Exception as e:
            logger.error(f"Error loading pretrained detector weights from {pretrained_path}: {e}. Starting with a fresh model.")
    else:
        logger.warning(f"Pretrained detector path {pretrained_path} not found. Initializing with random weights.")
    
    detector.eval() # Set to eval mode by default; training happens only during hardening
    return detector

def get_mask_generator_function(args, detector, train_loader, val_loader, device):
    """Initializes and optionally trains the Critical Mask Generator."""
    if args.use_learned_mask_generator:
        logger.info("--- Initializing/Loading Learned Critical Mask Generator ---")
        mask_gen_model = CriticalMaskGenerator(unet_base_c=args.mask_gen_unet_base_c).to(device) # Added arg
        
        if args.train_mask_generator:
            logger.info("--- Training Learned Critical Mask Generator ---")
            optimizer_mask_gen = optim.Adam(mask_gen_model.parameters(), lr=args.lr_mask_gen)
            best_kl = 0.0 
            best_epoch = -1
            best_model_state = None
            for epoch in range(args.epochs_mask_gen):
                mask_gen_model.train()
                avg_kl_div_train, avg_budget_loss_train = train_epoch_critical_mask_generator(
                    mask_gen_model, detector, train_loader, optimizer_mask_gen, device,
                    args.lambda_budget, args.target_mask_density, epoch, args.epochs_mask_gen
                )
                if val_loader:
                    mask_gen_model.eval()
                    val_kl_div, val_mask_density = evaluate_mask_generator_performance(
                        mask_gen_model, detector, val_loader, device,
                        args.lambda_budget, args.target_mask_density # Pass lambda and target for consistency
                    )
                    logger.info(f"MaskGen Epoch {epoch+1}/{args.epochs_mask_gen} | Train KL: {avg_kl_div_train:.4f}, Budget Loss: {avg_budget_loss_train:.4f} | Val KL: {val_kl_div:.4f}, Mask Density: {val_mask_density:.4f}")
                if val_kl_div > best_kl : # Simple criteria, can be more complex
                    best_kl = best_kl
                    best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(detector.state_dict())
                    
                    save_path = 'best_hardened_detector.pth'
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_model_state,
            
                        'args': args # Save args for reproducibility
                    }, save_path)
                    logger.info(f"🔥 New best hardened model saved to {save_path} from Epoch {best_epoch} ")
            
                else:
                    logger.info(f"MaskGen Epoch {epoch+1}/{args.epochs_mask_gen} | Train KL: {avg_kl_div_train:.4f}, Budget Loss: {avg_budget_loss_train:.4f} | Val loader not available for eval.")
            logger.info("Learned Critical Mask Generator training finished.")
        else:
            logger.info("Skipping Learned Critical Mask Generator training (assuming pre-trained or will load weights).")
            # Should? Add logic here to load pre-trained mask generator weights if applicable
            # if args.pretrained_mask_gen_path and Path(args.pretrained_mask_gen_path).exists():
            #    mask_gen_model.load_state_dict(torch.load(args.pretrained_mask_gen_path, map_location=device))
        
        mask_gen_model.eval()
        return mask_gen_model # Return the model instance
    else:
        logger.info("--- Using Gradient-Based Saliency for Critical Regions ---")
        # The function itself is returned, to be called with (mel_spectrograms, detector_model)
        return find_adversarial_mask_gradient_based


def get_adversarial_refiner(args, detector, mask_generator_fn, train_loader, val_loader, device):
    """Initializes and optionally trains the Adversarial Refiner."""
    if not args.train_refiner:
        logger.info("Skipping Adversarial Refiner training and setup.")
        return None

    logger.info("--- Initializing Adversarial Refiner (Region Reconstructor) ---")
    region_reconstructor = RegionReconstructor(
        n_channels_in=1, n_channels_out=1, unet_base_c=args.refiner_unet_base_c # Added arg
    ).to(device)
    
    adv_refiner = AdversarialRefiner(
        detector, region_reconstructor, mask_generator_fn,
        mask_threshold=args.mask_threshold, epsilon_clip=args.refiner_epsilon_clip
    ).to(device)

    logger.info("--- Training Adversarial Refiner ---")
    optimizer_reconstructor = optim.Adam(region_reconstructor.parameters(), lr=args.lr_refiner)
    # Note: The original code uses CrossEntropyLoss for adv_refiner, which implies the detector's output (logits)
    # is used directly. The refiner's goal is to make the detector classify refined fakes as 'bona-fide' (label 1).
    criterion_adv = nn.CrossEntropyLoss() # This is for the detector's output

    for epoch in range(args.epochs_refiner):
        adv_refiner.train() # Set refiner (and its submodules like reconstructor) to train
        avg_adv_loss_train, avg_realism_loss_train = train_epoch_adversarial_refiner_gan(
            adv_refiner, detector, train_loader, optimizer_reconstructor, criterion_adv, device,
            epoch, args.epochs_refiner, args.realism_loss_weight
        )
        if val_loader:
            adv_refiner.eval()
            val_adv_loss, val_realism_loss, refiner_success_rate = evaluate_adversarial_refiner_performance(
                adv_refiner, detector, val_loader, device, args.realism_loss_weight
            )
            logger.info(f"Refiner Epoch {epoch+1}/{args.epochs_refiner} | Train Adv: {avg_adv_loss_train:.4f}, Realism: {avg_realism_loss_train:.4f} | Val Adv: {val_adv_loss:.4f}, Realism: {val_realism_loss:.4f}, Success: {refiner_success_rate:.4f}")
        else:
            logger.info(f"Refiner Epoch {epoch+1}/{args.epochs_refiner} | Train Adv: {avg_adv_loss_train:.4f}, Realism: {avg_realism_loss_train:.4f} | Val loader not available for eval.")
    
    logger.info("Adversarial Refiner training finished.")
    adv_refiner.eval()
    return adv_refiner

def _evaluate_hardened_detector_on_refined_fakes(detector, adv_refiner, val_metadata_df, data_root_dir, processor, args, device):
    """Helper to evaluate the hardened detector on adversarially refined fake samples from validation set."""
    if adv_refiner is None:
        logger.info("Adversarial refiner not available, skipping evaluation on refined fakes.")
        return 0.0
    if val_metadata_df is None or val_metadata_df.empty:
        logger.info("Validation metadata not available, skipping evaluation on refined fakes.")
        return 0.0
        
    val_fake_metadata_df = val_metadata_df[val_metadata_df['label'] == 'spoof']
    if val_fake_metadata_df.empty:
        logger.info("No spoof samples in validation metadata to refine.")
        return 0.0

    val_fake_dataset = DeepfakeDataset(data_root_dir, val_fake_metadata_df, processor, augment=False)
    if len(val_fake_dataset) == 0:
        logger.info("Validation fake dataset is empty.")
        return 0.0
        
    val_fake_loader = torch.utils.data.DataLoader(
        val_fake_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_skip_none, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    total_refined_fake_correct = 0
    total_refined_fake_samples = 0
    detector.eval() # Ensure detector is in eval mode for this
    adv_refiner.eval() # Ensure refiner is in eval mode

    with torch.no_grad():
        for batch_data_fake in val_fake_loader:
            if batch_data_fake is None: continue
            features_dict_fake, labels_fake = batch_data_fake # labels_fake are all 0 (spoof)
            fake_mels_val = features_dict_fake["mel"].to(device)
            
            if fake_mels_val.size(0) == 0: continue

            # AdversarialRefiner expects (B, H, W)
            refined_val_fakes_bhw = adv_refiner(fake_mels_val) 
            
            # Detector expects (B, 1, H, W) or (B, H, W) which it unsqueezes
            outputs_refined = detector(refined_val_fakes_bhw) # Pass (B,H,W)
            _, predicted_refined = torch.max(outputs_refined.data, 1) # Predicted label
            
            # Correct if predicted as spoof (label 0)
            total_refined_fake_correct += (predicted_refined == 0).sum().item() 
            total_refined_fake_samples += labels_fake.size(0)
            
    hardened_acc_on_refined = total_refined_fake_correct / total_refined_fake_samples if total_refined_fake_samples > 0 else 0.0
    logger.info(f"Hardened Detector Val (on Refined Fakes) Accuracy: {hardened_acc_on_refined:.4f} ({total_refined_fake_correct}/{total_refined_fake_samples})")
    return hardened_acc_on_refined




def adversarially_harden_detector_on_mask(args, detector, mask_gen_model, train_loader, val_loader, 
                                  val_metadata_df, processor, data_root_dir, device):
    """Performs adversarial hardening of the detector."""
    if not args.adversarial_hardening_detector:
        logger.info("Skipping Detector Adversarial Hardening.")
        return
    if mask_gen_model is None:
        logger.error("Adversarial Refiner not available. Cannot perform hardening. Skipping.")
        return

    logger.info("--- Adversarially Hardening Detector ---")
    mask_gen_model.eval() # Refiner is used for attack generation, not training here

    optimizer_detector_hardened = optim.AdamW(detector.parameters(), lr=args.lr_detector_hardening)
    criterion_detector_hardened = nn.CrossEntropyLoss()

    best_clean_auc = 0.0
    best_refined_acc = 0.0 # Best accuracy on refined fakes
    best_epoch = -1
    best_model_state = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs_hardening):
        detector.train() # Set detector to train mode
        total_hardening_loss = 0.0
        num_hardening_samples = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None: continue
            features_dict, labels = batch_data # labels: 0 for spoof, 1 for bona-fide
            x_mels = features_dict["mel"].to(device) # (B, H, W)
            labels = labels.to(device)

            optimizer_detector_hardened.zero_grad()

            real_mels_orig = x_mels[labels == 1]
            fake_mels_orig = x_mels[labels == 0]
            
            real_labels_orig = labels[labels == 1]
            # fake_labels_for_refined = torch.zeros(fake_mels_orig.size(0), device=device, dtype=labels.dtype)

            current_batch_inputs = []
            current_batch_labels = []

            if real_mels_orig.size(0) > 0:
                current_batch_inputs.append(real_mels_orig)
                current_batch_labels.append(real_labels_orig)

            if fake_mels_orig.size(0) > 0:
                # Preserve a fraction of original fakes (unrefined)
                # This helps prevent catastrophic forgetting of original spoof characteristics
                keep_frac = args.hardening_keep_original_fake_frac # e.g., 0.1
                n_total_fakes = fake_mels_orig.size(0)
                n_keep_unrefined = int(keep_frac * n_total_fakes)
                
                # Ensure at least one sample is refined if there are fakes, unless n_keep_unrefined is all
                if n_keep_unrefined < n_total_fakes and n_total_fakes > 0 :
                    n_refine = n_total_fakes - n_keep_unrefined
                elif n_total_fakes > 0 : # only keep unrefined or only refine
                     n_refine = 0 if n_keep_unrefined == n_total_fakes else n_total_fakes
                else:
                    n_refine = 0

                indices = torch.randperm(n_total_fakes)
                unrefined_indices = indices[:n_keep_unrefined]
                refine_indices = indices[n_keep_unrefined : n_keep_unrefined + n_refine]

                if n_keep_unrefined > 0:
                    fake_mels_unrefined_keep = fake_mels_orig[unrefined_indices]
                    current_batch_inputs.append(fake_mels_unrefined_keep)
                    current_batch_labels.append(torch.zeros(n_keep_unrefined, device=device, dtype=labels.dtype))
                
                if n_refine > 0:
                    fake_mels_to_refine = fake_mels_orig[refine_indices]  # shape: (n_refine, H, W)
                    with torch.no_grad():
                        # 1) Generate mask only on the subset we want to refine
                        masks_prob = mask_gen_model(fake_mels_to_refine)  # → (n_refine, H, W) or (n_refine, C, H, W)

                        # 2) Turn that into (n_refine, 1, H, W) if needed—or keep (n_refine, C, H, W)
                        if masks_prob.dim() == 3:  # (n_refine, H, W)
                            masks_prob_bchw = masks_prob.unsqueeze(1)  # → (n_refine, 1, H, W)
                        else:  # already (n_refine, C, H, W)
                            masks_prob_bchw = masks_prob

                        # 3) Likewise expand fake_mels_to_refine to (n_refine, 1, H, W) if needed
                        if fake_mels_to_refine.dim() == 3:  # (n_refine, H, W)
                            x_subset_bchw = fake_mels_to_refine.unsqueeze(1)  # → (n_refine, 1, H, W)
                        else:  # (n_refine, C, H, W)
                            x_subset_bchw = fake_mels_to_refine

                        # 4) If the mask has only one channel but the data has >1 channel, repeat:
                        if masks_prob_bchw.shape[1] != x_subset_bchw.shape[1]:
                            if masks_prob_bchw.shape[1] == 1:
                                masks_prob_bchw = masks_prob_bchw.repeat(1, x_subset_bchw.shape[1], 1, 1)
                            else:
                                logger.warning(
                                    f"Mask channels ({masks_prob_bchw.shape[1]}) != data channels ({x_subset_bchw.shape[1]}). Skipping these {n_refine} samples."
                                )
                                n_refine = 0  # skip
                        # 5) Finally apply the mask on the subset only:
                        if n_refine > 0:
                            adv_fake_refined = x_subset_bchw * (1 - masks_prob_bchw)  # → (n_refine, C, H, W)
                            adv_fake_refined = adv_fake_refined.squeeze(1)           # → (n_refine, H, W) if C==1

                    if n_refine > 0:
                        current_batch_inputs.append(adv_fake_refined)  # shape matches current_batch_labels
                        current_batch_labels.append(torch.zeros(n_refine, device=device, dtype=labels.dtype))


                    current_batch_inputs.append(adv_fake_refined)
                    current_batch_labels.append(torch.zeros(n_refine, device=device, dtype=labels.dtype))


            if not current_batch_inputs: # Skip if batch becomes empty (e.g. all fakes and no reals)
                continue
                
            combined_inputs = torch.cat(current_batch_inputs, dim=0) # All are (B, H, W)
            combined_labels = torch.cat(current_batch_labels, dim=0)

            # Detector forward pass expects (B, H, W) or (B, 1, H, W)
            outputs = detector(combined_inputs) # Pass (B, H, W)
            loss = criterion_detector_hardened(outputs, combined_labels)
            
            loss.backward()
            optimizer_detector_hardened.step()

            total_hardening_loss += loss.item() * combined_inputs.size(0)
            num_hardening_samples += combined_inputs.size(0)

            if batch_idx % args.log_interval == 0: # Use args.log_interval
                logger.info(f"Hardening Epoch {epoch+1}/{args.epochs_hardening} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_hardening_loss = total_hardening_loss / num_hardening_samples if num_hardening_samples > 0 else 0.0
        logger.info(f"Hardening Epoch {epoch+1} Avg Train Loss: {avg_hardening_loss:.4f}")

        # Evaluate hardened detector
        if val_loader:
            detector.eval() # Switch to eval mode for validation
            logger.info(f"--- Evaluating Hardened Detector Performance (Epoch {epoch+1}) ---")
            hardened_acc_clean, hardened_auc_clean = evaluate_detector_performance(detector, val_loader, device)
            logger.info(f"Hardened Detector Val (Clean) Acc: {hardened_acc_clean:.4f}%, AUC: {hardened_auc_clean:.4f}")
            
            hardened_acc_on_refined_fakes = _evaluate_hardened_detector_on_refined_fakes(
                detector, mask_gen_model, val_metadata_df, data_root_dir, processor, args, device
            )

            # Checkpoint saving logic
            # maybe: save if combination of clean AUC and refined acc is better
            # Or prioritize one, e.g. clean AUC.
            if hardened_auc_clean > best_clean_auc : # Simple criteria, can be more complex
                best_clean_auc = hardened_auc_clean
                best_refined_acc = hardened_acc_on_refined_fakes # Store corresponding refined acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(detector.state_dict())
                
                save_path = output_dir / 'best_hardened_detector.pth'
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer_detector_hardened.state_dict(), # Optional
                    'clean_auc': best_clean_auc,
                    'refined_acc': best_refined_acc,
                    'args': args # Save args for reproducibility
                }, save_path)
                logger.info(f"🔥 New best hardened model saved to {save_path} from Epoch {best_epoch} | Clean AUC: {best_clean_auc:.4f} | Refined Acc: {best_refined_acc:.4f}")
        else:
            logger.info("Validation loader not available. Skipping evaluation during hardening epoch.")

    logger.info(f"Hardening complete. Best model from epoch {best_epoch} with Clean AUC: {best_clean_auc:.4f}, Refined Acc: {best_refined_acc:.4f}")
    
    final_save_path = output_dir / 'final_hardened_detector.pth'
    torch.save({
        'epoch': args.epochs_hardening,
        'model_state_dict': detector.state_dict(), # Save the state at the end of all epochs
        'args': args
    }, final_save_path)
    logger.info(f"Final hardened detector model saved to {final_save_path}")

    if best_model_state:
        detector.load_state_dict(best_model_state)
        logger.info(f"Loaded best hardened detector (Epoch {best_epoch}) for potential further use.")



def adversarially_harden_detector(args, detector, adv_refiner, train_loader, val_loader, 
                                  val_metadata_df, processor, data_root_dir, device):
    """Performs adversarial hardening of the detector."""
    if not args.adversarial_hardening_detector:
        logger.info("Skipping Detector Adversarial Hardening.")
        return
    if adv_refiner is None:
        logger.error("Adversarial Refiner not available. Cannot perform hardening. Skipping.")
        return

    logger.info("--- Adversarially Hardening Detector ---")
    adv_refiner.eval() # Refiner is used for attack generation, not training here

    optimizer_detector_hardened = optim.AdamW(detector.parameters(), lr=args.lr_detector_hardening)
    criterion_detector_hardened = nn.CrossEntropyLoss()

    best_clean_auc = 0.0
    best_refined_acc = 0.0 # Best accuracy on refined fakes
    best_epoch = -1
    best_model_state = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs_hardening):
        detector.train() # Set detector to train mode
        total_hardening_loss = 0.0
        num_hardening_samples = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None: continue
            features_dict, labels = batch_data # labels: 0 for spoof, 1 for bona-fide
            x_mels = features_dict["mel"].to(device) # (B, H, W)
            labels = labels.to(device)

            optimizer_detector_hardened.zero_grad()

            real_mels_orig = x_mels[labels == 1]
            fake_mels_orig = x_mels[labels == 0]
            
            real_labels_orig = labels[labels == 1]
            # fake_labels_for_refined = torch.zeros(fake_mels_orig.size(0), device=device, dtype=labels.dtype)

            current_batch_inputs = []
            current_batch_labels = []

            if real_mels_orig.size(0) > 0:
                current_batch_inputs.append(real_mels_orig)
                current_batch_labels.append(real_labels_orig)

            if fake_mels_orig.size(0) > 0:
                # Preserve a fraction of original fakes (unrefined)
                # This helps prevent catastrophic forgetting of original spoof characteristics
                keep_frac = args.hardening_keep_original_fake_frac # e.g., 0.1
                n_total_fakes = fake_mels_orig.size(0)
                n_keep_unrefined = int(keep_frac * n_total_fakes)
                
                # Ensure at least one sample is refined if there are fakes, unless n_keep_unrefined is all
                if n_keep_unrefined < n_total_fakes and n_total_fakes > 0 :
                    n_refine = n_total_fakes - n_keep_unrefined
                elif n_total_fakes > 0 : # only keep unrefined or only refine
                     n_refine = 0 if n_keep_unrefined == n_total_fakes else n_total_fakes
                else:
                    n_refine = 0

                indices = torch.randperm(n_total_fakes)
                unrefined_indices = indices[:n_keep_unrefined]
                refine_indices = indices[n_keep_unrefined : n_keep_unrefined + n_refine]

                if n_keep_unrefined > 0:
                    fake_mels_unrefined_keep = fake_mels_orig[unrefined_indices]
                    current_batch_inputs.append(fake_mels_unrefined_keep)
                    current_batch_labels.append(torch.zeros(n_keep_unrefined, device=device, dtype=labels.dtype))
                
                if n_refine > 0:
                    fake_mels_to_refine = fake_mels_orig[refine_indices]
                    with torch.no_grad(): # Generate adversarial examples
                        # adv_refiner expects (B, H, W) and outputs (B, H, W)
                        adv_fake_refined = adv_refiner(fake_mels_to_refine) 
                    current_batch_inputs.append(adv_fake_refined)
                    current_batch_labels.append(torch.zeros(n_refine, device=device, dtype=labels.dtype))
            
            if not current_batch_inputs: # Skip if batch becomes empty (e.g. all fakes and no reals)
                continue
                
            combined_inputs = torch.cat(current_batch_inputs, dim=0) # All are (B, H, W)
            combined_labels = torch.cat(current_batch_labels, dim=0)

            # Detector forward pass expects (B, H, W) or (B, 1, H, W)
            outputs = detector(combined_inputs) # Pass (B, H, W)
            loss = criterion_detector_hardened(outputs, combined_labels)
            
            loss.backward()
            optimizer_detector_hardened.step()

            total_hardening_loss += loss.item() * combined_inputs.size(0)
            num_hardening_samples += combined_inputs.size(0)

            if batch_idx % args.log_interval == 0: # Use args.log_interval
                logger.info(f"Hardening Epoch {epoch+1}/{args.epochs_hardening} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_hardening_loss = total_hardening_loss / num_hardening_samples if num_hardening_samples > 0 else 0.0
        logger.info(f"Hardening Epoch {epoch+1} Avg Train Loss: {avg_hardening_loss:.4f}")

        # Evaluate hardened detector
        if val_loader:
            detector.eval() # Switch to eval mode for validation
            logger.info(f"--- Evaluating Hardened Detector Performance (Epoch {epoch+1}) ---")
            hardened_acc_clean, hardened_auc_clean = evaluate_detector_performance(detector, val_loader, device)
            logger.info(f"Hardened Detector Val (Clean) Acc: {hardened_acc_clean:.4f}%, AUC: {hardened_auc_clean:.4f}")
            
            hardened_acc_on_refined_fakes = _evaluate_hardened_detector_on_refined_fakes(
                detector, adv_refiner, val_metadata_df, data_root_dir, processor, args, device
            )

            # Checkpoint saving logic
            # maybe: save if combination of clean AUC and refined acc is better
            # Or prioritize one, e.g. clean AUC.
            if hardened_auc_clean > best_clean_auc : # Simple criteria, can be more complex
                best_clean_auc = hardened_auc_clean
                best_refined_acc = hardened_acc_on_refined_fakes # Store corresponding refined acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(detector.state_dict())
                
                save_path = output_dir / 'best_hardened_detector.pth'
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer_detector_hardened.state_dict(), # Optional
                    'clean_auc': best_clean_auc,
                    'refined_acc': best_refined_acc,
                    'args': args # Save args for reproducibility
                }, save_path)
                logger.info(f"🔥 New best hardened model saved to {save_path} from Epoch {best_epoch} | Clean AUC: {best_clean_auc:.4f} | Refined Acc: {best_refined_acc:.4f}")
        else:
            logger.info("Validation loader not available. Skipping evaluation during hardening epoch.")

    logger.info(f"Hardening complete. Best model from epoch {best_epoch} with Clean AUC: {best_clean_auc:.4f}, Refined Acc: {best_refined_acc:.4f}")
    
    final_save_path = output_dir / 'final_hardened_detector.pth'
    torch.save({
        'epoch': args.epochs_hardening,
        'model_state_dict': detector.state_dict(), # Save the state at the end of all epochs
        'args': args
    }, final_save_path)
    logger.info(f"Final hardened detector model saved to {final_save_path}")

    if best_model_state:
        detector.load_state_dict(best_model_state)
        logger.info(f"Loaded best hardened detector (Epoch {best_epoch}) for potential further use.")


def run_adversarial_pipeline_refactored(args):
    """Orchestrates the refactored adversarial deepfake detection pipeline."""
    
    # --- 0. Setup ---
    device, audio_conf = setup_environment_and_config(args)

    # --- 1. Data Loading ---
    try:
        train_loader, val_loader, val_metadata_df, processor, data_root_dir = load_and_prepare_data(args, audio_conf)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data: {e}")
        return

    # --- 2. Detector Initialization ---
    # Allow specifying pretrained path via args, fallback to default
    detector_pretrained_path = getattr(args, 'detector_pretrained_path', "best_model_deepfake.pth")
    detector = initialize_detector(args, audio_conf, device, pretrained_path=detector_pretrained_path)
    
    # Optionally, evaluate initial detector
    if args.evaluate_initial_detector and val_loader:
        logger.info("--- Evaluating Initial Pretrained Detector ---")
        initial_acc, initial_auc = evaluate_detector_performance(detector, val_loader, device)
        logger.info(f"Initial Detector Val Accuracy: {initial_acc:.4f}%, AUC: {initial_auc:.4f}")

    # --- 3. Critical Mask Generator ---
    # The function returns either a model (if learned) or a function (if gradient-based)
    mask_generator_fn_for_refiner = get_mask_generator_function(args, detector, train_loader, val_loader, device)

    # --- 4. Adversarial Refiner ---
    adv_refiner = get_adversarial_refiner(args, detector, mask_generator_fn_for_refiner, train_loader, val_loader, device)
    
    # Optionally, evaluate refiner if it was trained/loaded
    if adv_refiner and args.evaluate_refiner_independently and val_loader:
        logger.info("--- Evaluating Trained Adversarial Refiner Independently ---")
        # This evaluation is already part of the training loop, but can be called again if needed
        val_adv_loss, val_realism_loss, refiner_success_rate = evaluate_adversarial_refiner_performance(
            adv_refiner, detector, val_loader, device, args.realism_loss_weight
        )
        logger.info(f"Independent Refiner Val Adv Loss: {val_adv_loss:.4f}, Realism Loss: {val_realism_loss:.4f}, Success Rate: {refiner_success_rate:.4f}")


    # --- 5. Adversarial Hardening of the Detector ---
    #adversarially_harden_detector(args, detector, adv_refiner, train_loader, val_loader,
    #                              val_metadata_df, processor, data_root_dir, device)
    adversarially_harden_detector_on_mask(args, detector, mask_generator_fn_for_refiner, train_loader, val_loader,
                                  val_metadata_df, processor, data_root_dir, device)
    

    logger.info("Adversarial pipeline run completed.")




def main():
    parser = argparse.ArgumentParser(description="Adversarial Suite for Deepfake Detection")
    
    # Data arguments
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="Path to the dataset metadata CSV.")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory for audio files.")
    parser.add_argument("--batch_size", type=int, default=40, 
                        help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="Number of data loading workers.")
    parser.add_argument("--val_split_ratio", type=float, default=0.2,
                        help="Ratio of unique speakers to reserve for validation set.")
    parser.add_argument('--output_dir', type=str, default='.')

    parser.add_argument('--save_interval', type=int, default=5, 
                    help='Save every N epochs')
    
    parser.add_argument('--log_interval', type=int, default=20, help="Interval for logging training batch loss")

    
    parser.add_argument('--evaluate_initial_detector', action='store_true', help="Evaluate the initial loaded detector")

    # Audio processing arguments
    parser.add_argument("--n_mels", type=int, default=64, 
                        help="Number of Mel bins for spectrograms.")
    parser.add_argument("--max_duration", type=float, default=5.0, 
                        help="Maximum audio duration in seconds for fixed-size spectrograms.")
    
    # Model architecture arguments (matching your working script)
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout rate for the classifier.")
    parser.add_argument("--initial_channels", type=int, default=32, 
                        help="Initial channels in the ResNet backbone.")
    parser.add_argument("--resnet_channels", nargs="+", type=int, default=[32, 64, 128, 256],
                        help="List of channel sizes for each ResNet block stage.")
    parser.add_argument("--resnet_blocks", nargs="+", type=int, default=[1, 1, 1, 1],
                        help="Number of residual blocks in each ResNet stage.")
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for the final classifier layer.")

    # Mask generator arguments
    parser.add_argument("--use_learned_mask_generator", action='store_true', 
                        help="Use learned UNet for mask generation. If false, uses gradient-based saliency.")
    parser.add_argument("--train_mask_generator", action='store_true', 
                        help="Train the learned mask generator.")
    parser.add_argument("--lr_mask_gen", type=float, default=1e-4, 
                        help="Learning rate for mask generator.")
    parser.add_argument('--mask_gen_unet_base_c', type=int, default=32, help="Base channels for mask generator U-Net")

    parser.add_argument("--epochs_mask_gen", type=int, default=1, 
                        help="Number of epochs to train mask generator.")
    parser.add_argument("--lambda_budget", type=float, default=0.1, 
                        help="Weight for mask density budget loss (for learned mask gen).")
    parser.add_argument("--target_mask_density", type=float, default=0.1, 
                        help="Target mean density for the mask (for learned mask gen).")
    
    # Refiner arguments
    parser.add_argument("--train_refiner", action='store_true', 
                        help="Train the adversarial refiner (region reconstructor).")
    parser.add_argument("--lr_refiner", type=float, default=1e-4, 
                        help="Learning rate for refiner.")
    parser.add_argument('--refiner_unet_base_c', type=int, default=16)

    parser.add_argument("--epochs_refiner", type=int, default=10, 
                        help="Number of epochs to train refiner.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, 
                        help="Threshold for binarizing mask in refiner's forward pass.")
    parser.add_argument("--refiner_epsilon_clip", type=float, default=None, 
                        help="Optional L_inf clip for reconstructor perturbation (e.g., 0.1).")
    parser.add_argument("--realism_loss_weight", type=float, default=0.0, 
                        help="Weight for realism loss in refiner training (e.g., 0.5).")
    parser.add_argument('--evaluate_refiner_independently', action='store_true', help="Evaluate the refiner after its training")


    # Hardening arguments
    parser.add_argument("--adversarial_hardening_detector", action='store_true', 
                        help="Perform adversarial hardening of the detector.")
    parser.add_argument("--lr_detector_hardening", type=float, default=5e-5, 
                        help="Learning rate for detector hardening.")
    parser.add_argument("--epochs_hardening", type=int, default=2, 
                        help="Number of epochs for detector hardening.")
    parser.add_argument('--hardening_keep_original_fake_frac', type=float, default=0.1, help="Fraction of original fakes to keep during hardening")

    
    args = parser.parse_args()
    run_adversarial_pipeline_refactored(args)


if __name__ == '__main__':
    main()
