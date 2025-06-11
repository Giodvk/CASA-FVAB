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
from DataBalancingDeepSeek import train_speaker, test_speaker
from DeepLearningModel import DeepfakeClassifier, AudioConfig, AudioProcessor, DeepfakeDataset, collate_fn_skip_none


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')






class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dimensions if necessary (e.g., if x2 is slightly larger due to odd dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in=1, n_channels_out=1, bilinear=True, base_c=64):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.base_c = base_c

        self.inc = UNetConvBlock(n_channels_in, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor)
        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        self.outc = OutConv(base_c, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# --- Region Reconstructor (New) ---
class RegionReconstructor(nn.Module):
    """
    A UNet-based network that reconstructs/modifies regions of the spectrogram.
    Used within the AdversarialRefiner.
    """
    def __init__(self, n_channels_in=1, n_channels_out=1, unet_base_c=32):
        super().__init__()
        # Input to UNet will be the original spectrogram (or a masked version)
        # Output will be the reconstructed/modified version of the spectrogram
        self.reconstructor_network = UNet(n_channels_in=n_channels_in, n_channels_out=n_channels_out, base_c=unet_base_c)

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        """
        x_mel: original mel spectrogram (B, C, H, W)
        Returns: reconstructed/modified mel spectrogram (B, C, H, W)
        """
        return self.reconstructor_network(x_mel)

# --- Adversarial Refiner (Revised: Mask and Reconstruct) ---
class AdversarialRefiner(nn.Module):
    """
    Orchestrates the mask generation, region reconstruction, and combines them.
    Aims to transform fake samples into 'real' samples as perceived by the detector.
    """
    def __init__(self, detector_model: nn.Module, region_reconstructor_model: nn.Module, critical_mask_generator_fn,
                 mask_threshold: float = 0.5, epsilon_clip: Optional[float] = None):
        super().__init__()
        self.detector = detector_model
        self.region_reconstructor = region_reconstructor_model
        self.critical_mask_generator_fn = critical_mask_generator_fn # Function or nn.Module
        self.mask_threshold = mask_threshold # For binarizing probabilistic masks
        self.epsilon_clip = epsilon_clip # Optional: L_inf bound on perturbation for reconstructed regions

    def forward(self, x_fake_original_mel: torch.Tensor) -> torch.Tensor:
        """
        x_fake_original_mel: batch of mel spectrograms (B, H, W) or (B, C, H, W)
        Returns: refined mel spectrograms (B, H, W) or (B, C, H, W)
        """
        self.detector.eval() # Detector is fixed during refiner's forward pass
        # Mask generator and reconstructor modes are set in training/evaluation loops
        # self.critical_mask_generator_fn.eval() # If it's an nn.Module
        # self.region_reconstructor.train() # Set in train_epoch_adversarial_refiner

        # Ensure input has channel dim: B, C, H, W for internal processing
        if x_fake_original_mel.dim() == 3:
            x_mel_bchw = x_fake_original_mel.unsqueeze(1)
        else: # Assuming B,C,H,W
            x_mel_bchw = x_fake_original_mel

        # 1. Generate Critical Mask
        # Gradients for mask generation are not backpropagated through the reconstructor
        with torch.no_grad():
            if isinstance(self.critical_mask_generator_fn, nn.Module):
                # Learned mask generator (UNet) expects B,H,W or B,C,H,W -> outputs B,H,W or B,C_out,H,W
                mask_prob = self.critical_mask_generator_fn(x_fake_original_mel)
                # Ensure mask_prob has a channel dimension matching x_mel_bchw
                if mask_prob.dim() == 3 and x_mel_bchw.shape[1] == 1: # B,H,W mask for B,1,H,W input
                    mask_prob = mask_prob.unsqueeze(1) # Make it B,1,H,W
                elif mask_prob.dim() != 4: # Should be B,C,H,W or B,H,W that can be unsqueezed
                    raise ValueError(f"Mask generator output unexpected shape: {mask_prob.shape}")
            else: # Gradient-based function (find_adversarial_mask_gradient_based)
                logger.info("Using Gradient-based function inside Adversarial Refiner instead of Learned mask generator ")
                # find_adversarial_mask_gradient_based expects B,C,H,W and returns B,C,H,W
                # It needs `requires_grad_(True)` on its input, which is handled inside the function
                # We pass a clone here to ensure it doesn't affect the original x_mel_bchw's graph.
                mask_prob = self.critical_mask_generator_fn(self.detector, x_mel_bchw.clone().detach().requires_grad_(True))
                mask_prob = mask_prob.detach() # Detach the mask from its computation graph

            # Ensure mask_prob channels match input channels for element-wise product
            if mask_prob.shape[1] != x_mel_bchw.shape[1]:
                if mask_prob.shape[1] == 1 and x_mel_bchw.shape[1] > 1: # Single channel mask for multi-channel input
                    mask_prob = mask_prob.repeat(1, x_mel_bchw.shape[1], 1, 1)
                else:
                    raise ValueError(f"Channel mismatch: Input {x_mel_bchw.shape}, Mask {mask_prob.shape}")

            # Binarize the mask based on threshold
            binary_mask = (mask_prob > self.mask_threshold).float()

        # 2. Get Reconstructed/Modified version from RegionReconstructor
        # The reconstructor network sees the whole original spectrogram and learns to modify it
        reconstructed_mel_bchw = self.region_reconstructor(x_mel_bchw)

        # Optional: Clip the perturbation if epsilon_clip is defined (L_inf bound)
        if self.epsilon_clip is not None:
            perturbation = reconstructed_mel_bchw - x_mel_bchw
            perturbation = torch.clamp(perturbation, -self.epsilon_clip, self.epsilon_clip)
            reconstructed_mel_bchw = x_mel_bchw + perturbation

        # 3. Combine original and reconstructed parts using the binary mask
        # X_refined = X_orig * (1-M) + X_reconstructed_by_model * M
        # This means original pixels are kept where mask is 0, and reconstructed pixels where mask is 1.
        refined_x_bchw = x_mel_bchw * (1 - binary_mask) + reconstructed_mel_bchw * binary_mask
        
        # Clip to valid spectrogram range (using original input's min/max for dynamic range)
        # This helps prevent unrealistic values in the refined spectrogram.
        min_val, max_val = x_mel_bchw.min(), x_mel_bchw.max()
        refined_x_bchw = torch.clamp(refined_x_bchw, min_val, max_val)
        
        # Return to original input dimension if it was 3D (B,H,W)
        return refined_x_bchw.squeeze(1) if x_fake_original_mel.dim() == 3 and refined_x_bchw.shape[1] == 1 else refined_x_bchw


def train_epoch_adversarial_refiner(refiner_model: AdversarialRefiner, detector_model: nn.Module, train_loader: DataLoader, optimizer_reconstructor: optim.Optimizer, criterion: nn.Module, device: torch.device, current_epoch: int, num_epochs: int, realism_loss_weight: float = 0.0):
    """
    Trains the RegionReconstructor part of the AdversarialRefiner.
    The goal is to make the detector classify 'fake' samples as 'real'.
    """
    refiner_model.train() # Sets refiner_model.region_reconstructor to train mode
    detector_model.eval() # Detector is fixed and used for adversarial loss

    total_adv_loss = 0.0
    total_realism_loss = 0.0
    num_processed_samples = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None: # Skip batches that failed to load
            continue
        features_dict, labels = batch_data
        
        # We only refine 'fake' samples (label 0)
        fake_mel_spectrograms = features_dict["mel"][labels == 0].to(device) # Shape: (B_fake, H, W)
        if fake_mel_spectrograms.size(0) == 0: # Skip if no fake samples in batch
            continue

        optimizer_reconstructor.zero_grad()
        
        # Generate refined output from the refiner model
        # refiner_model expects (B, H, W) or (B, C, H, W) and returns the same
        refined_output = refiner_model(fake_mel_spectrograms) 

        if batch_idx==3:visualize_masked_pair(
            x_orig=fake_mel_spectrograms[0], 
            x_masked=refined_output[0], 
            masks_prob=refined_output[0], 
            sample_idx=0
        )
        
        # Ensure refined_output has channel dim for detector (B, C, H, W)
        if refined_output.dim() == 3: # B,H,W
            refined_for_detector = refined_output.unsqueeze(1) # B,1,H,W
        else: # B,C,H,W
            refined_for_detector = refined_output

        # Get detector's logits on the refined samples
        refined_logits = detector_model(refined_for_detector)
        
        # Adversarial Loss: We want the detector to think x_refined is REAL (class 1).
        # Original labels: 0 for fake, 1 for real.
        # So, for refined fake samples, the target label for the detector is 1 (real).
        target_labels_real = torch.ones(refined_logits.size(0), dtype=torch.long, device=device)
        adv_loss = criterion(refined_logits, target_labels_real) # CrossEntropyLoss

        current_loss = adv_loss

        # Optional Realism Loss: Encourages the refined output to stay close to the original fake.
        if realism_loss_weight > 0:
            # Ensure shapes match for L1 loss (remove channel dim if present for 1-channel input)
            if refined_output.dim() == 4 and refined_output.shape[1] == 1 and fake_mel_spectrograms.dim() == 3:
                refined_output_for_realism = refined_output.squeeze(1)
            else:
                refined_output_for_realism = refined_output

            if refined_output_for_realism.shape == fake_mel_spectrograms.shape:
                realism_loss = F.l1_loss(refined_output_for_realism, fake_mel_spectrograms)
                current_loss += realism_loss_weight * realism_loss
                total_realism_loss += realism_loss.item() * fake_mel_spectrograms.size(0)
            else:
                logger.warning(f"Skipping realism loss due to shape mismatch: refined {refined_output_for_realism.shape}, original {fake_mel_spectrograms.shape}")

        current_loss.backward()
        optimizer_reconstructor.step()

        total_adv_loss += adv_loss.item() * fake_mel_spectrograms.size(0)
        num_processed_samples += fake_mel_spectrograms.size(0)

        if batch_idx % 20 == 0:
            log_msg = f"Refiner Train Epoch {current_epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Adv Loss: {adv_loss.item():.4f}"
            if realism_loss_weight > 0 and 'realism_loss' in locals() and realism_loss is not None :
                log_msg += f" | Realism Loss: {realism_loss.item():.4f}"
            logger.info(log_msg)

    avg_adv_loss = total_adv_loss / num_processed_samples if num_processed_samples > 0 else 0
    avg_realism_loss = total_realism_loss / num_processed_samples if num_processed_samples > 0 and realism_loss_weight > 0 else 0
    
    log_summary = f"Refiner Train Epoch {current_epoch}/{num_epochs} | Avg Adv Loss: {avg_adv_loss:.4f}"
    if realism_loss_weight > 0:
        log_summary += f" | Avg Realism Loss: {avg_realism_loss:.4f}"
    logger.info(log_summary)
    return avg_adv_loss, avg_realism_loss


# --- Critical Mask Generator (Learned UNet-based) ---
class CriticalMaskGenerator(nn.Module):
    """
    A UNet-based network that learns to generate a probabilistic mask
    indicating critical regions for the detector.
    """
    def __init__(self, mask_unet_channels_out=1, unet_base_c=32):
        super().__init__()
        # Output channel is 1 for a single-channel mask (e.g., for Mel spectrogram)
        self.mask_network = UNet(n_channels_in=1, n_channels_out=mask_unet_channels_out, base_c=unet_base_c)

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        """
        x_mel: Input mel spectrogram (B, H, W) or (B, C, H, W).
        Returns: Probabilistic mask (B, H, W) or (B, C_out, H, W) in [0,1].
        """
        if x_mel.dim() == 3: # Add channel dimension if missing
            x_mel_reshaped = x_mel.unsqueeze(1) # (B, 1, H, W)
        else: # Assuming B,C,H,W
            x_mel_reshaped = x_mel
            
        mask_logits = self.mask_network(x_mel_reshaped) # Output B,C_out,H,W (logits)
        mask_prob = torch.sigmoid(mask_logits) # Convert logits to probabilities [0,1]
        
        # Return to original input dimension if it was 3D and output channel is 1
        if x_mel.dim() == 3 and mask_prob.shape[1] == 1:
            return mask_prob.squeeze(1) # (B, H, W)
        return mask_prob # (B, C_out, H, W)
def visualize_masked_pair(x_orig: torch.Tensor,
                          x_masked: torch.Tensor,
                          masks_prob: torch.Tensor,
                          sample_idx: int = 0):
    """
    x_orig / x_masked: torch.Tensor of shape (B, 1, H, W) or (B, H, W)
    masks_prob:         torch.Tensor of same spatial shape (B, 1, H, W) or (B, H, W)
                      OR None, in which case we skip the mask‐plot.
    sample_idx:         which element of batch to plot
    """
    def to_np(t):
        # If t is None, we’ll never call this. Only call to_np if t is a real tensor.
        if t.dim() == 4:  # (B, C, H, W)
            t = t[sample_idx, 0]
        elif t.dim() == 3:  # (B, H, W)
            t = t[sample_idx]
        return t.detach().cpu().numpy()

    orig = to_np(x_orig)
    masked = to_np(x_masked)

    if masks_prob is not None:
        mask = to_np(masks_prob)
        ncols = 3
    else:
        mask = None
        ncols = 2  # we’ll only plot “orig” and “masked”

    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Plot Original
    im0 = axs[0].imshow(orig, origin='lower', aspect='auto')
    axs[0].set_title("Original Mel")
    plt.colorbar(im0, ax=axs[0], format="%+2.0f dB")

    if mask is not None:
        # Plot Mask Probability
        im1 = axs[1].imshow(mask, origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title("Mask Probability")
        plt.colorbar(im1, ax=axs[1])

        # Plot Masked‐Out Mel
        im2 = axs[2].imshow(masked, origin='lower', aspect='auto')
        axs[2].set_title("Masked Mel")
        plt.colorbar(im2, ax=axs[2])
    else:
        # Only plot “Masked Mel” in the second axis
        im1 = axs[1].imshow(masked, origin='lower', aspect='auto')
        axs[1].set_title("Refined/Masked Mel")
        plt.colorbar(im1, ax=axs[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig("spectrogram_comparison.png")
    plt.close(fig)
def train_epoch_critical_mask_generator(mask_gen_model: CriticalMaskGenerator, detector_model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device,
                                        lambda_budget: float = 0.1, target_mask_density: float = 0.1, current_epoch: int = 0, num_epochs: int = 10, save_path="best_mask_generator.pth"):
    """
    Trains the CriticalMaskGenerator.
    Loss: Maximize KL divergence between detector's predictions on original vs. masked input,
          while penalizing mask density to stay close to target_mask_density.
    """
    mask_gen_model.train()
    detector_model.eval() # Detector is fixed

    total_loss_kl = 0.0 # We will store KL directly (which is minimized by optimizer on -KL)
    total_loss_budget = 0.0
    total_samples = 0

    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None:
            continue
        features_dict, _ = batch_data # Labels are not used for mask generation training
        x_mels = features_dict["mel"].to(device) # (B, H, W)
        if x_mels.size(0) == 0:
            continue
        optimizer.zero_grad()

        # mask_gen_model expects (B,H,W) or (B,C,H,W) -> outputs (B,H,W) or (B,C_out,H,W)
        masks_prob = mask_gen_model(x_mels) # Probabilistic mask in [0,1]
        
        # Prepare for multiplication and detector: ensure (B,C,H,W)
        if masks_prob.dim() == 3: # (B,H,W)
            masks_prob_bchw = masks_prob.unsqueeze(1) # (B,1,H,W)
        else: # (B,C,H,W)
            masks_prob_bchw = masks_prob
        
        if x_mels.dim() == 3: # (B,H,W)
            x_mels_bchw = x_mels.unsqueeze(1) # (B,1,H,W)
        else: # (B,C,H,W)
            x_mels_bchw = x_mels
    
        # Ensure mask channels match input channels for element-wise product
        if masks_prob_bchw.shape[1] != x_mels_bchw.shape[1]:
            if masks_prob_bchw.shape[1] == 1: # Expand mask channels if mask is single-channel
                masks_prob_bchw = masks_prob_bchw.repeat(1, x_mels_bchw.shape[1], 1, 1)
            else: # Cannot reconcile, skip batch
                logger.warning(f"Mask channel {masks_prob_bchw.shape[1]} mismatch with input {x_mels_bchw.shape[1]}. Skipping batch.")
                continue

        # Apply the mask: x_masked = x_orig * (1 - mask_prob)
        # This means regions with high mask_prob are "removed" or attenuated.
        x_mels_masked = x_mels_bchw * (1 - masks_prob_bchw)
        if batch_idx==140:visualize_masked_pair(
            x_orig=x_mels_bchw, 
            x_masked=x_mels_masked, 
            masks_prob=masks_prob_bchw, 
            sample_idx=0
        )

        # Get detector's predictions on original and masked inputs
        with torch.no_grad(): # Detector's gradients are not computed for mask generator's loss
            original_logits = detector_model(x_mels_bchw)
            p_original_log = F.log_softmax(original_logits, dim=1) # Log probabilities for original
        
        masked_logits = detector_model(x_mels_masked)
        p_masked_log = F.log_softmax(masked_logits, dim=1) # Log probabilities for masked

        # KL Divergence Loss: We want to maximize the difference between original and masked predictions.
        # F.kl_div(input_log_probs, target_probs) computes D_KL(target_probs || input_log_probs)
        # To maximize D_KL(P_original || P_masked), we minimize -D_KL(P_original || P_masked)
        # So, input_log_probs = p_masked_log, target_probs = p_original_log.exp()
        loss_kl = F.kl_div(p_masked_log, p_original_log.exp(), reduction='batchmean', log_target=False)

        actual_loss_kl_neg = -loss_kl # We want to MAXIMIZE KL divergence, so minimize its negative

        # Mask Budget Loss: Penalizes deviation from target mask density
        # Using the mean of the probabilistic mask (before binarization)
        loss_budget = (masks_prob.mean() - target_mask_density)**2
        
        # Total loss for mask generator
        total_combined_loss = actual_loss_kl_neg + lambda_budget * loss_budget
        total_combined_loss.backward()
        optimizer.step()

        total_loss_kl += loss_kl.item() * x_mels.size(0) # Accumulate positive KL for reporting
        total_loss_budget += loss_budget.item() * x_mels.size(0)
        total_samples += x_mels.size(0)

        if batch_idx % 20 == 0:
            logger.info(f"MaskGen Train Epoch {current_epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | KL Div (Maximized): {loss_kl.item():.4f} | Budget Loss: {loss_budget.item():.4f}")
            
    avg_loss_kl = total_loss_kl / total_samples if total_samples > 0 else 0
    avg_loss_budget = total_loss_budget / total_samples if total_samples > 0 else 0
    

    logger.info(f"MaskGen Train Epoch {current_epoch}/{num_epochs} | Avg KL Div: {avg_loss_kl:.4f} | Avg Budget Loss: {avg_loss_budget:.4f}")
    return avg_loss_kl, avg_loss_budget


# --- Gradient-based Saliency (Algorithmic Mask Generation) ---
def find_adversarial_mask_gradient_based(detector_model: nn.Module, x_mel: torch.Tensor, target_class_idx: Optional[int] = None, use_abs: bool = True) -> torch.Tensor:
    """
    Computes a saliency map (mask) based on gradients of the detector's output w.r.t. the input.
    This mask indicates regions most critical for the detector's prediction.
    x_mel: Input mel spectrogram (B, C, H, W). Must have requires_grad_(True).
    Returns: Normalized saliency map (B, C, H, W) in [0,1].
    """
    detector_model.eval()
    
    # Ensure x_mel requires gradients
    if not x_mel.requires_grad:
        raise ValueError("Input x_mel must have requires_grad=True for gradient-based mask generation.")
    
    # Clear gradients before backward pass for this specific computation
    if x_mel.grad is not None:
        x_mel.grad.zero_()

    logits = detector_model(x_mel)

    if target_class_idx is None:
        # For mask generation, we want to find regions that are CRITICAL for the detector's
        # *current* prediction. So we maximize the score of the predicted class for the original input.
        target_score = logits.max(dim=1)[0].sum() # Sum of max logit for each sample in batch
    else:
        # Maximize the score of a specific target class (e.g., 'fake' or 'real')
        target_score = logits[:, target_class_idx].sum()
    
    # Compute gradients of the target score w.r.t. the input spectrogram
    target_score.backward(retain_graph=True) # retain_graph=True because x_mel might be part of a larger graph
    
    saliency = x_mel.grad.data # Get the gradients (saliency map)
    if use_abs:
        saliency = saliency.abs() # Use absolute gradients for saliency

    # Normalize saliency to [0,1] for use as a probability mask
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    if saliency_max > saliency_min:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    else: # Handle case where all saliency values are the same (e.g., all zero gradients)
        saliency = torch.zeros_like(saliency)

    return saliency # Returns (B, C, H, W) normalized to [0,1]


# --- Diffusion-Style Attack (Conceptual Baseline) ---
# This is an algorithmic attack, not part of the learned refiner training.
# It's included as a potential baseline or for analysis.
def diffusion_style_attack(x_orig_mel: torch.Tensor, detector_model: nn.Module, critical_mask_generator_fn,
                           steps: int = 10, max_noise_alpha: float = 0.1, device: torch.device = 'cpu', mask_threshold: float = 0.5) -> List[torch.Tensor]:
    """
    Applies a diffusion-style adversarial attack by iteratively adding noise to masked regions.
    x_orig_mel: single mel spectrogram (H,W) to be attacked.
    Returns: A list of attacked spectrograms at each step of the trajectory.
    """
    detector_model.eval()
    if isinstance(critical_mask_generator_fn, nn.Module):
        critical_mask_generator_fn.eval()

    x_curr_bhw = x_orig_mel.unsqueeze(0).to(device) # (B=1, H, W)
    x_orig_bchw = x_curr_bhw.unsqueeze(1) # (B=1, 1, H, W) for processing

    attacked_trajectory = [x_orig_mel.clone().cpu()] # Store initial (H,W) on CPU

    alphas = torch.linspace(0, max_noise_alpha, steps, device=device) # Noise scaling factors

    for i in range(1, steps):
        # Generate mask for the current attacked sample
        if isinstance(critical_mask_generator_fn, nn.Module):
            mask_prob_output = critical_mask_generator_fn(x_curr_bhw) # (B,H,W) or (B,C,H,W)
            if mask_prob_output.dim() == 4 and mask_prob_output.shape[1] == 1: # (B,1,H,W)
                mask_prob_bchw = mask_prob_output
            elif mask_prob_output.dim() == 3: # (B,H,W)
                mask_prob_bchw = mask_prob_output.unsqueeze(1) # (B,1,H,W)
            else:
                raise ValueError("Unexpected mask shape from learned generator in diffusion attack")
        else: # Gradient-based mask generator
            # Need to ensure input has requires_grad=True for gradient computation
            x_curr_bchw_for_grad = x_curr_bhw.unsqueeze(1).clone().detach().requires_grad_(True)
            mask_prob_bchw = critical_mask_generator_fn(detector_model, x_curr_bchw_for_grad) # Returns (B,1,H,W)
            mask_prob_bchw = mask_prob_bchw.detach() # Detach the mask from its computation graph

        binary_mask_bchw = (mask_prob_bchw > mask_threshold).float()

        # Add noise to the masked regions
        current_noise_component = torch.randn_like(x_orig_bchw) * alphas[i]
        
        # Apply noise only to masked regions, keep original for unmasked
        x_attacked_step_bchw = x_orig_bchw * (1 - binary_mask_bchw) + (x_orig_bchw + current_noise_component) * binary_mask_bchw
        
        # Clip to original spectrogram range
        min_val, max_val = x_orig_mel.min(), x_orig_mel.max()
        x_attacked_step_bchw = torch.clamp(x_attacked_step_bchw, min_val, max_val)
        
        attacked_trajectory.append(x_attacked_step_bchw.squeeze(0).squeeze(0).cpu()) # Store as (H,W) on CPU
        x_curr_bhw = x_attacked_step_bchw.squeeze(1).detach() # Update current sample for next iteration

    return attacked_trajectory


# --- Evaluation Functions ---

def evaluate_detector_performance(detector_model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the detector's accuracy and AUC on a given dataset.
    Returns (accuracy, auc_score).
    """
    detector_model.eval()
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_probs = [] # To collect probabilities for AUC calculation

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_data is None:
                continue
            features_dict, labels = batch_data
            x_mels = features_dict["mel"].to(device)
            labels = labels.to(device)

            if x_mels.dim() == 3: # Add channel dimension if missing
                x_mels_bchw = x_mels.unsqueeze(1) # (B,1,H,W)
            else: # (B,C,H,W)
                x_mels_bchw = x_mels
            
            outputs = detector_model(x_mels_bchw) # Logits
            probabilities = F.softmax(outputs, dim=1) # Convert logits to probabilities
            
            _, predicted = torch.max(outputs.data, 1) # Get predicted class
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy()) # Probabilities for the positive class (label 1: real/bona-fide)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    from sklearn.metrics import roc_auc_score
    try:
        # AUC requires at least two classes present in the labels
        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_probs)
        else:
            logger.warning("AUC cannot be calculated: Only one class present in labels.")
            auc_score = 0.0
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc_score = 0.0

    return accuracy, auc_score

def evaluate_mask_generator_performance(mask_gen_model: CriticalMaskGenerator, detector_model: nn.Module, data_loader: DataLoader, device: torch.device, lambda_budget: float, target_mask_density: float) -> Tuple[float, float]:
    """
    Evaluates the Mask Generator's performance (KL Divergence and Mask Density).
    Returns (avg_kl_div, avg_mask_density).
    """
    mask_gen_model.eval()
    detector_model.eval()
    total_kl_div = 0.0
    total_mask_density = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_data is None:
                continue
            features_dict, _ = batch_data # Labels not directly used for mask gen eval
            x_mels = features_dict["mel"].to(device) # (B, H, W)
            if x_mels.size(0) == 0:
                continue

            masks_prob = mask_gen_model(x_mels) # Probabilistic mask (B, H, W) or (B, C_out, H, W)
            
            # Prepare for multiplication and detector: ensure (B,C,H,W)
            if masks_prob.dim() == 3: # (B,H,W)
                masks_prob_bchw = masks_prob.unsqueeze(1) # (B,1,H,W)
            else: # (B,C,H,W)
                masks_prob_bchw = masks_prob
            
            if x_mels.dim() == 3: # (B,H,W)
                x_mels_bchw = x_mels.unsqueeze(1) # (B,1,H,W)
            else: # (B,C,H,W)
                x_mels_bchw = x_mels

            # Ensure mask channels match input channels for element-wise product
            if masks_prob_bchw.shape[1] != x_mels_bchw.shape[1]:
                if masks_prob_bchw.shape[1] == 1: # Expand mask channels
                    masks_prob_bchw = masks_prob_bchw.repeat(1, x_mels_bchw.shape[1], 1, 1)
                else:
                    logger.warning(f"Mask channel {masks_prob_bchw.shape[1]} mismatch with input {x_mels_bchw.shape[1]} during mask eval. Skipping batch.")
                    continue

            x_mels_masked = x_mels_bchw * (1 - masks_prob_bchw)

            original_logits = detector_model(x_mels_bchw)
            p_original_log = F.log_softmax(original_logits, dim=1)
            masked_logits = detector_model(x_mels_masked)
            p_masked_log = F.log_softmax(masked_logits, dim=1)
            
            loss_kl = F.kl_div(p_masked_log, p_original_log.exp(), reduction='batchmean', log_target=False)
            
            total_kl_div += loss_kl.item() * x_mels.size(0)
            total_mask_density += masks_prob.mean().item() * x_mels.size(0) # Use original masks_prob
            total_samples += x_mels.size(0)

    avg_kl_div = total_kl_div / total_samples if total_samples > 0 else 0.0
    avg_mask_density = total_mask_density / total_samples if total_samples > 0 else 0.0
    
    logger.info(f"MaskGen Val | Avg KL Div: {avg_kl_div:.4f} | Avg Mask Density: {avg_mask_density:.4f}")
    return avg_kl_div, avg_mask_density

def evaluate_adversarial_refiner_performance(refiner_model: AdversarialRefiner, detector_model: nn.Module, data_loader: DataLoader, device: torch.device, realism_loss_weight: float = 0.0) -> Tuple[float, float, float]:
    """
    Evaluates the Adversarial Refiner's performance.
    Metrics: Average adversarial loss, average realism loss, and refiner's success rate
             (percentage of refined fakes classified as 'real' by the detector).
    Returns (avg_adv_loss, avg_realism_loss, refiner_success_rate).
    """
    refiner_model.eval() # Refiner (and its reconstructor) in eval mode
    detector_model.eval()
    
    total_adv_loss = 0.0
    total_realism_loss = 0.0
    total_fake_samples_processed = 0
    correct_on_refined_fake = 0 # How many refined fakes are classified as REAL (label 1) by detector

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_data is None:
                continue
            features_dict, labels = batch_data
            
            # Only evaluate on fake samples for refiner's effectiveness
            fake_mel_spectrograms = features_dict["mel"][labels == 0].to(device)
            if fake_mel_spectrograms.size(0) == 0:
                continue
            
            total_fake_samples_processed += fake_mel_spectrograms.size(0)

            # Generate refined output
            refined_output = refiner_model(fake_mel_spectrograms)
            
            # Ensure refined_output has channel dim for detector
            if refined_output.dim() == 3:
                refined_for_detector = refined_output.unsqueeze(1)
            else:
                refined_for_detector = refined_output

            refined_logits = detector_model(refined_for_detector)
            
            # Adversarial Loss (how well refiner fools detector)
            # Target is 'real' (label 1)
            target_labels_real = torch.ones(refined_logits.size(0), dtype=torch.long, device=device)
            adv_loss = F.cross_entropy(refined_logits, target_labels_real)
            total_adv_loss += adv_loss.item() * fake_mel_spectrograms.size(0)

            # Realism Loss
            if realism_loss_weight > 0:
                # Ensure shapes match for L1 loss
                if refined_output.dim() == 4 and refined_output.shape[1] == 1 and fake_mel_spectrograms.dim() == 3:
                    refined_output_for_realism = refined_output.squeeze(1)
                else:
                    refined_output_for_realism = refined_output

                if refined_output_for_realism.shape == fake_mel_spectrograms.shape:
                    realism_loss = F.l1_loss(refined_output_for_realism, fake_mel_spectrograms)
                    total_realism_loss += realism_loss.item() * fake_mel_spectrograms.size(0)
                else:
                    logger.warning(f"Skipping realism loss in eval due to shape mismatch: refined {refined_output_for_realism.shape}, original {fake_mel_spectrograms.shape}")

            # Check how many refined samples are classified as REAL (target for refiner)
            _, predicted_refined = torch.max(refined_logits.data, 1)
            # If detector predicts 'real' (label 1), it means refiner was successful
            correct_on_refined_fake += (predicted_refined == 1).sum().item()

    avg_adv_loss = total_adv_loss / total_fake_samples_processed if total_fake_samples_processed > 0 else 0.0
    avg_realism_loss = total_realism_loss / total_fake_samples_processed if total_fake_samples_processed > 0 and realism_loss_weight > 0 else 0.0
    
    # Success rate of refiner in fooling detector (percentage of refined fakes classified as real)
    refiner_success_rate = correct_on_refined_fake / total_fake_samples_processed if total_fake_samples_processed > 0 else 0.0

    log_summary = f"Refiner Val | Avg Adv Loss: {avg_adv_loss:.4f}"
    if realism_loss_weight > 0:
        log_summary += f" | Avg Realism Loss: {avg_realism_loss:.4f}"
    log_summary += f" | Refiner Success Rate (as Real): {refiner_success_rate:.4f}"
    logger.info(log_summary)
    return avg_adv_loss, avg_realism_loss, refiner_success_rate



def run_adversarial_pipeline(args):
    """
    Orchestrates the entire adversarial deepfake detection pipeline:
    1. Data Loading with speaker-disjoint splitting
    2. Detector Initialization/Evaluation
    3. Critical Mask Generator Training (if learned)
    4. Adversarial Refiner (Region Reconstructor) Training
    5. Adversarial Hardening of the Detector
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set multiprocessing method for CUDA compatibility
    if args.num_workers > 0 and device.type == 'cuda':
        if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            logger.info("Setting multiprocessing start method to 'spawn' for CUDA compatibility with num_workers > 0.")
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError as e:
                logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA in child processes.")

    audio_conf = AudioConfig(n_mels=args.n_mels, max_duration=args.max_duration)

    # --- 1. Data Loading with Speaker-Disjoint Splitting ---
    logger.info("--- Loading and Splitting Metadata ---")
    metadata_path = Path(args.metadata_path)
    data_root_dir = Path(args.data_dir)

    if not metadata_path.exists():
        logger.error(f"Metadata CSV file not found at {args.metadata_path}. Exiting.")
        return

    full_metadata_df = pd.read_csv(metadata_path)
    if full_metadata_df.empty:
        logger.error(f"Metadata CSV file at {args.metadata_path} is empty. Exiting.")
        return

    # Ensure 'label' column exists and is valid
    if 'label' not in full_metadata_df.columns:
        logger.error("Metadata CSV must contain a 'label' column ('bona-fide' or 'spoof'). Exiting.")
        return
    if not all(label in ['bona-fide', 'spoof'] for label in full_metadata_df['label'].unique()):
        logger.error("The 'label' column must only contain 'bona-fide' or 'spoof' values. Exiting.")
        return


    # Create train and validation DataFrames based on speaker IDs
    train_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(train_speaker)].reset_index(drop=True)
    val_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(test_speaker)].reset_index(drop=True)

    # Log data statistics
    logger.info(f"Training samples: {len(train_metadata_df)}")
    logger.info(f"Validation samples: {len(val_metadata_df)}")
    
    if not train_metadata_df.empty:
        train_label_counts = train_metadata_df['label'].value_counts()
        logger.info(f"Training label balance: {train_label_counts.to_dict()}")
    
    if not val_metadata_df.empty:
        val_label_counts = val_metadata_df['label'].value_counts()
        logger.info(f"Validation label balance: {val_label_counts.to_dict()}")

    # Create datasets and dataloaders
    processor = AudioProcessor(audio_conf)
    train_dataset = DeepfakeDataset(data_root_dir, train_metadata_df, processor, augment=True)
    val_dataset = DeepfakeDataset(data_root_dir, val_metadata_df, processor, augment=False)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty after filtering. Please check data_dir and metadata_path.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_skip_none,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_skip_none,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        raise ValueError("Validation dataset is empty. Please provide a non‐empty split.")



    logger.info("DataLoaders created.")
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset) if val_dataset else 0}")


    detector = DeepfakeClassifier(
        num_classes=2, # Fixed to 2
        config=audio_conf,
        dropout=args.dropout,
        initial_channels=args.initial_channels,
        resnet_channels=args.resnet_channels,
        resnet_blocks=args.resnet_blocks,
        classifier_hidden_dim=args.classifier_hidden_dim
    )  

# 2. Load state_dict
    detector.load_state_dict(torch.load("best_model_deepfake.pth"))

    # 3. Send to device and set eval mode
    detector.to(device)
    detector.eval()
    # --- 3. Initialize and Train Critical Mask Generator (if learned) ---
    critical_mask_gen_model = None
    if args.use_learned_mask_generator:
        logger.info("--- Initializing/Loading Learned Critical Mask Generator ---")
        critical_mask_gen_model = CriticalMaskGenerator(unet_base_c=32).to(device)
        if args.train_mask_generator:
            logger.info("--- Training Learned Critical Mask Generator ---")
            optimizer_mask_gen = optim.Adam(critical_mask_gen_model.parameters(), lr=args.lr_mask_gen)
            for epoch in range(args.epochs_mask_gen):
                avg_kl_div_train, avg_budget_loss_train = train_epoch_critical_mask_generator(
                    critical_mask_gen_model, detector, train_loader, optimizer_mask_gen, device, 
                    args.lambda_budget, args.target_mask_density, epoch, args.epochs_mask_gen
                )
                val_kl_div, val_mask_density = evaluate_mask_generator_performance(
                    critical_mask_gen_model, detector, val_loader, device, 
                    args.lambda_budget, args.target_mask_density
                )
                logger.info(f"MaskGen Epoch {epoch} | Train KL Div: {avg_kl_div_train:.4f}, Train Budget Loss: {avg_budget_loss_train:.4f}")
                logger.info(f"MaskGen Epoch {epoch} | Val KL Div: {val_kl_div:.4f}, Val Mask Density: {val_mask_density:.4f}")
            logger.info("Learned Critical Mask Generator training finished.")
        else:
            logger.info("Skipping Learned Critical Mask Generator training (assuming pre-trained).")
        mask_generator_fn_for_refiner = critical_mask_gen_model
    else:
        logger.info("--- Using Gradient-Based Saliency for Critical Regions ---")
        mask_generator_fn_for_refiner = find_adversarial_mask_gradient_based

    # --- 4. Train Adversarial Refiner (Region Reconstructor) ---
    adv_refiner = None
    if args.train_refiner:
        logger.info("--- Training Adversarial Refiner (Region Reconstructor) ---")
        region_reconstructor = RegionReconstructor(n_channels_in=1, n_channels_out=1, unet_base_c=32).to(device)
        adv_refiner = AdversarialRefiner(
            detector, region_reconstructor, mask_generator_fn_for_refiner,
            mask_threshold=args.mask_threshold, epsilon_clip=args.refiner_epsilon_clip
        ).to(device)
        
        optimizer_reconstructor = optim.Adam(region_reconstructor.parameters(), lr=args.lr_refiner)
        criterion_adv = nn.CrossEntropyLoss()

        for epoch in range(args.epochs_refiner):
            avg_adv_loss_train, avg_realism_loss_train = train_epoch_adversarial_refiner(
                adv_refiner, detector, train_loader, optimizer_reconstructor, criterion_adv, device, 
                epoch, args.epochs_refiner, args.realism_loss_weight
            )
            val_adv_loss, val_realism_loss, refiner_success_rate = evaluate_adversarial_refiner_performance(
                adv_refiner, detector, val_loader, device, args.realism_loss_weight
            )
            logger.info(f"Refiner Epoch {epoch} | Train Adv Loss: {avg_adv_loss_train:.4f}, Train Realism Loss: {avg_realism_loss_train:.4f}")
            logger.info(f"Refiner Epoch {epoch} | Val Adv Loss: {val_adv_loss:.4f}, Val Realism Loss: {val_realism_loss:.4f}, Refiner Success Rate: {refiner_success_rate:.4f}")
        logger.info("Adversarial Refiner (Region Reconstructor) training finished.")
    else:
        logger.info("Skipping Adversarial Refiner training.")

    # --- 5. Adversarial Hardening of the Detector ---
    if args.adversarial_hardening and adv_refiner:
        logger.info("--- Adversarially Hardening Detector ---")
        adv_refiner.eval()
        adv_refiner.region_reconstructor.eval()
        

        optimizer_detector_hardened = optim.AdamW(detector.parameters(), lr=args.lr_detector_hardening)
        criterion_detector_hardened = nn.CrossEntropyLoss()

        best_clean_auc = 0.0
        best_refined_acc = 0.0
        best_epoch = -1
        best_model_state = None

        for epoch in range(args.epochs_hardening):
            detector.train()
            total_hardening_loss = 0.0
            num_hardening_samples = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_data is None:
                    continue
                features_dict, labels = batch_data
                x_mels = features_dict["mel"].to(device)
                labels = labels.to(device)

                optimizer_detector_hardened.zero_grad()

                # Separate real and fake samples
                fake_mels_orig = x_mels[labels == 0]
                real_mels_orig = x_mels[labels == 1]
                fake_labels_orig = labels[labels == 0]
                real_labels_orig = labels[labels == 1]

                # Skip batch if empty (no real or fake samples)
                if fake_mels_orig.size(0) == 0 and real_mels_orig.size(0) == 0:
                    continue

                # Ensure real samples have channel dimension for concatenation
                if real_mels_orig.dim() == 3:
                    real_mels_orig_bchw = real_mels_orig.unsqueeze(1)
                else:
                    real_mels_orig_bchw = real_mels_orig

                batch_size_fakes = fake_mels_orig.size(0)
                if batch_size_fakes > 0:
                    # Preserve 10% of original fakes (unrefined) to prevent catastrophic forgetting
                    keep_frac = 0.1
                    n_keep = int(keep_frac * batch_size_fakes)
                    if n_keep == 0:  # Ensure at least 1 sample if fraction rounds to zero
                        n_keep = 1
                    
                    # Randomly select which fakes to keep unrefined
                    indices = torch.randperm(batch_size_fakes)
                    keep_indices = indices[:n_keep]
                    refine_indices = indices[n_keep:]
                    
                    fake_mels_orig_keep = fake_mels_orig[keep_indices]    # Unrefined fakes
                    fake_mels_orig_refine = fake_mels_orig[refine_indices]  # Fakes to refine

                    # Generate adversarial examples for refined portion
                    with torch.no_grad():
                        adv_fake_refine = adv_refiner(fake_mels_orig_refine)

                    # Adjust dimensions for refined fakes
                    if adv_fake_refine.dim() == 3:
                        adv_fake_refine_bchw = adv_fake_refine.unsqueeze(1)
                    else:
                        adv_fake_refine_bchw = adv_fake_refine

                    # Adjust dimensions for unrefined fakes
                    if fake_mels_orig_keep.dim() == 3:
                        fake_mels_orig_keep_bchw = fake_mels_orig_keep.unsqueeze(1)
                    else:
                        fake_mels_orig_keep_bchw = fake_mels_orig_keep

                    # Combine: reals + refined fakes + unrefined fakes
                    combined_inputs = torch.cat([
                        real_mels_orig_bchw,
                        adv_fake_refine_bchw,
                        fake_mels_orig_keep_bchw
                    ], dim=0)

                    # Create labels: reals=1, refined=0, unrefined=0
                    combined_labels = torch.cat([
                        real_labels_orig,
                        torch.zeros(len(refine_indices), device=device, dtype=labels.dtype),
                        torch.zeros(len(keep_indices), device=device, dtype=labels.dtype)
                    ])
                
                else:  # No fakes in this batch
                    combined_inputs = real_mels_orig_bchw
                    combined_labels = real_labels_orig

                outputs = detector(combined_inputs)
                loss = criterion_detector_hardened(outputs, combined_labels)
                
                loss.backward()
                optimizer_detector_hardened.step()

                total_hardening_loss += loss.item() * combined_inputs.size(0)
                num_hardening_samples += combined_inputs.size(0)

                if batch_idx % 20 == 0:
                    logger.info(f"Hardening Train Epoch {epoch}/{args.epochs_hardening} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            avg_hardening_loss = total_hardening_loss / num_hardening_samples if num_hardening_samples > 0 else 0.0
            logger.info(f"Hardening Epoch {epoch} | Avg Loss: {avg_hardening_loss:.4f}")

            # Evaluate hardened detector
            logger.info(f"--- Evaluating Hardened Detector Performance (Epoch {epoch}) ---")
            hardened_acc_clean, hardened_auc_clean = evaluate_detector_performance(detector, val_loader, device)
            logger.info(f"Hardened Detector Val (Clean) Accuracy: {hardened_acc_clean:.4f}, AUC: {hardened_auc_clean:.4f}")
            
            # Evaluate on refined fake samples
            val_fake_metadata_df = val_metadata_df[val_metadata_df['label'] == 'spoof']
            if not val_fake_metadata_df.empty:
                val_fake_dataset = DeepfakeDataset(data_root_dir, val_fake_metadata_df, processor, augment=False)
                val_fake_loader = DataLoader(
                    val_fake_dataset, batch_size=args.batch_size, shuffle=False, 
                    collate_fn=collate_fn_skip_none, num_workers=args.num_workers
                )

                total_refined_fake_correct = 0
                total_refined_fake_samples = 0
                
                with torch.no_grad():
                    for batch_data_fake in val_fake_loader:
                        if batch_data_fake is None:
                            continue
                        features_dict_fake, labels_fake = batch_data_fake
                        fake_mels_val = features_dict_fake["mel"].to(device)
                        
                        if fake_mels_val.size(0) == 0:
                            continue

                        refined_val_fakes = adv_refiner(fake_mels_val)
                        if refined_val_fakes.dim() == 3:
                            refined_val_fakes_bchw = refined_val_fakes.unsqueeze(1)
                        else:
                            refined_val_fakes_bchw = refined_val_fakes

                        outputs_refined = detector(refined_val_fakes_bchw)
                        _, predicted_refined = torch.max(outputs_refined.data, 1)
                        
                        total_refined_fake_samples += labels_fake.size(0)
                        total_refined_fake_correct += (predicted_refined == 0).sum().item()
                
                hardened_acc_on_refined = total_refined_fake_correct / total_refined_fake_samples if total_refined_fake_samples > 0 else 0.0
                logger.info(f"Hardened Detector Val (on Refined Fakes) Accuracy: {hardened_acc_on_refined:.4f}")


                is_best_clean = hardened_auc_clean > best_clean_auc
                is_best_refined = hardened_acc_on_refined > best_refined_acc
                 # Update best metrics
                if is_best_clean:
                    best_clean_auc = hardened_auc_clean
                    best_refined_acc = hardened_acc_on_refined
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(detector.state_dict())
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'clean_auc': best_clean_auc,
                        'refined_acc': best_refined_acc,
                        'optimizer_state_dict': optimizer_detector_hardened.state_dict(),
                    },'best_hardened_detector.pth')
                    
                    logger.info(f"🔥 New best model! Epoch {epoch} | Clean AUC: {best_clean_auc:.4f} | Refined Acc: {best_refined_acc:.4f}")



            # Final save after hardening completes
            logger.info(f"Hardening complete. Best model from epoch {best_epoch} with AUC: {best_clean_auc:.4f}")
            torch.save({
                'epoch': args.epochs_hardening,
                'model_state_dict': detector.state_dict(),
                'clean_auc': hardened_auc_clean,
                'refined_acc': hardened_acc_on_refined,
            }, os.path.join(args.output_dir, 'final_hardened_detector.pth'))

            # Load best model for further use if desired
            if best_model_state:
                detector.load_state_dict(best_model_state)
                logger.info(f"Loaded best hardened detector from epoch {best_epoch}")

        logger.info("Detector adversarial hardening finished.")

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
    parser.add_argument("--epochs_refiner", type=int, default=10, 
                        help="Number of epochs to train refiner.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, 
                        help="Threshold for binarizing mask in refiner's forward pass.")
    parser.add_argument("--refiner_epsilon_clip", type=float, default=None, 
                        help="Optional L_inf clip for reconstructor perturbation (e.g., 0.1).")
    parser.add_argument("--realism_loss_weight", type=float, default=0.0, 
                        help="Weight for realism loss in refiner training (e.g., 0.5).")

    # Hardening arguments
    parser.add_argument("--adversarial_hardening", action='store_true', 
                        help="Perform adversarial hardening of the detector.")
    parser.add_argument("--lr_detector_hardening", type=float, default=5e-5, 
                        help="Learning rate for detector hardening.")
    parser.add_argument("--epochs_hardening", type=int, default=2, 
                        help="Number of epochs for detector hardening.")
    
    args = parser.parse_args()
    run_adversarial_pipeline(args)


if __name__ == '__main__':
    main()