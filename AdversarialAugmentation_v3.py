


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
from AdversarialAugmentation import AdversarialRefiner, RegionReconstructor, CriticalMaskGenerator, evaluate_detector_performance, evaluate_mask_generator_performance, find_adversarial_mask_gradient_based, train_epoch_critical_mask_generator, evaluate_adversarial_refiner_performance, visualize_masked_pair
from DataBalancingDeepSeek import train_speaker, test_speaker
from DeepLearningModel import DeepfakeClassifier, AudioConfig, AudioProcessor, DeepfakeDataset, collate_fn_skip_none

# --- Main Co-Training Pipeline ---
from AdversarialAugmentation import evaluate_detector_performance, find_adversarial_mask_gradient_based
from AdversarialAugmentation_v2 import PatchDiscriminator, adversarially_harden_detector_on_mask, initialize_detector, load_and_prepare_data, setup_environment_and_config, train_epoch_adversarial_refiner_gan



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_mask_generator(args, device) -> Optional[CriticalMaskGenerator]:
    if not args.use_learned_mask_generator:
        return None
    logger.info("--- Initializing Learned Critical Mask Generator ---")
    mask_gen_model = CriticalMaskGenerator(unet_base_c=args.mask_gen_unet_base_c).to(device)
    # Add loading pretrained weights if path provided in args
    # if args.mask_gen_pretrained_path and Path(args.mask_gen_pretrained_path).exists(): ...
    return mask_gen_model

def initialize_refiner_and_discriminator(args, detector_model, mask_gen_fn_or_model, device) -> Tuple[AdversarialRefiner, Optional[PatchDiscriminator]]:
    logger.info("--- Initializing Adversarial Refiner and Patch Discriminator ---")
    region_reconstructor = RegionReconstructor(
        n_channels_in=1, n_channels_out=1, unet_base_c=args.refiner_unet_base_c
    ).to(device)
    
    adv_refiner = AdversarialRefiner(
        detector_model, region_reconstructor, mask_gen_fn_or_model,
        mask_threshold=args.mask_threshold, epsilon_clip=args.refiner_epsilon_clip
    ).to(device)

    patch_discriminator = None
    if args.use_gan_for_refiner: # New arg
        patch_discriminator = PatchDiscriminator(
            n_channels_in=1, base_c=args.patch_disc_base_c, n_layers=args.patch_disc_n_layers # New args
        ).to(device)
    return adv_refiner, patch_discriminator


def run_co_training_pipeline(args):
    # --- 0. Setup ---
    device, audio_conf = setup_environment_and_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Loading ---
    try:
        train_loader, val_loader, val_metadata_df, processor, data_root_dir = load_and_prepare_data(args, audio_conf)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load data: {e}"); return

    # --- 2. Model & Optimizer Initialization ---

    detector_model = initialize_detector(args, audio_conf, device) # f_theta
    
    mask_gen_model = None # g_psi (if learned)
    actual_mask_generator_fn = None # This will be passed to AdversarialRefiner
    if args.use_learned_mask_generator:
        mask_gen_model = initialize_mask_generator(args, device)
        actual_mask_generator_fn = mask_gen_model # Pass the nn.Module instance
    else:
        actual_mask_generator_fn = find_adversarial_mask_gradient_based # Pass the function

    refiner_model, patch_discriminator_model = initialize_refiner_and_discriminator(
        args, detector_model, actual_mask_generator_fn, device
    ) # r_xi (in refiner_model) and D_phi

    # Optimizers
    optimizer_detector = optim.AdamW(detector_model.parameters(), lr=args.lr_detector_hardening) # Initial LR for hardening
    optimizer_mask_gen = optim.Adam(mask_gen_model.parameters(), lr=args.lr_mask_gen) if mask_gen_model else None
    optimizer_reconstructor = optim.Adam(refiner_model.region_reconstructor.parameters(), lr=args.lr_refiner)
    optimizer_discriminator = optim.Adam(patch_discriminator_model.parameters(), lr=args.lr_patch_disc) if patch_discriminator_model else None # New arg: lr_patch_disc

    # Criterions
    criterion_detector_ce = nn.CrossEntropyLoss()
    # criterion_mask_gen_kl_budget (handled internally in train_epoch_mask_generator for now)

    # --- Initial Evaluation (Optional) ---
    if args.evaluate_initial_detector and val_loader:
        logger.info("--- Evaluating Initial Detector ---")
        acc, auc = evaluate_detector_performance(detector_model, val_loader, device)
        logger.info(f"Initial Detector Val Acc: {acc:.4f}, AUC: {auc:.4f}")

    # --- 3. Co-Training Loop ---
    logger.info("--- Starting Ping-Pong Co-Training ---")
    best_detector_auc_val = 0.0 # For checkpointing detector
    x=True
    for meta_epoch in range(1, args.co_training_meta_epochs + 1):
        logger.info(f"--- Co-Training Meta Epoch {meta_epoch}/{args.co_training_meta_epochs} ---")
        

        if args.use_learned_mask_generator and args.train_mask_generator and mask_gen_model and x==True:
            x=False #TO CHANGE, solo per test
            logger.info(f"  Cycle {meta_epoch}: Training Mask Generator (g_psi)...")
            for sub_epoch in range(1, args.mask_gen_epochs_per_meta + 1):
                train_epoch_critical_mask_generator(
                    mask_gen_model, detector_model, train_loader, optimizer_mask_gen, device,
                      args.lambda_budget_mask, args.target_mask_density_mask, sub_epoch, args.mask_gen_epochs_per_meta+1# New args
                )

        # A. Train Refiner (r_xi) and Patch Discriminator (D_phi) if GAN is used
        if args.train_refiner and args.use_gan_for_refiner:
            logger.info(f"  Cycle {meta_epoch}: Training Refiner (r_xi) and Patch Discriminator (D_phi)...")
            for sub_epoch in range(1, args.refiner_gan_epochs_per_meta + 1):
                # The train_epoch_adversarial_refiner_gan function is from the previous step
                # Make sure it's correctly defined and imported.
                # For now, assume it's available in the global scope or properly imported.
                train_epoch_adversarial_refiner_gan( # Defined in previous response
                    refiner_model, detector_model, patch_discriminator_model, train_loader,
                    optimizer_reconstructor, optimizer_discriminator, criterion_detector_ce, device,
                    sub_epoch, args.refiner_gan_epochs_per_meta,
                    args.lambda_gan_refiner, args.realism_l1_weight_refiner # New args
                )
        elif args.train_refiner: # Train refiner without GAN (original L1 realism only)
            logger.info(f" Cycle {meta_epoch}: Training Refiner (r_xi) with L1 realism...")
            # You would need a non-GAN version of train_epoch_adversarial_refiner here
            # For simplicity, this path is less emphasized by the "optimal strategy"
            # Placeholder:
            # for sub_epoch in range(1, args.refiner_gan_epochs_per_meta + 1):
            #    train_epoch_adversarial_refiner_simple(...) 
            logger.warning("Simple refiner training (non-GAN) not fully implemented in this co-training example.")


        # B. Train Mask Generator (g_psi) if learned
        if args.use_learned_mask_generator and args.train_mask_generator and mask_gen_model:
            continue #TO CHANGE
            logger.info(f"  Cycle {meta_epoch}: Training Mask Generator (g_psi)...")
            for sub_epoch in range(1, args.mask_gen_epochs_per_meta + 1):
                train_epoch_critical_mask_generator(
                    mask_gen_model, detector_model, train_loader, optimizer_mask_gen, device,
                      args.lambda_budget_mask, args.target_mask_density_mask, sub_epoch, args.mask_gen_epochs_per_meta+1# New args
                )
            # Ensure refiner_model uses the updated mask_gen_model (it does if actual_mask_generator_fn is a reference)

        # C. Train/Harden Detector (f_theta)
        if args.adversarial_hardening_detector: # New arg for enabling detector training in co-loop
            logger.info(f"  Cycle {meta_epoch}: Hardening Detector (f_theta)...")
            # Prepare args_hardening namespace for train_epoch_detector_hardening
            class HardeningArgs: pass
            args_hardening_ns = HardeningArgs()
            args_hardening_ns.hardening_keep_original_fake_frac = args.hardening_keep_original_fake_frac
            args_hardening_ns.log_interval = args.log_interval

            for sub_epoch in range(1, args.detector_hardening_epochs_per_meta + 1):
                adversarially_harden_detector_on_mask(args,
                    detector_model, refiner_model, train_loader, val_loader, val_metadata_df,
                    processor, data_root_dir, device
                )
        
        # D. Evaluation and Checkpointing (Periodically, e.g., end of each meta_epoch)
        if val_loader and (meta_epoch % args.eval_every_meta_epochs == 0 or meta_epoch == args.co_training_meta_epochs): # New arg
            logger.info(f"--- Evaluating at end of Meta Epoch {meta_epoch} ---")
            detector_model.eval() # Ensure eval mode for all evaluations
            if mask_gen_model: mask_gen_model.eval()
            refiner_model.eval()
            if patch_discriminator_model: patch_discriminator_model.eval()

            # Evaluate Detector
            acc_clean, auc_clean = evaluate_detector_performance(detector_model, val_loader, device)
            logger.info(f"Detector Val (Clean) MetaEpoch {meta_epoch}: Acc: {acc_clean:.4f}, AUC: {auc_clean:.4f}")

            # Evaluate on refined fakes (requires _evaluate_hardened_detector_on_refined_fakes or similar)
            # acc_refined = _evaluate_hardened_detector_on_refined_fakes(...) # Adapt this function
            # logger.info(f"Detector Val (Refined) MetaEpoch {meta_epoch}: Acc: {acc_refined:.4f}")


            if auc_clean > best_detector_auc_val:
                best_detector_auc_val = auc_clean
                logger.info(f"🔥 New best detector AUC: {best_detector_auc_val:.4f} at Meta Epoch {meta_epoch}. Saving checkpoint...")
                torch.save({
                    'meta_epoch': meta_epoch,
                    'detector_state_dict': detector_model.state_dict(),
                    'mask_gen_state_dict': mask_gen_model.state_dict() if mask_gen_model else None,
                    'refiner_reconstructor_state_dict': refiner_model.region_reconstructor.state_dict(),
                    'patch_discriminator_state_dict': patch_discriminator_model.state_dict() if patch_discriminator_model else None,
                    'optimizer_detector_state_dict': optimizer_detector.state_dict(),
                    'optimizer_mask_gen_state_dict': optimizer_mask_gen.state_dict() if optimizer_mask_gen else None,
                    'optimizer_reconstructor_state_dict': optimizer_reconstructor.state_dict(),
                    'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict() if optimizer_discriminator else None,
                    'best_auc_val': best_detector_auc_val,
                    'args': args
                }, output_dir / f"co_trained_checkpoint_meta_epoch_{meta_epoch}_best_auc.pth")

    logger.info("--- Co-Training Pipeline Finished ---")
    logger.info(f"Best Detector Validation AUC achieved: {best_detector_auc_val:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Deepfake Co-Training Pipeline")

    # Environment & Config
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--max_duration', type=float, default=4.0)
    parser.add_argument('--output_dir', type=str, default="experiments/co_train_output")
    parser.add_argument('--log_interval', type=int, default=50, help="Log interval for sub-epoch training loops")

    # Data
    parser.add_argument('--metadata_path', type=str, required=True) 
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40) 

    # Detector Model (f_theta)
    parser.add_argument('--detector_pretrained_path', type=str, default="best_model_deepfake.pth")
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
    parser.add_argument('--evaluate_initial_detector', action='store_true')
    
    # Mask Generator (g_psi)
    parser.add_argument('--use_learned_mask_generator', action='store_true', default=True) # Default to true for co-train
    parser.add_argument('--train_mask_generator', action='store_true', default=False) # Default to true
    parser.add_argument('--mask_gen_unet_base_c', type=int, default=32)
    parser.add_argument('--lr_mask_gen', type=float, default=1e-4)

    parser.add_argument("--epochs_mask_gen", type=int, default=1, 
                        help="Number of epochs to train mask generator.")
    parser.add_argument('--lambda_budget_mask', type=float, default=0.1)
    parser.add_argument('--target_mask_density_mask', type=float, default=0.1)

    # Refiner (r_xi) & Patch Discriminator (D_phi)
    parser.add_argument('--train_refiner', action='store_true', default=False) # Default to true
    parser.add_argument('--refiner_unet_base_c', type=int, default=16)
    parser.add_argument('--lr_refiner', type=float, default=1e-4)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--refiner_epsilon_clip', type=float, default=0.03)
    parser.add_argument('--use_gan_for_refiner', action='store_true', default=False) # Default to true
    parser.add_argument('--patch_disc_base_c', type=int, default=32)
    parser.add_argument('--patch_disc_n_layers', type=int, default=3)
    parser.add_argument('--lr_patch_disc', type=float, default=1e-4)
    parser.add_argument('--lambda_gan_refiner', type=float, default=0.5, help="Weight for GAN loss for refiner fooling D_phi")
    parser.add_argument('--realism_l1_weight_refiner', type=float, default=0.1, help="Weight for L1 fidelity loss for refiner")

    # Detector Hardening (f_theta training part of co-loop)
    parser.add_argument('--adversarial_hardening_detector', action='store_true', default=True) # Default to true
    parser.add_argument('--lr_detector_hardening', type=float, default=5e-5) # Usually smaller LR for hardening
    parser.add_argument('--hardening_keep_original_fake_frac', type=float, default=0.1)

    parser.add_argument("--epochs_hardening", type=int, default=1, 
                        help="Number of epochs for detector hardening.") #I think this should be removed
    # Co-Training Loop Control
    parser.add_argument('--co_training_meta_epochs', type=int, default=10)
    parser.add_argument('--refiner_gan_epochs_per_meta', type=int, default=1, help="Sub-epochs for r_xi & D_phi per meta epoch")
    parser.add_argument('--mask_gen_epochs_per_meta', type=int, default=1, help="Sub-epochs for g_psi per meta epoch")
    parser.add_argument('--detector_hardening_epochs_per_meta', type=int, default=1, help="Sub-epochs for f_theta per meta epoch")
    parser.add_argument('--eval_every_meta_epochs', type=int, default=1, help="Evaluate on validation set every N meta epochs")

    args = parser.parse_args()
    run_co_training_pipeline(args)