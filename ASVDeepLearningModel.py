import argparse
import torch
import logging
from pathlib import Path
from torch import nn
# --- Assume all your other imports and helper functions are here ---
# from your_project.dataset import DeepfakeDataset, AudioConfig, AudioProcessor, collate_fn_skip_none
# from your_project.model import MultiViewCollaborativeNet
# from your_project.training import train_collaborative, evaluate_collaborative, CollaborativeLoss

# Stubs for self-contained example
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Controlla se MPS è disponibile E costruito per la tua versione di PyTorch
    device = torch.device("mps")

else:
    device = torch.device("cpu")


# --- (You would import your actual classes here) ---
from DeepLearningModel import DeepfakeClassifier, train
from IntermediateCrossFusion import CollaborativeLoss, MultiViewCollaborativeNet, collate_fn_skip_none, evaluate_collaborative, train_collaborative
from dataAudio import AudioConfig, AudioProcessor, DeepfakeASVDataset 
# Let's assume these other components exist


def main():
    parser = argparse.ArgumentParser(description="Multi-View Collaborative Training for Deepfake Detection")

    # --- Data Arguments (ADAPTED FOR NEW DATASET) ---
    parser.add_argument("--train_data_dir", type=Path, required=True, help="Root directory of the training audio data (.flac files).")
    parser.add_argument("--train_metadata", type=Path, required=True, help="Path to the training metadata file (e.g., ASVspoof train.txt).")
    parser.add_argument("--eval_data_dir", type=Path, help="Root directory of the evaluation audio data (.flac files).")
    parser.add_argument("--eval_metadata", type=Path, help="Path to the evaluation metadata file (e.g., ASVspoof eval.txt).")
    
    # --- General Training Arguments ---
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    
    # --- Optimizer and Scheduler Arguments ---
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.") # Adjusted default LR
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization.")

    # --- Model Architecture Arguments ---
    parser.add_argument("--wav2vec_model_name", type=str, default="facebook/wav2vec2-large-xlsr-53", help="Hugging Face model name for the frozen branch.")

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the classifier.")
    parser.add_argument("--initial_channels", type=int, default=32, help="Initial channels in the ResNet backbone.")
    parser.add_argument("--resnet_channels", nargs="+", type=int, default=[32, 64, 128, 256],
                        help="List of channel sizes for each ResNet block stage.")
    parser.add_argument("--resnet_blocks", nargs="+", type=int, default=[1, 1, 1, 1],  # Fixed type annotation
                        help="Number of residual blocks in each ResNet stage.")
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for the final classifier layer.")


    args = parser.parse_args()

    logger.info(f"Starting multi-view collaborative training with args: {args}")
    logger.info(f"Using device: {device}")
    
    if args.num_workers > 0 and device.type == 'cuda':
        if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            torch.multiprocessing.set_start_method('spawn', force=True)

    # --- Initialize Audio Processor ---
    audio_conf = AudioConfig()
    processor = AudioProcessor(audio_conf, wav2vec_model_name=args.wav2vec_model_name)

    # --- Create Datasets and DataLoaders (REWRITTEN) ---
    logger.info("--- Creating Datasets and DataLoaders ---")
    
    # Create Training Dataset
    train_dataset = DeepfakeASVDataset(
        root_dir=args.train_data_dir,
        metadata_path=args.train_metadata,
        mode='train',
        processor=processor,
        augment=True
    )
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Check your data and metadata paths. Exiting.")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
    )

    # Create Evaluation Dataset (Optional)
    eval_loader = None
    if args.eval_data_dir and args.eval_metadata:
        logger.info("Evaluation data and metadata provided. Creating evaluation dataloader.")
        eval_dataset = DeepfakeASVDataset(
            root_dir=args.eval_data_dir,
            metadata_path=args.eval_metadata,
            mode='eval',
            processor=processor,
            augment=False
        )
        if len(eval_dataset) > 0:
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
            )
        else:
            logger.warning("Evaluation dataset is empty, even though paths were provided.")
    else:
        logger.info("Evaluation data/metadata not provided. Skipping evaluation during training.")

    # --- Initialize the Collaborative Model ---
    logger.info("--- Initializing Multi-View Collaborative Network ---")
    model = DeepfakeClassifier( # Changed to DeepfakeClassifier
        num_classes=2, # Fixed to 2
        config=audio_conf,
        dropout=args.dropout,
        initial_channels=args.initial_channels,
        resnet_channels=args.resnet_channels,
        resnet_blocks=args.resnet_blocks,
        classifier_hidden_dim=args.classifier_hidden_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable model parameters: {trainable_params:,}")
    
    # --- Setup Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    # --- Start Training ---
    logger.info("--- Starting Training ---")
    save_path = "bestDeepLASVmodel.pth"
    criterion = nn.CrossEntropyLoss() # Suitable for binary classification (2 classes)

    metrics = train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        args.epochs,
        save_path  # Pass the save path
    )


    # --- Final Evaluation on Evaluation Set ---
    if eval_loader:
        logger.info("--- Loading best model for final evaluation on the evaluation set ---")
        if not Path(save_path).exists():
            logger.warning(f"Best model file {save_path} not found. Cannot perform final evaluation.")
            return
            
        final_model = MultiViewCollaborativeNet(wav2vec_model_name=args.wav2vec_model_name).to(device)
        final_model.load_state_dict(torch.load(save_path, map_location=device))
        logger.info(f"Successfully loaded best model from {save_path}")
        
        labels = [s['label'] for s in train_dataset.samples]
        num_bonafide = labels.count(0)
        num_spoof = labels.count(1)

        # 2. Calculate weights
        if num_bonafide > 0 and num_spoof > 0:
            total = len(train_dataset)
            weight_bonafide = total / (2 * num_bonafide)
            weight_spoof = total / (2 * num_spoof)
            class_weights = torch.tensor([weight_bonafide, weight_spoof], dtype=torch.float).to(device)
            
            logger.info(f"Using class weights for loss: Bonafide={weight_bonafide:.2f}, Spoof={weight_spoof:.2f}")
        else:
            logger.warning("Could not calculate class weights. Using unweighted loss.")
            class_weights = None

        # 3. Instantiate your loss function with the weights
        eval_criterion = CollaborativeLoss(class_weights=class_weights).to(device)
        test_results = evaluate_collaborative(final_model, eval_loader, eval_criterion)
        
        logger.info(
            f"Final Evaluation Results:\n"
            f"  Accuracy: {test_results['acc']:.4f}\n"
            f"  Loss: {test_results['loss']:.4f}\n"
            f"  Precision: {test_results['precision']:.4f}\n"
            f"  Recall: {test_results['recall']:.4f}\n"
            f"  EER: {test_results['eer']:.4f}"
        )
    else:
        logger.info("Skipping final evaluation as no evaluation set was provided.")

if __name__ == "__main__":
    main()