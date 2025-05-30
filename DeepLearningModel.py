import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import collections
import pandas as pd # Added for CSV reading

# Assuming DataBalancingDeepSeek.py exists, but its speaker lists are no longer directly used for splits
# for deepfake detection, as the CSV will define the data.
try:
    from DataBalancingDeepSeek import train_speaker, test_speaker
except ImportError:
    print("Warning: DataBalancingDeepSeek.py not found or train_speaker/test_speaker not defined.")
    print("This is okay for deepfake detection, as splits are now handled via CSV metadata.")
    train_speaker = [] # Placeholder to avoid NameError if not found
    test_speaker = [] # Placeholder to avoid NameError if not found


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# collate_fn_skip_none MUST be at the global scope for multiprocessing (num_workers > 0)
def collate_fn_skip_none(batch):
    """
    Collate function that filters out None items from a batch and then uses
    the default collate function.
    """
    batch = [item for item in batch if item is not None]
    if not batch: # If all items were None, or the batch was empty to begin with
        return None
    return torch.utils.data.dataloader.default_collate(batch)

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    win_length: int = 1024
    f_min: float = 20.0
    f_max: float = 8000.0
    mfcc_bins: int = 40 # Number of MFCCs to return
    max_duration: float = 5.0  # seconds, for potential padding/truncating if needed


class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._mel_basis = None
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.mfcc_bins,
            melkwargs={
                'n_fft': self.config.n_fft,
                'hop_length': self.config.hop_length,
                'n_mels': self.config.n_mels,
                'f_min': self.config.f_min,
                'f_max': self.config.f_max,
                'win_length': self.config.win_length,
            },
            log_mels=True
        )

    @property
    def mel_basis(self):
        if self._mel_basis is None:
            self._mel_basis = librosa.filters.mel(
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                n_mels=self.config.n_mels,
                fmin=self.config.f_min,
                fmax=self.config.f_max
            )
        return self._mel_basis

    def load_audio(self, path: Path) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(path, normalize=True)
        except Exception as e:
            logger.error(f"Error loading audio file {path}: {e}")
            return torch.empty(0)

        if waveform.numel() == 0:
            logger.warning(f"Audio file {path} is empty or could not be loaded.")
            return torch.empty(0)

        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate
            )
            waveform = resampler(waveform)

        max_samples = int(self.config.max_duration * self.config.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        elif waveform.shape[-1] < max_samples:
            padding_needed = max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

        return waveform.squeeze(0)

    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        if waveform.numel() == 0:
            logger.warning("Cannot extract features from empty waveform.")
            dummy_mel_frames = int(self.config.max_duration * self.config.sample_rate / self.config.hop_length) + 1
            return {
                "zcr": torch.zeros(1, dummy_mel_frames),
                "rmse": torch.zeros(1, dummy_mel_frames),
                "mel": torch.zeros(self.config.n_mels, dummy_mel_frames),
                "mfcc": torch.zeros(self.config.mfcc_bins, dummy_mel_frames)
            }

        features = {}
        wf_for_frames = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

        features["zcr"] = self._zero_crossing_rate(wf_for_frames)
        features["rmse"] = self._root_mean_square_energy(wf_for_frames)

        window = torch.hann_window(self.config.win_length, device=waveform.device)

        stft_result = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            return_complex=True
        )
        magnitude = torch.abs(stft_result)

        mel_basis_tensor = torch.tensor(self.mel_basis, dtype=magnitude.dtype, device=magnitude.device)
        mel_spec = mel_basis_tensor @ magnitude
        features["mel"] = torch.log(mel_spec + 1e-6)

        mfcc_input = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        features["mfcc"] = self.mfcc_transform(mfcc_input).squeeze(0)

        return features

    def _zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        frames = waveform.unfold(dimension=-1, size=self.config.win_length, step=self.config.hop_length)
        sign_changes = (frames[..., :-1] * frames[..., 1:] < 0).sum(dim=-1).float()
        zcr = sign_changes / (self.config.win_length - 1)
        return zcr

    def _root_mean_square_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        frames = waveform.unfold(dimension=-1, size=self.config.win_length, step=self.config.hop_length)
        mse = frames.pow(2).mean(dim=-1)
        rmse = torch.sqrt(mse + 1e-10)
        return rmse


class DeepfakeDataset(torch.utils.data.Dataset): # Renamed from SpeakerDataset
    def __init__(
            self,
            root_dir: Path,
            metadata_df: pd.DataFrame, # Pass a DataFrame directly
            processor: AudioProcessor,
            augment: bool = False,
    ):
        self.root_dir = root_dir
        self.metadata_df = metadata_df
        self.processor = processor
        self.augment = augment

        # Map 'bona-fide' to 0 and 'spoof' to 1
        self.label_map = {'bona-fide': 0, 'spoof': 1}
        self.samples = self._load_samples()
        if not self.samples:
            logger.warning(f"No audio samples found based on provided metadata in {root_dir}. Dataset is empty.")
        self._init_augmentations()

    def _load_samples(self) -> List[Dict]:
        samples = []
        for idx, row in self.metadata_df.iterrows():
            # CORRECTED: Construct path as root_dir / speaker / file
            audio_path = self.root_dir / row['speaker'] / row['file']
            if not audio_path.exists():
                logger.warning(f"Audio file {audio_path} not found. Skipping.")
                continue
            samples.append({
                "path": audio_path,
                "speaker": row['speaker'],
                "label": self.label_map[row['label']] # Convert 'bona-fide'/'spoof' to 0/1
            })
        return samples

    def _init_augmentations(self):
        self.spec_augment_chain = None
        if self.augment:
            self.spec_augment_chain = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=self.processor.config.n_mels // 8),
                torchaudio.transforms.TimeMasking(time_mask_param=35)
            )

    def __getitem__(self, idx: int) -> Optional[Tuple[Dict[str, torch.Tensor], int]]:
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample_info = self.samples[idx]
        waveform = self.processor.load_audio(sample_info["path"])

        if waveform.numel() == 0:
            logger.warning(f"Skipping sample {sample_info['path']} due to loading error or empty audio. Returning None.")
            return None

        features = self.processor.extract_features(waveform)

        if self.augment and self.spec_augment_chain is not None:
            features["mel"] = self.spec_augment_chain(features["mel"].unsqueeze(0)).squeeze(0)

        # The label is now 0 for bona-fide, 1 for spoof
        deepfake_label = sample_info["label"]
        return features, deepfake_label

    def __len__(self) -> int:
        return len(self.samples)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# --- ResNet Architecture (Classifier now outputs 2 classes) ---
class DeepfakeClassifier(nn.Module): # Renamed from SpeakerClassifier
    def __init__(
            self,
            num_classes: int = 2, # Fixed to 2 for bona-fide/spoof
            config: AudioConfig = AudioConfig(), # Default config for convenience
            dropout: float = 0.2,
            initial_channels: int = 32,
            resnet_channels: List[int] = [32, 64, 128, 256],
            resnet_blocks: List[int] = [1, 1, 1, 1],
            classifier_hidden_dim: int = 256
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("DeepfakeClassifier must have num_classes = 2 (bona-fide/spoof).")
        self.num_classes = num_classes
        self.config = config
        self.current_in_channels = initial_channels
        self.feature_encoder = self._build_feature_encoder(initial_channels, resnet_channels, resnet_blocks)
        encoder_out_features = resnet_channels[-1]
        self.classifier = self._build_classifier(encoder_out_features, classifier_hidden_dim, dropout)

    def _make_layer(self, block_class: nn.Module, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block_class(self.current_in_channels, out_channels, s))
            self.current_in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_feature_encoder(self, initial_channels: int, resnet_channels: List[int], resnet_blocks: List[int]) -> nn.Module:
        if not (len(resnet_channels) == len(resnet_blocks)):
            raise ValueError("resnet_channels and resnet_blocks must have the same length.")

        layers = [
            nn.Conv2d(1, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.current_in_channels = initial_channels

        layers.append(self._make_layer(ResidualBlock, resnet_channels[0], resnet_blocks[0], stride=1))
        for i in range(1, len(resnet_channels)):
            layers.append(self._make_layer(ResidualBlock, resnet_channels[i], resnet_blocks[i], stride=2))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _build_classifier(self, encoder_out_features: int, hidden_dim: int, dropout: float) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(encoder_out_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes) # Output 2 classes
        )

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        if x_mel.dim() == 3:
             x_mel = x_mel.unsqueeze(1)
        features = self.feature_encoder(x_mel)
        flattened_features = torch.flatten(features, 1)
        logits = self.classifier(flattened_features)
        return logits

def train(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: torch.nn.Module,
        device: torch.device,
        epochs: int = 50,
) -> Dict[str, List[float]]:
    best_val_acc = 0.0
    metrics = {"train_loss": [], "val_acc": [], "train_acc": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if batch_data is None:
                logger.warning(f"Skipping None batch at epoch {epoch}, index {batch_idx} in training (collate_fn returned None).")
                continue

            features_dict, labels = batch_data

            mel_spectrograms = features_dict["mel"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * mel_spectrograms.size(0)
            preds = outputs.argmax(dim=1)
            epoch_train_correct += preds.eq(labels).sum().item()
            epoch_train_samples += labels.size(0)

            if batch_idx > 0 and batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{epochs-1} | Batch {batch_idx}/{len(train_loader)-1} | Train Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        avg_epoch_train_loss = epoch_train_loss / epoch_train_samples if epoch_train_samples > 0 else 0
        avg_epoch_train_acc = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0

        metrics["train_loss"].append(avg_epoch_train_loss)
        metrics["train_acc"].append(avg_epoch_train_acc)

        # Validation
        if val_loader:
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            metrics["val_acc"].append(val_acc)
            metrics["val_loss"].append(val_loss)
            logger.info(f"Epoch {epoch}/{epochs-1} Summary: Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {avg_epoch_train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    torch.save(model.state_dict(), "best_model_deepfake.pth") # Changed model name
                    logger.info(f"New best model saved with Val Acc: {best_val_acc:.4f}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")
        else:
            logger.info(f"Epoch {epoch}/{epochs-1} Summary: Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {avg_epoch_train_acc:.4f} | No validation.")
            metrics["val_acc"].append(0.0)
            metrics["val_loss"].append(0.0)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step()

    logger.info(f"Training finished. Best Validation Accuracy (if applicable): {best_val_acc:.4f}")
    return metrics

def evaluate(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_data is None:
                logger.warning(f"Skipping None batch at index {batch_idx} during evaluation (collate_fn returned None).")
                continue

            features_dict, labels = batch_data
            mel_spectrograms = features_dict["mel"].to(device)
            labels = labels.to(device)

            outputs = model(mel_spectrograms)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * mel_spectrograms.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    average_loss = total_loss / total_samples if total_samples > 0 else 0
    return accuracy, average_loss

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training with ResNet")

    parser.add_argument("--metadata_path", type=Path, required=True,
                        help="Path to the CSV metadata file (e.g., PROCESSED_AUDIO/chunkedDF.csv).")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root directory of audio data (e.g., PROCESSED_AUDIO).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization (weight decay).")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for the classifier.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers. Set to 0 for debugging pickling issues.")
    parser.add_argument("--test_split_ratio", type=float, default=0.2,
                        help="Ratio of unique speakers to reserve for the test set (e.g., 0.2 for 20%).")


    # Model architecture arguments
    parser.add_argument("--initial_channels", type=int, default=32, help="Initial channels in the ResNet backbone.")
    parser.add_argument("--resnet_channels", nargs="+", type=int, default=[32, 64, 128, 256],
                        help="List of channel sizes for each ResNet block stage.")
    parser.add_argument("--resnet_blocks", nargs="+", type=int, default=[1, 1, 1, 1],
                        help="Number of residual blocks in each ResNet stage.")
    parser.add_argument("--classifier_hidden_dim", type=int, default=256,
                        help="Hidden dimension for the final classifier layer.")


    args = parser.parse_args()

    logger.info(f"Starting deepfake detection training with args: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if args.num_workers > 0 and device.type == 'cuda':
        if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            logger.info(f"Setting multiprocessing start method to 'spawn' for CUDA compatibility with num_workers > 0.")
            try:
                torch.multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError as e:
                logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA in child processes.")


    audio_conf = AudioConfig()
    processor = AudioProcessor(audio_conf)

    # --- Load and Split Metadata for Deepfake Detection ---
    logger.info("--- Loading and Splitting Deepfake Metadata ---")
    if not args.metadata_path.exists():
        logger.error(f"Metadata CSV file not found at {args.metadata_path}. Exiting.")
        return

    full_metadata_df = pd.read_csv(args.metadata_path)
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

    # Get all unique speakers
    all_speakers = full_metadata_df['speaker'].unique().tolist()
    np.random.shuffle(all_speakers) # Shuffle speakers for random split

    # Perform speaker-disjoint split
    num_test_speakers = int(len(all_speakers) * args.test_split_ratio)
    if num_test_speakers == 0 and len(all_speakers) > 0: # Ensure at least one test speaker if possible
        num_test_speakers = 1
    if num_test_speakers >= len(all_speakers): # Handle case where split ratio is too high
        logger.warning(f"Test split ratio {args.test_split_ratio} is too high for {len(all_speakers)} speakers. Using all speakers for training and no separate test set.")
        train_speakers = all_speakers
        test_speakers = []
    else:
        test_speakers = all_speakers[:num_test_speakers]
        train_speakers = all_speakers[num_test_speakers:]

    logger.info(f"Total unique speakers in metadata: {len(all_speakers)}")
    logger.info(f"Number of training speakers: {len(train_speakers)}")
    logger.info(f"Training speakers IDs (first 5): {train_speakers[:5]} ... (last 5): {train_speakers[-5:]}")
    logger.info(f"Number of testing speakers: {len(test_speakers)}")
    logger.info(f"Testing speakers IDs (first 5): {test_speakers[:5]} ... (last 5): {test_speakers[-5:] if test_speakers else 'N/A'}")

    # Create train and test DataFrames based on speaker IDs
    train_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(train_speakers)].reset_index(drop=True)
    test_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(test_speakers)].reset_index(drop=True)

    # --- Data Statistics: Samples per Speaker and Bona-fide/Spoof Balance ---
    logger.info("\n--- Training Data Statistics ---")
    logger.info(f"Total training samples: {len(train_metadata_df)}")
    if not train_metadata_df.empty:
        train_speaker_counts = train_metadata_df['speaker'].value_counts()
        logger.info(f"Average samples per training speaker: {train_speaker_counts.mean():.2f}")
        logger.info(f"Training samples per speaker (top 5): {train_speaker_counts.head(5).to_dict()}")
        logger.info(f"Training samples per speaker (bottom 5): {train_speaker_counts.tail(5).to_dict()}")

        train_label_counts = train_metadata_df['label'].value_counts()
        logger.info(f"Training label balance: {train_label_counts.to_dict()}")
        if 'bona-fide' in train_label_counts and 'spoof' in train_label_counts:
            bona_fide_ratio = train_label_counts['bona-fide'] / len(train_metadata_df) # Corrected variable name
            spoof_ratio = train_label_counts['spoof'] / len(train_metadata_df)
            logger.info(f"Training Bona-fide ratio: {bona_fide_ratio:.2f}, Spoof ratio: {spoof_ratio:.2f}")
        else:
            logger.warning("Training set missing 'bona-fide' or 'spoof' labels for balance calculation.")
    else:
        logger.error("Training metadata DataFrame is empty. Cannot proceed.")
        return

    logger.info("\n--- Testing Data Statistics ---")
    logger.info(f"Total testing samples: {len(test_metadata_df)}")
    if not test_metadata_df.empty:
        test_speaker_counts = test_metadata_df['speaker'].value_counts()
        logger.info(f"Average samples per testing speaker: {test_speaker_counts.mean():.2f}")
        logger.info(f"Testing samples per speaker (top 5): {test_speaker_counts.head(5).to_dict()}")
        logger.info(f"Testing samples per speaker (bottom 5): {test_speaker_counts.tail(5).to_dict()}")

        test_label_counts = test_metadata_df['label'].value_counts()
        logger.info(f"Testing label balance: {test_label_counts.to_dict()}")
        if 'bona-fide' in test_label_counts and 'spoof' in test_label_counts:
            bona_fide_ratio = test_label_counts['bona-fide'] / len(test_metadata_df) # Corrected variable name
            spoof_ratio = test_label_counts['spoof'] / len(test_metadata_df)
            logger.info(f"Testing Bona-fide ratio: {bona_fide_ratio:.2f}, Spoof ratio: {spoof_ratio:.2f}")
        else:
            logger.warning("Testing set missing 'bona-fide' or 'spoof' labels for balance calculation.")
    else:
        logger.warning("Testing metadata DataFrame is empty. Test loader will not be created.")
    logger.info("------------------------------------------")


    train_dataset = DeepfakeDataset(args.data_dir, train_metadata_df, processor, augment=True)
    test_dataset = None
    if not test_metadata_df.empty:
        test_dataset = DeepfakeDataset(args.data_dir, test_metadata_df, processor, augment=False)


    if len(train_dataset) == 0:
        logger.error("Training dataset is empty after filtering. Please check data_dir and metadata_path.")
        return

    model = DeepfakeClassifier( # Changed to DeepfakeClassifier
        num_classes=2, # Fixed to 2
        config=audio_conf,
        dropout=args.dropout,
        initial_channels=args.initial_channels,
        resnet_channels=args.resnet_channels,
        resnet_blocks=args.resnet_blocks,
        classifier_hidden_dim=args.classifier_hidden_dim
    ).to(device)
    logger.info(f"Model initialized: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss() # Suitable for binary classification (2 classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn_skip_none
    )

    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=collate_fn_skip_none
        )
    else:
        logger.info("Test loader will not be created as the test dataset is empty or not initialized.")

    if len(train_loader) == 0 and len(train_dataset) > 0 :
        logger.error("Train loader is empty but dataset was not. All training samples might have been filtered by collate_fn.")
        return
    elif len(train_loader) == 0:
        logger.error("Train loader is empty. Dataset might be empty or all samples filtered.")
        return

    # --- Feature Visualization (Logging Statistics) ---
    logger.info("--- Feature Statistics from First Batch ---")
    try:
        train_batch = next(iter(train_loader))
        if train_batch is not None:
            train_mels = train_batch[0]["mel"]
            logger.info(f"Train Mel Spectrogram Batch Shape: {train_mels.shape}")
            logger.info(f"Train Mel Spectrogram Stats: Min={train_mels.min():.4f}, Max={train_mels.max():.4f}, Mean={train_mels.mean():.4f}, Std={train_mels.std():.4f}")
        else:
            logger.warning("Could not get a valid first batch from train_loader for feature stats.")

        if test_loader and len(test_loader) > 0:
            test_batch = next(iter(test_loader))
            if test_batch is not None:
                test_mels = test_batch[0]["mel"]
                logger.info(f"Test Mel Spectrogram Batch Shape: {test_mels.shape}")
                logger.info(f"Test Mel Spectrogram Stats: Min={test_mels.min():.4f}, Max={test_mels.max():.4f}, Mean={test_mels.mean():.4f}, Std={test_mels.std():.4f}")
            else:
                logger.warning("Could not get a valid first batch from test_loader for feature stats.")
        else:
            logger.info("Skipping test feature stats as test loader is not available.")
    except Exception as e:
        logger.error(f"Error getting feature statistics: {e}")
    logger.info("------------------------------------------")


    logger.info("Starting training...")
    metrics = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        args.epochs
    )

    logger.info(f"Training completed. Metrics: {metrics}")
    logger.info(f"Best model saved to best_model_deepfake.pth (if validation accuracy improved).")

    if test_loader:
        logger.info("Loading best model for final evaluation on the test set...")
        try:
            final_model = DeepfakeClassifier(
                num_classes=2, # Fixed to 2
                config=audio_conf,
                dropout=args.dropout,
                initial_channels=args.initial_channels,
                resnet_channels=args.resnet_channels,
                resnet_blocks=args.resnet_blocks,
                classifier_hidden_dim=args.classifier_hidden_dim
            ).to(device)
            final_model.load_state_dict(torch.load("best_model_deepfake.pth", map_location=device))
            final_test_acc, final_test_loss = evaluate(final_model, test_loader, criterion, device)
            logger.info(f"Final Test Accuracy (best model): {final_test_acc:.4f}, Final Test Loss: {final_test_loss:.4f}")
        except FileNotFoundError:
            logger.warning("best_model_deepfake.pth not found. Skipping final evaluation with best model.")
        except Exception as e:
            logger.error(f"Error during final evaluation with best model: {e}")
    else:
        logger.info("Skipping final evaluation on test set as test loader was not available.")

if __name__ == "__main__":
    main()
