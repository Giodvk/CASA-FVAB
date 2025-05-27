# core.py
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
from DataBalancingDeepSeek import  train_speaker, test_speaker

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    win_length: int = 1024
    f_min: float = 20.0
    f_max: float = 8000.0
    mfcc_bins: int = 40
    zcr_threshold: float = 0.01
    max_duration: float = 4.0  # seconds


class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self._mel_basis = None  # Lazy initializatioon

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
        waveform, sr = torchaudio.load(path, normalize=True)
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate
            )
            waveform = resampler(waveform)
        return waveform.squeeze(0)

    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        # Time-domain features
        features["zcr"] = self._zero_crossing_rate(waveform)
        features["rmse"] = self._root_mean_square_energy(waveform)

        # Frequency-domain features
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)

        # Mel spectrogram
        mel_spec = torch.tensor(self.mel_basis).to(magnitude.device) @ magnitude
        features["mel"] = torch.log(mel_spec + 1e-6)

        # MFCCs
        transformMFCC = torchaudio.transforms.MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.mfcc_bins,
            log_mels=True
        )
        features["mfcc"] = transformMFCC(waveform)

        return features

    def _zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        # Ensure channel dim
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Frame the waveform
        frames = waveform.unfold(-1, self.config.win_length, self.config.hop_length)
        # Count zero crossings per frame
        sign_changes = (frames[..., :-1] * frames[..., 1:] < 0).sum(dim=-1).float()
        # Normalize by window length
        zcr = sign_changes / (self.config.win_length - 1)
        return zcr

    def _root_mean_square_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        ritorna a tensor of shape (num_channels, num_frames).
        """
        # Ensure channel dim
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Frame the waveform
        frames = waveform.unfold(-1, self.config.win_length, self.config.hop_length)
        # Compute mean square per frame
        mse = frames.pow(2).mean(dim=-1)
        # Root mean square
        rmse = torch.sqrt(mse + 1e-10)
        return rmse


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            speaker_ids: List[str],
            processor: AudioProcessor,
            augment: bool = True,
            max_samples: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.speaker_ids = speaker_ids
        self.processor = processor
        self.augment = augment
        self.max_samples = max_samples

        self.samples = self._load_metadata()
        self._init_augmentations()

    def _load_metadata(self) -> List[Dict]:
        samples = []
        for spk_id in self.speaker_ids:
            spk_dir = self.root_dir / spk_id
            for audio_path in spk_dir.glob("*.wav"):
                samples.append({
                    "path": audio_path,
                    "speaker": spk_id
                })
        return samples[:self.max_samples]

    def _init_augmentations(self):
        self.augment_chain = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        ) if self.augment else None

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        sample = self.samples[idx]
        waveform = self.processor.load_audio(sample["path"])
        """
        if self.augment:
            waveform = self._apply_augmentations(waveform)
        """
        features = self.processor.extract_features(waveform)
        return features, self.speaker_ids.index(sample["speaker"])

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        # come lo facciamo??
        ...


class SpeakerClassifier(nn.Module):
    def __init__(
            self,
            num_speakers: int,
            config: AudioConfig,
            dropout: float = 0.2
    ):
        super().__init__()
        self.config = config
        self.num_speakers = num_speakers

        self.feature_encoder = self._build_feature_encoder()
        self.classifier = self._build_classifier(dropout)

    def _build_feature_encoder(self) -> nn.Module:
        return nn.Sequential(

            #First Convolution
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #Second Convolution
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((16, 16))
        )

    def _build_classifier(self, dropout: float) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, self.num_speakers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_encoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        valid_N : int,
        epochs: int = 50,
) -> Dict[str, List[float]]:
    best_acc = 0.0
    metrics = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:

            features, labels = batch
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features["mel"].unsqueeze(1))
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device, valid_N)
        scheduler.step(int(val_acc))
        print(f"Epoch {epoch} accuracy : {val_acc:.4f}")

        metrics["train_loss"].append(loss.item())
        metrics["val_acc"].append(val_acc)

        if val_acc > best_acc:
            torch.save(model.state_dict(), "best_model.pth")
            best_acc = val_acc

    return metrics


def evaluate(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        valid_N: int
) -> float:
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for x, y in data_loader:
            features = {k: v.to(device) for k, v in x.items()}
            output = model(features["mel"].unsqueeze(1))
            accuracy += get_batch_accuracy(output, y, valid_N)
    return accuracy

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_speakers", nargs="+", required=True)
    parser.add_argument("--test_speakers", nargs="+", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(['--train_speakers', *train_speaker ,'--test_speaker', *test_speaker ,
                              "--data_dir", "./processed_audio/"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AudioConfig()

    # Initialize components
    processor = AudioProcessor(config)
    train_dataset = SpeakerDataset(args.data_dir, args.train_speakers, processor)
    test_dataset = SpeakerDataset(args.data_dir, args.test_speakers, processor, augment=False)

    model = SpeakerClassifier(
        num_speakers=len(args.train_speakers),
        config=config
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    len_valid = len(test_loader.dataset)

    # Training loop
    metrics = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        len_valid,
        args.epochs
    )


if __name__ == "__main__":
    main()
