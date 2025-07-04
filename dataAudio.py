import logging
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd # Added for CSV reading
from transformers import Wav2Vec2FeatureExtractor # <-- ADD THIS IMPORT


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, config: AudioConfig, wav2vec_model_name: str = "facebook/wav2vec2-large-xlsr-53"):
        self.config = config
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=2.0,                # power spectrogram (magnitude squared)
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power',             # convert power to dB
            top_db=80.0
        )       
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_name)
        # Ensure the target sampling rate matches our config
        if self.wav2vec_feature_extractor.sampling_rate != config.sample_rate:
            raise ValueError(f"Wav2Vec processor expects sample rate {self.wav2vec_feature_extractor.sampling_rate}, "
                             f"but config is set to {config.sample_rate}")


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
    def compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (batch=1, time) or (batch, time)
        mel_power = self.melspec(waveform)      # -> (batch, n_mels, time_frames)
        mel_db = self.amplitude_to_db(mel_power)
        return mel_db
    
    def process_for_wav2vec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Takes a raw waveform tensor and prepares it for the Wav2Vec2 model.
        The Hugging Face processor handles normalization internally.
        """
        # The processor expects a raw waveform, not a batch.
        # It also expects the native sampling rate of the audio.
        # Since we already resampled to 16kHz in load_audio, this is simple.
        return self.wav2vec_feature_extractor(
            waveform, 
            sampling_rate=self.config.sample_rate,
            return_tensors="pt"
        ).input_values.squeeze(0) # Squeeze to remove the batch dim it adds


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
            logger.warning("!!!!Cannot extract features from empty waveform.!!!!!! LOADING DUMMY FEATURES")
            dummy_mel_frames = int(self.config.max_duration * self.config.sample_rate / self.config.hop_length) + 1
            dummy_wav2vec_len = int(self.config.max_duration * self.config.sample_rate)

            return {
                "zcr": torch.zeros(1, dummy_mel_frames),
                "rmse": torch.zeros(1, dummy_mel_frames),
                "mel": torch.zeros(self.config.n_mels, dummy_mel_frames),
                "wav2vec_input": torch.zeros(dummy_wav2vec_len), # Add this

                "mfcc": torch.zeros(self.config.mfcc_bins, dummy_mel_frames)
            }

        features = {}
        wf_for_frames = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform


        features["mel"] = self.compute_mel(waveform)
        features["wav2vec_input"] = self.process_for_wav2vec(waveform)


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
                "label": self.label_map[row['label']]  # Convert 'bona-fide'/'spoof' to 0/1
            })
        return samples

    def _init_augmentations(self):
        self.spec_augment_chain = None
        if self.augment:
            self.spec_augment_chain = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=self.processor.config.n_mels // 8),
                torchaudio.transforms.TimeMasking(time_mask_param=35)
            )

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]: # <-- Return type changed for clarity
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample_info = self.samples[idx]
        waveform = self.processor.load_audio(sample_info["path"])

        if waveform.numel() == 0:
            logger.warning(f"Skipping sample {sample_info['path']}. Returning None.")
            return None

        # This now returns a dict with "mel" and "wav2vec_input"
        features = self.processor.extract_features(waveform)

        # Augmentation only on mel spectrogram
        if self.augment and self.spec_augment_chain is not None:
            features["mel"] = self.spec_augment_chain(features["mel"].unsqueeze(0)).squeeze(0)

        deepfake_label = sample_info["label"]
        
        # --- KEY CHANGE: Return a dictionary that our new collate_fn can handle ---
        return {"features": features, "labels": torch.tensor(deepfake_label, dtype=torch.long)}

    def __len__(self) -> int:
        return len(self.samples)