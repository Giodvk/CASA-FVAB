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
            self.raw_boost_augment = RawBoost(sample_rate=self.processor.config.sample_rate)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]: # <-- Return type changed for clarity
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample_info = self.samples[idx]
        waveform = self.processor.load_audio(sample_info["path"])

        if waveform.numel() == 0:
            logger.warning(f"Skipping sample {sample_info['path']}. Returning None.")
            return None
        
        if self.augment and self.raw_boost_augment is not None:
            waveform = self.raw_boost_augment(waveform)

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
    

# raw_boost.py
import numpy as np
import scipy.signal
import torch
from typing import List

# A small utility to convert dB to linear scale
def db_to_linear(db_value):
    return 10**(db_value / 20.0)

class RawBoost:
    """
    Implementation of the RawBoost data augmentation algorithm.
    Based on the paper: "RawBoost: A Raw Data Boosting and Augmentation Method..."
    https://www.isca-speech.org/archive/interspeech_2020/tak20_interspeech.html
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Define ranges for random parameters as per the paper (Table 1)
        self.n_fir_coeffs = [10, 100]  # Range for Nfir
        self.freq_lims = [20, 8000]     # Range for fc
        self.bw_lims = [100, 1000]      # Range for Δf
        self.snr_lims = [15, 35]       # Range for SNR (dB) for additive noise
        
        self.gcn_lims_db = [-5, -20]   # Range for gcn_2 to gcn_Nf (dB)
        self.n_nonlinear_order = 5     # Nf

        self.prel_lims = [0.01, 0.10]  # Range for Prel (%) for impulsive noise
        self.gsd_db = 2                # Gain for impulsive noise (dB)

    def _design_random_fir_filter(self):
        """Designs a random FIR filter with multiple notches."""
        n_coeffs = np.random.randint(self.n_fir_coeffs[0], self.n_fir_coeffs[1])
        n_notches = 5
        
        notch_freqs = np.random.uniform(self.freq_lims[0], self.freq_lims[1], n_notches)
        notch_bws = np.random.uniform(self.bw_lims[0], self.bw_lims[1], n_notches)
        
        # Create frequency bands to keep (everything except the notches)
        bands = [0]
        for f, bw in zip(notch_freqs, notch_bws):
            bands.extend([f - bw/2, f + bw/2])
        bands.append(self.sample_rate / 2)
        
        # Define desired gains for each band (1 for passbands, 0 for stopbands)
        desired = []
        for i in range(len(bands) // 2):
            desired.extend([1, 0])
        if len(bands) % 2 != 0:
            desired.append(1)
            
        # Normalize frequencies to Nyquist
        bands = np.array(bands) / (self.sample_rate / 2.0)
        
        # Ensure bands are within [0, 1] and sorted
        bands = np.clip(bands, 0, 1)
        bands = np.sort(bands)
        
        # Design the filter
        try:
            # firwin2 is great for arbitrary frequency response shapes
            fir_coeffs = scipy.signal.firwin2(n_coeffs, bands, desired)
            return fir_coeffs
        except ValueError:
            # Fallback to a simpler filter if firwin2 fails (e.g., due to bad random params)
            return scipy.signal.firwin(n_coeffs, cutoff=0.5, window='hamming')


    def _process1_convolutional(self, x: np.ndarray) -> np.ndarray:
        """Applies linear and non-linear convolutional noise."""
        y = np.zeros_like(x)
        
        # Linear part (j=1)
        filter_coeffs = self._design_random_fir_filter()
        y += scipy.signal.lfilter(filter_coeffs, 1, x)
        
        # Non-linear part (j=2 to Nf)
        for j in range(2, self.n_nonlinear_order + 1):
            filter_coeffs = self._design_random_fir_filter()
            gain = db_to_linear(np.random.uniform(self.gcn_lims_db[0], self.gcn_lims_db[1]))
            # Apply filter to the powered signal
            y += gain * scipy.signal.lfilter(filter_coeffs, 1, x**j)
            
        return y

    def _get_impulsive_random_values(self, num_samples):
        """Generates random values from a distribution similar to f(r) = -log(|r|)"""
        # Start with a uniform distribution in (0, 1]
        u = np.random.uniform(low=1e-6, high=1.0, size=num_samples)
        # Transform to get the desired shape (more small values)
        # r = exp(-u*k) where k controls the skew. Let's use k=5 for a strong skew.
        r = np.exp(-u * 5) 
        # Randomly assign sign
        signs = np.random.choice([-1, 1], size=num_samples)
        return r * signs
    def _process2_impulsive(self, x: np.ndarray) -> np.ndarray:
        """Applies signal-dependent additive impulsive noise."""
        y = x.copy()
        
        prel = np.random.uniform(self.prel_lims[0], self.prel_lims[1])
        n_samples_to_corrupt = int(prel * len(x))
        
        if n_samples_to_corrupt == 0:
            return y
            
        indices = np.random.choice(len(x), n_samples_to_corrupt, replace=False)
        
        # Generate random values for the noise impulse
        # The paper's distribution is complex. A uniform distribution is a reasonable simplification.
        random_values = self._get_impulsive_random_values(n_samples_to_corrupt)

        gain = db_to_linear(self.gsd_db)
        
        noise = gain * random_values * x[indices]
        y[indices] += noise
        
        return y

    def _process3_additive(self, x: np.ndarray) -> np.ndarray:
        """Applies signal-independent stationary additive noise."""
        # Generate white noise
        white_noise = np.random.randn(len(x))
        
        # Color the noise with a random filter
        filter_coeffs = self._design_random_fir_filter()
        colored_noise = scipy.signal.lfilter(filter_coeffs, 1, white_noise)
        
        # Pick a random SNR
        snr_db = np.random.uniform(self.snr_lims[0], self.snr_lims[1])
        
        # Calculate signal and noise power
        signal_power = np.sum(x**2)
        noise_power = np.sum(colored_noise**2)
        
        if noise_power == 0: # Avoid division by zero
            return x

        # Calculate gain to achieve the target SNR
        # SNR_db = 10 * log10(P_signal / P_noise)
        # P_noise_target = P_signal / 10^(SNR_db/10)
        # gain^2 * P_noise_current = P_noise_target
        # gain = sqrt(P_noise_target / P_noise_current)
        gain = np.sqrt(signal_power / (noise_power * (10**(snr_db/10))))
        
        y = x + gain * colored_noise
        return y

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies the full RawBoost augmentation pipeline.
        The paper found that a series of (1) and (2) was effective.
        We can randomly choose a combination of the three.
        """
        # Convert to numpy, ensure it's a 1D array
        original_shape = waveform.shape
        # Ensure we are working with a 1D numpy array for processing
        x = waveform.numpy().squeeze()
        if x.ndim > 1: # Defensive check
            x = x.flatten()
        
        # Randomly choose which augmentations to apply
        # This adds more variety than applying all three every time.
        # The paper's best result was a series of 1 and 2. Let's prioritize that.
        
        # Apply Process 1 (Convolutional)
        if np.random.rand() < 0.3: # Apply with 30% probability
            x = self._process1_convolutional(x)

        # Apply Process 2 (Impulsive)
        if np.random.rand() < 0.3: # Apply with 30% probability
            x = self._process2_impulsive(x)

        # Apply Process 3 (Additive)
        if np.random.rand() < 0.0: # Apply with 0% probability
            x = self._process3_additive(x)
            
        # Final normalization to prevent clipping/overflow
        max_val = np.abs(x).max()
        if max_val > 0:
            x = x / max_val

        augmented_x = torch.from_numpy(x).float() 
        # Convert back to a torch tensor with a channel dimension
        return augmented_x.view(original_shape)


class DeepfakeASVDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            metadata_path: Path,
            mode: str, # NEW: 'train' or 'eval' to handle different metadata formats
            processor: AudioProcessor,
            augment: bool = False,
    ):
        self.root_dir = root_dir
        self.metadata_path = metadata_path
        self.processor = processor
        self.augment = augment

        if mode not in ['train', 'eval']:
            raise ValueError(f"Mode must be 'train' or 'eval', but got '{mode}'")
        self.mode = mode

        # Map labels to integers. Handles 'bonafide' and 'spoof'.
        self.label_map = {'bonafide': 0, 'spoof': 1}
        
        self.samples = self._load_samples()
        if not self.samples:
            logger.warning(f"No audio samples found based on metadata in {metadata_path}. Dataset is empty.")
        
        self._init_augmentations()

    def _load_samples(self) -> List[Dict]:
        """
        Parses the ASVspoof2019 metadata file and creates a list of samples.
        """
        samples = []
        logger.info(f"Loading samples for mode: '{self.mode}' from {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # Skip empty or malformed lines
                if not parts:
                    continue

                filename_stem = ""
                label_str = ""

                # Parse line based on the mode
                if self.mode == 'train':
                    # Format: LA_0079 LA_T_1138215 - - bonafide
                    if len(parts) < 5:
                        logger.warning(f"Skipping malformed train line: '{line.strip()}'")
                        continue
                    filename_stem = parts[1]
                    label_str = parts[4]
                
                elif self.mode == 'eval':
                    # Format: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
                    if len(parts) < 6:
                        logger.warning(f"Skipping malformed eval line: '{line.strip()}'")
                        continue
                    filename_stem = parts[1]
                    label_str = parts[5]

                # Construct the full path to the .flac file
                audio_path = self.root_dir / f"{filename_stem}.flac"

                if not audio_path.exists():
                    logger.warning(f"Audio file {audio_path} not found. Skipping.")
                    continue
                
                # Convert string label to integer
                if label_str not in self.label_map:
                    logger.warning(f"Unknown label '{label_str}' in line: {line.strip()}. Skipping.")
                    continue
                
                label = self.label_map[label_str]
                
                samples.append({
                    "path": audio_path,
                    "label": label
                    # We no longer need the speaker ID, but you could add it if needed:
                    # "speaker": parts[0] 
                })
        
        logger.info(f"Successfully loaded {len(samples)} samples.")
        return samples

    def _init_augmentations(self):
        self.spec_augment_chain = None
        self.raw_boost_augment = None # Initialize to None
        if self.augment:
            # Your existing spectrogram augmentation
            self.spec_augment_chain = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=self.processor.config.n_mels // 8),
                torchaudio.transforms.TimeMasking(time_mask_param=35)
            )
            # NEW: Initialize RawBoost for waveform augmentation
            self.raw_boost_augment = RawBoost(sample_rate=self.processor.config.sample_rate)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample_info = self.samples[idx]
        waveform = self.processor.load_audio(sample_info["path"])

        if waveform.numel() == 0:
            logger.warning(f"Skipping sample {sample_info['path']} due to loading error. Returning None.")
            # Note: Your DataLoader's collate_fn should handle None values, e.g., by filtering them out.
            return None

        if self.augment and self.raw_boost_augment is not None:
            waveform = self.raw_boost_augment(waveform)
            
        features = self.processor.extract_features(waveform)

        if self.augment and self.spec_augment_chain is not None:
            features["mel"] = self.spec_augment_chain(features["mel"].unsqueeze(0)).squeeze(0)

        deepfake_label = sample_info["label"]
        
        return {"features": features, "labels": torch.tensor(deepfake_label, dtype=torch.long)}

    def __len__(self) -> int:
        return len(self.samples)