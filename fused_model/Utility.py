from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchaudio

from ASVSpoofDataset import ASVSpoofProcessor
from tqdm import tqdm
import librosa
from models.RandomForest import extract_features
from models.SupportVector import SupportVectorSpoof
from dataAudio import AudioProcessor

DATA_ROOT_DIR = 'B:/4835108/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac'

IN_THE_WILD = "../processed_audio"
def get_predictions_test(df: pd.DataFrame,
                    cnn_model,
                    rf_model,
                    scaler_svm,
                    scaler_rf,
                    svm_model: SupportVectorSpoof,
                    audio_processor: ASVSpoofProcessor,
                    device):
    """
    Iterates through a dataframe to get predictions.
    - Loads/resamples audio for ResNet via AudioProcessor.
    - Loads raw audio for RF features via librosa.
    """
    cnn_model.eval()
    y_true, probs_resnet, probs_rf, probs_svm = [], [], [], []
    label_map = {'bona-fide': 0, 'spoof': 1}

    print(f"Generating predictions for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # build path
        file_path = Path(row['file'])
        if not file_path.exists():
            print(f"Warning: File not found {file_path}. Skipping.")
            continue

        # label
        lbl = row['label']
        if lbl not in label_map:
            print(f"Warning: Unknown label '{lbl}'. Skipping.")
            continue
        y_true.append(label_map[lbl])

        # ----- ResNet -----
        # 1) load & pad/truncate to config.sample_rate
        waveform, sr = torchaudio.load(file_path)
        if waveform.numel() == 0:
            print(f"Warning: empty waveform for {file_path}. Skipping.")
            y_true.pop()  # revert
            continue

        # 2) compute mel-spectrogram & batch-dim
        processed_audio = audio_processor.process_audio(waveform, 16000, 'eval')
        melspec = audio_processor.compute_mel(processed_audio)   # (n_mels, T)
        melspec = melspec.unsqueeze(0).to(device)         # (1, n_mels, T)

        with torch.no_grad():
            logits = cnn_model(melspec)
            probs = torch.softmax(logits, dim=1)
            probs_resnet.append(probs[0, 1].item())

        # ----- Random Forest -----
        try:
            audio_native, sr_native = librosa.load(str(file_path), sr=None)
            if audio_native.size == 0:
                raise ValueError("zero‐length audio")
        except Exception as e:
            print(f"Warning: RF load failed for {file_path} ({e}). Skipping.")
            y_true.pop()
            probs_resnet.pop()
            continue

        feats = extract_features(audio_native, sr_native)
        X_rf = np.array(list(feats.values())).reshape(1, -1)
        X_rf = scaler_rf.transform(X_rf)
        X_svm = svm_model.extract_features(audio_native, sr_native).reshape(1, -1)
        probs_rf.append(rf_model.predict_proba(X_rf)[0, 1])
        X_svm = scaler_svm.transform(X_svm)
        probs_svm.append(svm_model.model.predict_proba(X_svm)[0, 1])

    return np.array(y_true), np.array(probs_resnet), np.array(probs_rf), np.array(probs_svm)

def get_predictions_train(df: pd.DataFrame,
                    cnn_model,
                    rf_model,
                    scaler_svm,
                    scaler_rf,
                    svm_model: SupportVectorSpoof,
                    audio_processor: AudioProcessor,
                    device):
    """
    Iterates through a dataframe to get predictions.
    - Loads/resamples audio for ResNet via AudioProcessor.
    - Loads raw audio for RF features via librosa.
    """
    cnn_model.eval()
    y_true, probs_resnet, probs_rf, probs_svm = [], [], [], []
    label_map = {'bona-fide': 0, 'spoof': 1}

    print(f"Generating predictions for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # build path
        file_path = Path(IN_THE_WILD) / row['speaker'] / row['file']
        if not file_path.exists():
            print(f"Warning: File not found {file_path}. Skipping.")
            continue

        # label
        lbl = row['label']
        if lbl not in label_map:
            print(f"Warning: Unknown label '{lbl}'. Skipping.")
            continue
        y_true.append(label_map[lbl])

        # ----- ResNet -----
        # 1) load & pad/truncate to config.sample_rate
        waveform = audio_processor.load_audio(file_path)
        if waveform.numel() == 0:
            print(f"Warning: empty waveform for {file_path}. Skipping.")
            y_true.pop()  # revert
            continue

        # 2) compute mel-spectrogram & batch-dim
        melspec = audio_processor.compute_mel(waveform)   # (n_mels, T)
        melspec = melspec.unsqueeze(0).to(device)         # (1, n_mels, T)

        with torch.no_grad():
            logits = cnn_model(melspec)
            probs = torch.softmax(logits, dim=1)
            probs_resnet.append(probs[0, 1].item())

        # ----- Random Forest -----
        try:
            audio_native, sr_native = librosa.load(str(file_path), sr=None)
            if audio_native.size == 0:
                raise ValueError("zero‐length audio")
        except Exception as e:
            print(f"Warning: RF load failed for {file_path} ({e}). Skipping.")
            y_true.pop()
            probs_resnet.pop()
            continue

        feats = extract_features(audio_native, sr_native)
        X = np.array(list(feats.values())).reshape(1, -1)
        X = scaler_rf.transform(X)
        x_svm = svm_model.extract_features(audio_native, sr_native).reshape(1, -1)
        probs_rf.append(rf_model.predict_proba(X)[0, 1])
        x_svm = scaler_svm.transform(x_svm)
        probs_svm.append(svm_model.model.predict_proba(x_svm)[0, 1])

    return np.array(y_true), np.array(probs_resnet), np.array(probs_rf), np.array(probs_svm)
