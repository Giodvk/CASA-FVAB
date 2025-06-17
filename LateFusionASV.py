from pathlib import Path

import joblib
import numpy as np
import librosa
from CreateCSVASVSpoof import df, ASVSpoofProcessor
from dataAudio import AudioConfig, AudioProcessor
from lateFusion import evaluate_model, find_optimal_params
from models.ShallowLearningModel import extract_features
from DeepLearningModel import DeepfakeClassifier
import pandas as pd
import torch
from tqdm import tqdm

DATA_ROOT_DIR = 'B:/4835108/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac'
RESNET_MODEL_PATH = "./saved_models/best_model_deepfake.pth"
RF_MODEL_PATH = "./saved_models/Random_Forest.pkl"
SCALER_PATH = "scaler.pkl"
IN_THE_WILD = "./processed_audio"

train_metadata = pd.read_csv("./processed_audio/chunkedDf.csv")
val_metadata = df[:8787]


def get_predictions_test(df: pd.DataFrame,
                    resnet_model,
                    rf_model,
                    scaler,
                    audio_processor: ASVSpoofProcessor,
                    device):
    """
    Iterates through a dataframe to get predictions.
    - Loads/resamples audio for ResNet via AudioProcessor.
    - Loads raw audio for RF features via librosa.
    """
    resnet_model.eval()
    y_true, probs_resnet, probs_rf = [], [], []
    label_map = {'bona-fide': 0, 'spoof': 1}

    print(f"Generating predictions for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # build path
        file_path = Path(DATA_ROOT_DIR) / row['file']
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
            logits = resnet_model(melspec)
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
        Xs = scaler.transform(X)
        probs_rf.append(rf_model.predict_proba(Xs)[0, 1])

    return np.array(y_true), np.array(probs_resnet), np.array(probs_rf)

def get_predictions_train(df: pd.DataFrame,
                    resnet_model,
                    rf_model,
                    scaler,
                    audio_processor: AudioProcessor,
                    device):
    """
    Iterates through a dataframe to get predictions.
    - Loads/resamples audio for ResNet via AudioProcessor.
    - Loads raw audio for RF features via librosa.
    """
    resnet_model.eval()
    y_true, probs_resnet, probs_rf = [], [], []
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
            logits = resnet_model(melspec)
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
        Xs = scaler.transform(X)
        probs_rf.append(rf_model.predict_proba(Xs)[0, 1])

    return np.array(y_true), np.array(probs_resnet), np.array(probs_rf)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_config = AudioConfig()
    asv_processor = ASVSpoofProcessor(config=audio_config)
    audio_processor = AudioProcessor(config=audio_config)

    detector = DeepfakeClassifier(
        num_classes=2, config=audio_config, dropout=0.3, initial_channels=32,
        resnet_channels=[32, 64, 128, 256], resnet_blocks=[1, 1, 1, 1],
        classifier_hidden_dim=256
    ).to(device)
    detector.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))

    clf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    train_true, train_prob_resnet, train_prob_rf = get_predictions_train(
        train_metadata, detector, clf, scaler, audio_processor, device
    )

    best_alpha, best_threshold = find_optimal_params(train_true, train_prob_resnet, train_prob_rf)

    test_y_true, test_prob_resnet, test_prob_rf = get_predictions_test(val_metadata, detector,
                                                                       clf, scaler, asv_processor, device=device)

    results = {}
    results["Random Forest"] = evaluate_model("Random Forest Baseline", test_y_true, test_prob_rf, best_threshold)
    results["ResNet"] = evaluate_model("ResNet Baseline", test_y_true, test_prob_resnet, best_threshold)

    fused_probs = best_alpha * test_prob_resnet + (1 - best_alpha) * test_prob_rf
    results["Fused Model"] = evaluate_model(f"Fused Model (alpha={best_alpha:.2f})", test_y_true, fused_probs,
                                            best_threshold)

    # 6. Display summary
    print("\n--- PERFORMANCE SUMMARY ---\n")
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
