import logging
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import joblib
from tqdm import tqdm
from DataBalancingDeepSeek import train_speaker, test_speaker
from DeepLearningModel import DeepfakeClassifier
from ShallowLearningModel import extract_features
from dataAudio import AudioConfig, AudioProcessor

# --- Configuration & Setup ---
# Set your paths and parameters here
DATA_ROOT_DIR = "/Users/salvatorebasilicata/Desktop/Magistrale/CASA-FVAB/processed_audio" # Root directory for audio chunks
CSV_PATH = "/Users/salvatorebasilicata/Desktop/Magistrale/CASA-FVAB/processed_audio/chunkedDf.csv"       # Path to the directory containing chunkedDf.csv
RESNET_MODEL_PATH = "best_model_deepfake.pth"
RF_MODEL_PATH = "Random_Forest.pkl"
SCALER_PATH = "scaler.pkl"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Assuming you have these pre-defined from your training scripts
# Replace with your actual train/val/test speaker lists or dataframes
# For this script, we'll assume test_metadata_df is what we want to evaluate on
# You would typically have a separate validation set to find alpha
full_metadata_df = pd.read_csv(CSV_PATH)

train_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(train_speaker)].reset_index(drop=True)
val_metadata_df = full_metadata_df[full_metadata_df['speaker'].isin(test_speaker)].reset_index(drop=True)

# --- Model & Feature Extraction Definitions (Copied from your snippets) ---
# NOTE: Make sure these class/function definitions match your training environment exactly.



# --- Main Logic ---


def get_predictions(df: pd.DataFrame,
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
        file_path = Path(DATA_ROOT_DIR) / row['speaker'] / row['file']
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


def find_optimal_params(y_true, probs_resnet, probs_rf):
    """
    Finds the best fusion weight (alpha) and the optimal classification
    threshold on the validation set.
    """
    print("\nFinding optimal fusion alpha on validation set...")
    best_alpha = 0
    best_roc_auc = 0

    alphas = np.arange(0, 1.01, 0.01)
    for alpha in alphas:
        fused_probs = alpha * probs_resnet + (1 - alpha) * probs_rf
        roc_auc = roc_auc_score(y_true, fused_probs)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_alpha = alpha
            
    print(f"Best alpha found: {best_alpha:.2f} with ROC AUC: {best_roc_auc:.4f}")
    
    print("Finding optimal classification threshold...")
    fused_probs = best_alpha * probs_resnet + (1 - best_alpha) * probs_rf
    fpr, tpr, thresholds = roc_curve(y_true, fused_probs)
    
    j_scores = tpr - fpr
    best_threshold_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_idx]

    print(f"Best threshold found: {best_threshold:.4f}")
    
    return best_alpha, best_threshold


def evaluate_model(name, y_true, y_prob, threshold):
    """
    Calculates and prints performance metrics using a given threshold.
    """
    y_pred = (y_prob > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"\n----- {name} -----")
    print(f"Using Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("--------------------" + "-"*len(name))
    
    return {"Accuracy": accuracy, "ROC AUC": roc_auc}


def main():
    """Main function to run the model fusion and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For demonstration, setup a dummy environment.
    # In your real workflow, you would just load your existing dataframes.

    # 1. Load models and processors
    audio_conf = AudioConfig()
    audio_processor = AudioProcessor(audio_conf)
    
    detector = DeepfakeClassifier(
        num_classes=2, config=audio_conf, dropout=0.3, initial_channels=32,
        resnet_channels=[32, 64, 128, 256], resnet_blocks=[1, 1, 1, 1],
        classifier_hidden_dim=256
    ).to(device)
    detector.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    
    clf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Get predictions on VALIDATION set
    val_y_true, val_probs_resnet, val_probs_rf = get_predictions(
        val_metadata_df, detector, clf, scaler, audio_processor, device
    )
    
    if val_y_true.size == 0:
        print("\n[CRITICAL ERROR] No samples were successfully processed from the validation set.", file=sys.stderr)
        print("Please check file paths, audio file integrity, and label values ('spoof'/'bonafide').", file=sys.stderr)
        return

    # 3. Find optimal alpha and threshold from VALIDATION predictions
    best_alpha, best_threshold = find_optimal_params(val_y_true, val_probs_resnet, val_probs_rf)
    
    # 4. Get predictions on the TEST set for final evaluation
    test_y_true, test_probs_resnet, test_probs_rf = get_predictions(
        val_metadata_df, detector, clf, scaler, audio_processor, device
    )

    if test_y_true.size == 0:
        print("\n[CRITICAL ERROR] No samples were successfully processed from the test set.", file=sys.stderr)
        return

    # 5. Evaluate baselines and the fused model on the TEST set
    print("\n--- FINAL EVALUATION ON TEST SET ---")
    
    results = {}
    results["Random Forest"] = evaluate_model("Random Forest Baseline", test_y_true, test_probs_rf, best_threshold)
    results["ResNet"] = evaluate_model("ResNet Baseline", test_y_true, test_probs_resnet, best_threshold)
    
    fused_probs = best_alpha * test_probs_resnet + (1 - best_alpha) * test_probs_rf
    results["Fused Model"] = evaluate_model(f"Fused Model (alpha={best_alpha:.2f})", test_y_true, fused_probs, best_threshold)
    
    # 6. Display summary
    print("\n--- PERFORMANCE SUMMARY ---\n")
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string(float_format="%.4f"))
    print("\n--- END OF SCRIPT ---")

if __name__ == "__main__":
    main()
