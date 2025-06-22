import joblib
import numpy as np
import pandas as pd
import torch
from cross_validation.Test_on_ASV import df, ASVSpoofProcessor
from dataAudio import AudioConfig, AudioProcessor
from models.LightCNNRNN import LightCNNRNN
from Utility import get_predictions_test, get_predictions_train
from lateFusion import evaluate_model
from sklearn.linear_model import LogisticRegression
from transformers import Wav2Vec2Model, Wav2Vec2Processor

CNN_MODEL_PATH = "../models/best_model.pth"
RF_MODEL_PATH = "../saved_models/Random_Forest.pkl"
SCALER_PATH = "../scaler.pkl"
train_metadata = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv")
val_metadata = df[:20000]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_config = AudioConfig()
    asv_processor = ASVSpoofProcessor(config=audio_config)
    audio_processor = AudioProcessor(config=audio_config)

    cnn_rnn = LightCNNRNN().to(device=device)
    cnn_rnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))

    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    train_true, train_prob_resnet, train_prob_rf = get_predictions_train(
        train_metadata, cnn_rnn, rf, scaler, audio_processor, device
    )

    X_train_stack = np.stack([train_prob_resnet, train_prob_rf], axis=1)
    Y_train = train_true
    print(X_train_stack)

    reg = LogisticRegression()
    reg.fit(X_train_stack, Y_train)

    test_y_true, test_prob_resnet, test_prob_rf = get_predictions_test(val_metadata, cnn_rnn,
                                                                       rf, scaler, asv_processor, device)

    X_test_stack = np.stack([test_prob_resnet, test_prob_rf], axis=1)
    Y_test = test_y_true

    fused_prob = reg.predict_proba(X_test_stack)[:, 1]

    result = evaluate_model("Stacked_fusion", Y_test, fused_prob, 0.5)

if __name__ == "__main__":
    main()

