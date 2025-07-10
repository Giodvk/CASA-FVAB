import joblib
import numpy as np
import pandas as pd
import torch
from ASVSpoofDataset import ASVSpoofProcessor
from models.SupportVector import SupportVectorSpoof
from split_dataset import asv_balanced
from dataAudio import AudioConfig, AudioProcessor
from models.LightCNNRNN import LightCNNRNN
from Utility import get_predictions_test, get_predictions_train
from lateFusion import evaluate_model
from sklearn.linear_model import LogisticRegression

CNN_MODEL_PATH = "../models/lighcnn_rnn.pth"
RF_MODEL_PATH = "../saved_models/Random_Forest.pkl"
SVM_MODEL_PATH = "../saved_models/SupportVector.pkl"
SCALER_PATH = "../saved_models/ScalerSVM.pkl"
SCALERRF_PATH = "../saved_models/RFscaler.pkl"
train_metadata = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    audio_config = AudioConfig()
    asv_processor = ASVSpoofProcessor(config=audio_config,
                                      wav2vec_model_name="C:\\Users\dmc\PycharmProjects\CASA-FVAB\wav2vec2-xlsr")
    audio_processor = AudioProcessor(config=audio_config)

    cnn_rnn = LightCNNRNN().to(device=device)
    cnn_rnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))

    rf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    scaler_rf = joblib.load(SCALERRF_PATH)
    svm = SupportVectorSpoof()
    svm.model = joblib.load(SVM_MODEL_PATH)

    train_true, train_prob_resnet, train_prob_rf, train_prob_svm = get_predictions_test(
        asv_balanced, cnn_rnn, rf, scaler, scaler_rf, svm, asv_processor, device
    )

    X_train_stack = np.stack([train_prob_resnet, train_prob_rf, train_prob_svm], axis=1)
    Y_train = train_true

    reg = LogisticRegression()
    reg.fit(X_train_stack, Y_train)

    test_y_true, test_prob_resnet, test_prob_rf, test_prob_svm = get_predictions_train(train_metadata, cnn_rnn,
                                                                       rf, scaler, scaler_rf, svm, audio_processor, device)
    X_test_stack = np.stack([test_prob_resnet, test_prob_rf, test_prob_svm], axis=1)
    Y_test = test_y_true

    fused_prob = reg.predict_proba(X_test_stack)[:, 1]

    result = evaluate_model("Stacked_fusion", Y_test, fused_prob, 0.5)

if __name__ == "__main__":
    main()

