import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import librosa
import librosa.feature as feature
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy import signal
from torchaudio.transforms import LFCC



class SupportVectorSpoof:
    def __init__(self):
        self.scaler = StandardScaler()
        self.sample_rate = 16000
        self.model = SVC(kernel='rbf', gamma="space", C=10.0, probability=True)
        self.lfcc_transform = LFCC(16000, 20, n_lfcc=20, log_lf=True)

    def extract_features(self, audio_file, sample_rate):

        cqc = librosa.cqt(audio_file, sr=sample_rate, n_bins=80, hop_length=160)
        logC = librosa.amplitude_to_db(np.abs(cqc), ref=np.max)
        cqcc = dct(logC, axis=0, norm='ortho')[:20]

        audio_tensor = torch.tensor(audio_file).float()
        lfcc_features = self.lfcc_transform(audio_tensor).numpy()

        analytic_signal = signal.hilbert(audio_file)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.diff(phase)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio_file, sr=sample_rate, hop_length=160)[np.newaxis]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_file, sr=sample_rate, hop_length=160)

        zcr = librosa.feature.zero_crossing_rate(audio_file)

        def aggregate(features):
            return np.concatenate([
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.median(features, axis=1),
                np.max(features, axis=1) - np.min(features, axis=1)
                ]
            )
        return np.concatenate([aggregate(cqcc),
                              aggregate(lfcc_features.T),
                              aggregate(frequency[np.newaxis, :]),
                              aggregate(spectral_centroid),
                              aggregate(spectral_contrast),
                              aggregate(zcr)])

    def train_evaluate(self, train_files, train_labels, test_files, test_labels):
        """Addestramento completo con gestione efficiente della memoria"""
        # Estrazione feature training
        X_train = []
        for file in train_files:
            audio, _ = librosa.load(file, sr=self.sample_rate)
            X_train.append(self.extract_features(audio, self.sample_rate))

        # Estrazione feature test
        X_test = []
        for file in test_files:
            audio, _ = librosa.load(file, sr=self.sample_rate, duration=5)
            X_test.append(self.extract_features(audio, self.sample_rate))

        # Normalizzazione
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Addestramento e valutazione
        self.model.fit(X_train_scaled, train_labels)
        y_pred = self.model.predict(X_test_scaled)

        print(classification_report(test_labels, y_pred, digits=4))
        return y_pred


if __name__ == '__main__':
    model = SupportVectorSpoof()

    print("Decidere Dataset d'addestramento: 0 (In_The_Wild), 1 (ASVspoof2021)")

    choice = int(input())

    print("Inserire il path dei metadati: ")

    source, csv = input().split(" ")

    match choice:
        case 0:
           print("niente")
        case 1:
            df = pd.read_csv(csv)
            train_set, valid_set = train_test_split(df, test_size=0.25, random_state=42)
            model.train_evaluate(train_set['file'], train_set['label'], valid_set['file'], valid_set['label'])

