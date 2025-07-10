import torch
import torchaudio
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
from split_dataset import train_speaker, test_speaker, asv_balanced
from ASVSpoofDataset import ASVSpoofProcessor
from dataAudio import AudioConfig
import joblib

saved_models = "C:\\Users\dmc\PycharmProjects\CASA-FVAB\saved_models"


class SupportVectorSpoof:
    def __init__(self):
        self.scaler = StandardScaler()
        self.sample_rate = 16000
        self.model = SVC(kernel='rbf', gamma="scale", C=10.0, probability=True)
        self.lfcc_transform = LFCC(16000, 20, n_lfcc=20, log_lf=True)

    def extract_features(self, audio_file, sample_rate):

        cqc = librosa.cqt(audio_file, sr=sample_rate, n_bins=80, hop_length=160)
        logC = librosa.amplitude_to_db(np.abs(cqc), ref=np.max)
        cqcc = dct(logC, axis=0, type=2, norm='ortho')[:20].mean()

        audio_tensor = torch.tensor(audio_file).float()
        lfcc_features = self.lfcc_transform(audio_tensor).numpy().mean()

        analytic_signal = signal.hilbert(audio_file)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.diff(phase).mean()

        spectral_centroid = librosa.feature.spectral_centroid(y=audio_file, sr=sample_rate, hop_length=160).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_file, sr=sample_rate, hop_length=160).mean()

        zcr = librosa.feature.zero_crossing_rate(audio_file).mean()

        rsp = self._compute_rps(audio_file).mean()

        flatness = librosa.feature.spectral_flatness(
            y=audio_file, hop_length=160).mean()

        return np.array([cqcc,
                        lfcc_features.T,
                        frequency,
                        spectral_centroid,
                        spectral_contrast,
                        zcr,
                        rsp,
                        flatness])

    @staticmethod
    def _compute_rps(audio, sample_rate=16000):
        """Calcola il Relative Phase Shift con gestione robusta degli edge case"""
        # Calcola STFT con parametri ottimizzati
        stft = librosa.stft(audio,
                            n_fft=1024,
                            hop_length=160,
                            center=True,
                            pad_mode='reflect')

        if stft.size == 0:
            return np.zeros(1)  # Gestione audio vuoto

        phase = np.angle(stft)

        # Controllo dimensionale
        if phase.ndim == 1 or phase.shape[1] < 2:
            # Audio troppo corto per calcolare differenze temporali
            return np.zeros(1)

        # Calcolo differenze temporali (frame-to-frame)
        rps = np.diff(phase, axis=1) / (2 * np.pi)  # Asse CORRETTO (1)

        # Media lungo l'asse delle frequenze
        return np.mean(rps, axis=0)

    def train_evaluate(self, train_files, train_labels, test_files, test_labels):
        """Addestramento completo con gestione efficiente della memoria"""
        # Estrazione feature training
        X_train = []
        processor = ASVSpoofProcessor(config=AudioConfig(),
                                      wav2vec_model_name="C:\\Users\dmc\PycharmProjects\CASA-FVAB\wav2vec2-xlsr")
        for file in train_files:
            audio, sr = torchaudio.load(file)
            audio = processor.process_audio(audio, sr, "train").numpy()
            X_train.append(self.extract_features(audio, self.sample_rate))

        # Estrazione feature test
        X_test = []
        for file in test_files:
            audio, sr = torchaudio.load(file)
            audio = processor.process_audio(audio, sr, "eval").numpy()
            X_test.append(self.extract_features(audio, self.sample_rate))

        # Normalizzazione
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Addestramento e valutazione
        self.model.fit(X_train_scaled, train_labels)
        y_pred = self.model.predict(X_test_scaled)

        joblib.dump(self.model, saved_models + "\SupportVector.pkl")
        joblib.dump(self.scaler, saved_models + "\ScalerSVM.pkl")

        print(classification_report(test_labels, y_pred, digits=4))
        return y_pred


if __name__ == '__main__':
    model = SupportVectorSpoof()

    print("Decidere Dataset d'addestramento: 0 (In_The_Wild), 1 (ASVspoof2021)")

    choice = int(input())

    match choice:
        case 0:
            print("Inserire il path dei metadati: ")
            source, csv = input().split(" ")
            df = pd.read_csv(csv)

            train_dataset = df[df['speaker'].isin(train_speaker)].reset_index(drop=True)
            valid_dataset = df[df['speaker'].isin(test_speaker)].reset_index(drop=True)
            train_files = train_dataset.apply(lambda x: source + x['speaker'] + "\\" + x['file'], axis=1)
            valid_files = valid_dataset.apply(lambda x: source + x['speaker'] + "\\" + x['file'], axis=1)
            model.train_evaluate(train_files, train_dataset['label'], valid_files, valid_dataset['label'])
        case 1:
            train_set, valid_set = train_test_split(asv_balanced, test_size=0.25, random_state=42)
            model.train_evaluate(train_set['file'], train_set['label'],
                                 valid_set['file'], valid_set['label'])

