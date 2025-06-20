import os
import numpy as np
import pandas as pd
import librosa.feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from split_dataset import train_speaker, test_speaker
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report

DIR_PATH = "C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv"

params_rf = {'n_estimators': [200, 210, 230, 300, 400], 'max_depth': [7, 8, 9, 10, 11, 12], 'random_state': [42]}


AUDIO_FEATURES = [
    'zcr', 'rmse', 'mfcc', 'spectral_centroid',
    'spectral_bandwidth', 'spectral_rolloff', 'chroma'
]

def extract_features(audio: np.ndarray, sr: int) -> dict:
    """Estrai feature audio da una traccia."""
    return {
        'zcr': librosa.feature.zero_crossing_rate(audio).mean(),
        'rmse': librosa.feature.rms(y=audio).mean(),
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(),
        'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr).mean(),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sr).mean(),
        'chroma': librosa.feature.chroma_stft(y=audio, sr=sr).mean()
    }

def prepare_data(speakers: np.ndarray, path: str, csv_path: str) -> pd.DataFrame:
    """Prepara il dataset estraendo feature audio dai file specificati."""
    df = pd.read_csv(os.path.join(csv_path, "chunkedDf.csv"))
    data = []

    for speaker in speakers:
        speaker_df = df[df['speaker'] == speaker]
        for _, row in speaker_df.iterrows():
            file_path = os.path.join(path, speaker, row['file'])
            audio, sr = librosa.load(file_path, sr=None)
            features = extract_features(audio, sr)
            features.update({
                'file': row['file'],
                'speaker': speaker,
                'label': row['label']
            })
            data.append(features)

    return pd.DataFrame(data)

def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Allena il classificatore e valuta la performance con metriche complete."""
    y_train = train_df.pop('label')
    y_test = test_df.pop('label')

    X_train = train_df.drop(columns=['file', 'speaker'])
    X_test = test_df.drop(columns=['file', 'speaker'])

    # Feature scaling (opzionale ma utile per modelli diversi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modello
    clf = RandomForestClassifier()

    search = GridSearchCV(estimator=clf, param_grid=params_rf, n_jobs=-1)

    search.fit(X_train, y_train)
    print(search.best_params_)
    best_clf = search.best_estimator_

    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # Metriche
    print("Performance metrics:")
    print(f"Accuracy      : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall        : {recall_score(y_test, y_pred, average='macro'):.4f}")
    if y_prob is not None and len(np.unique(y_test)) == 2:
        print(f"ROC AUC       : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


# === ESECUZIONE ===
if __name__ == "__main__":
    train_df = prepare_data(train_speaker, DIR_PATH, DIR_PATH)
    test_df = prepare_data(test_speaker, DIR_PATH, DIR_PATH)
    train_and_evaluate(train_df, test_df)
