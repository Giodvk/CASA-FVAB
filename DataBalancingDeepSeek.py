import random

import numpy as np
import pandas as pd
import librosa
import soundfile
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from voicefixer import VoiceFixer
import pytsmod as tsm

DIR_PATH = './processed_audio/'
voice_fixer = VoiceFixer() #Da usare solo con GPU


class AudioBalancer:
    def __init__(self, reduction_threshold=0.4, augmentation_threshold=0.2,
                 n_clusters=1000, augmentation_factor=1.5):
        self.reduction_threshold = reduction_threshold
        self.augmentation_threshold = augmentation_threshold
        self.n_clusters = n_clusters
        self.augmentation_factor = augmentation_factor


    def _cluster_reduction(self, features, files):
        """Seleziona i campioni più rappresentativi usando clustering"""
        model = KMeans(n_clusters=self.n_clusters, init='k-means++')
        labels = model.fit_predict(features)

        closest_indices = []
        for cluster_id in range(model.n_clusters):
            cluster_mask = np.where(labels == cluster_id)[0]
            cluster_samples = features[cluster_mask]
            distances = np.linalg.norm(cluster_samples - model.cluster_centers_[cluster_id], axis=1)

            # Trova l'indice GLOBALE del campione più vicino
            closest_local_idx = np.argmin(distances)
            closest_global_idx = cluster_mask[closest_local_idx]
            closest_indices.append(closest_global_idx)

        return [files[i] for i in closest_indices]

    def _balance_speaker(self, speaker_df, path):
        """Bilancia i dati per un singolo speaker"""
        spoof = speaker_df[speaker_df['label'] == 0]
        bona_fide = speaker_df[speaker_df['label'] == 1]

        # Calcola lo sbilanciamento
        imbalance = (len(bona_fide) - len(spoof)) / (len(bona_fide) + 1e-8)

        # Data Reduction per sbilanciamenti estremi
        if abs(imbalance) > self.reduction_threshold:
            if len(bona_fide) > len(spoof):
                mfcc_features = [self._extract_mfcc(row,path) for _, row in bona_fide.iterrows()]
                reduced_files = self._cluster_reduction(np.array(mfcc_features), bona_fide['file'].tolist())
                bona_fide = bona_fide[bona_fide['file'].isin(reduced_files)]
            else:
                mfcc_features = [self._extract_mfcc(row, path) for _, row in spoof.iterrows()]
                reduced_files = self._cluster_reduction(np.array(mfcc_features), spoof['file'].tolist())
                spoof = spoof[spoof['file'].isin(reduced_files)]

        return pd.concat([spoof, bona_fide])

    def augmentation(self, speaker_df):
        spoof = speaker_df[speaker_df['label'] == 0]
        bona_fide = speaker_df[speaker_df['label'] == 1]

        # Calcola lo sbilanciamento
        imbalance = (len(bona_fide) - len(spoof)) / (len(bona_fide) + 1e-8)

        # Data Augmentation per sbilanciamenti residui
        if abs(imbalance) > self.augmentation_threshold:
            if len(spoof) < len(bona_fide):
                augment_class = spoof
                target_size = int(len(bona_fide) * self.augmentation_factor)
                print("Augmentation")
                augmented_samples = []
                while len(augmented_samples) < (target_size - len(augment_class)):
                    sample = augment_class.sample(1).iloc[0]
                    augmented_samples.append(self._augment_audio(sample))

                return pd.concat([spoof, bona_fide, pd.DataFrame(augmented_samples)])


    def _extract_mfcc(self, row, path):
        """Estrae features MFCC da un file audio"""
        audio, sr = librosa.load(f"{path}/{row['speaker']}/{row['file']}", sr=None)
        return np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

    def _augment_audio(self, row):
        """Genera un sample aumentato"""
        audio, sr = librosa.load(f"{DIR_PATH}{row['speaker']}/{row['file']}", sr=None)
        print(audio.shape)
        audioaug = voice_fixer.restore(np.ravel(audio), sr, mode=2, cuda=True)
        audioaug = tsm.wsola(audioaug, 1.25)

        soundfile.write(f"{DIR_PATH}{row['speaker']}/{row['file']}_augmented", audioaug, samplerate=sr)
        print(soundfile)
        return {
            'file': f"{row['file']}_augmented",
            'speaker': row['speaker'],
            'label': row['label']
        }

    def balance_dataset(self, df, path):
        """Applica il bilanciamento a tutto il dataset"""
        balanced_dfs = []
        for speaker in df['speaker'].unique():
            speaker_df = df[df['speaker'] == speaker]
            if(len(speaker_df) > 500):
                balanced_dfs.append(self._balance_speaker(speaker_df, path))
            else:
                balanced_dfs.append(speaker_df)

        return pd.concat(balanced_dfs).reset_index(drop=True)


class AudioDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if 'audio' not in row:  # Caricamento lazy per non aumentare la RAM
            audio, sr = librosa.load(f"{DIR_PATH}{row['speaker']}/{row['file']}", sr=None)
        else:
            audio, sr = row['audio'], 22050

        if self.augment and self.augmenter:
            audio = self.augmenter(samples=audio, sample_rate=sr)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return torch.FloatTensor(mfcc), torch.tensor(row['label'])


# Esempio di utilizzo
if __name__ == "__main__":
    # Inizializzazione
    balancer = AudioBalancer(
        reduction_threshold=0.5,
        augmentation_threshold=0.2,
        n_clusters=400
    )

    # Caricamento dati
    df = pd.read_csv('./processed_audio/chunkedDf.csv')
    for i in range(len(df['label'])):
        if df['label'][i] == 'spoof':
            df.loc[i, 'label'] = 0
        else:
            df.loc[i, 'label'] = 1
    #balanced_df = balancer.balance_dataset(df, DIR_PATH)
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker]
        augmented_df = balancer.augmentation(speaker_df)
    # Salvataggio del dataset bilanciato
    #balanced_df.to_csv('balanced_dataset.csv', index=False)

    speaker = df['speaker'].unique()
    train_speaker, test_speaker = train_test_split(speaker, test_size=0.2)