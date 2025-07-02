import random
import os
from tqdm import tqdm
import soundfile
from voicefixer import VoiceFixer
import audiomentations
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

DIR_PATH = 'C:/Users/dmc/PycharmProjects/CASA-FVAB/processed_audio/'
AUG_PATH = "./augmented_data/"
voice_fixer = VoiceFixer()

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

    def augmentation(self, df):
        transform = audiomentations.Compose([
            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            audiomentations.Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.3)
        ])

        # Step 2: Creazione dataset aumentato
        augmented_data = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if row['label'] == 0:
                augmented_data.append(balancer._augment_audio(row, transform))

            if random.random() < 0.3:  # 30% di real augmentati
                augmented_data.append(balancer._augment_audio(row, transform, fixer_prob=0))

        # Step 3: Unione con metadati
        aug_df = pd.DataFrame(augmented_data)
        final_df = pd.concat([df, aug_df], ignore_index=True)

        # Step 4: Salvataggio
        final_df.to_csv('augmented_dataset.csv', index=False)

    def _extract_mfcc(self, row, path):
        """Estrae features MFCC da un file audio"""
        audio, sr = librosa.load(f"{path}/{row['speaker']}/{row['file']}", sr=None)
        return np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

    def _augment_audio(self, row, composer, fixer_prob: float = 0.2):
        """Genera un sample aumentato"""
        prob = random.random()

        if prob < fixer_prob:

            cuda = torch.cuda.is_available()
            os.makedirs(AUG_PATH + row['speaker'], exist_ok=True)
            voice_fixer.restore(input=os.path.join(DIR_PATH, f"{row['speaker']}/{row['file']}"),
                                output=os.path.join(AUG_PATH, f"{row['speaker']}/{row['file'][:-4]}_voice.wav")
                                , cuda=cuda, mode=0)
            return {
                'file': f"{row['file']}_voice.wav",
                'speaker': row['speaker'],
                'label': row['label']
            }

        audio, sr = librosa.load(f"{DIR_PATH}{row['speaker']}/{row['file']}", sr=None)
        audio = composer(audio, sr)
        os.makedirs(AUG_PATH + row['speaker'], exist_ok=True)
        soundfile.write(os.path.join(AUG_PATH, f"{row['speaker']}/{row['file'][:-4]}_augmented.wav"), audio, samplerate=sr)
        return {
            'file': f"{row['file']}_augmented.wav",
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


# Esempio di utilizzo
if __name__ != "__main__":
    # Inizializzazione
    balancer = AudioBalancer(
        reduction_threshold=0.5,
        augmentation_threshold=0.2,
        n_clusters=400
    )

    # Caricamento dati
    data = pd.read_csv(DIR_PATH + 'chunkedDf.csv')

    for _, row in data.iterrows():
        if row['label'] == 'spoof':
            row['label'] = 1
        else :
            row['label'] = 0

    #balancer.augmentation(data)
