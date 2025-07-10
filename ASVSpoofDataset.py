import os
from pathlib import Path
from typing import Dict
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from dataAudio import AudioProcessor
from sklearn.metrics import roc_curve


class ASVSpoofProcessor(AudioProcessor):
    def __init__(self, config, target=5, target_sample_rate=16000, wav2vec_model_name: str = "facebook/wav2vec2-large-xls-r-300m"):
        super(ASVSpoofProcessor, self).__init__(config, wav2vec_model_name)
        self.target_samples = int(target * target_sample_rate)

    def process_audio(self, waveform: torch.Tensor, original_sample_rate: int, mode: str = 'eval') -> torch.Tensor:

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        current_samples = waveform.shape[1]

        if current_samples == self.target_samples:
            return waveform

        # Caso 2: Audio più lungo del target
        elif current_samples > self.target_samples:
            if mode == 'train':
                return self._random_crop_gaussian(waveform)
            else:
                return self._center_crop(waveform)

        # Caso 3: Audio più corto del target
        else:
            # Tecnica 3: Padding intelligente con estensione contestuale
            return self._contextual_padding(waveform)

    def _center_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        start = max(0, (waveform.shape[1] - self.target_samples) // 2)
        return waveform[:, start:start + self.target_samples]

    def _random_crop_gaussian(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[1]
        std_dev = length / 8

        mean = length // 2
        start = int(torch.normal(mean, std_dev, (1,)).clip(0, length - self.target_samples))
        return waveform[:, start:start + self.target_samples]

    def _contextual_padding(self, waveform: torch.Tensor) -> torch.Tensor:
        current_samples = waveform.shape[1]
        padding_needed = self.target_samples - current_samples
        left_extension = None
        right_extension = None

        # Calcola il padding per ogni lato
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        context_size = min(500, current_samples // 2)

        # Prepara l'estensione sinistra
        if left_pad > 0:
            left_context = waveform[:, :context_size]
            left_extension = self._fade_extend(left_context, left_pad, side='left')

        # Prepara l'estensione destra
        if right_pad > 0:
            right_context = waveform[:, -context_size:]
            right_extension = self._fade_extend(right_context, right_pad, side='right')

        # Combina i segmenti
        return torch.cat([
            left_extension if left_pad > 0 else torch.tensor([], dtype=waveform.dtype),
            waveform,
            right_extension if right_pad > 0 else torch.tensor([], dtype=waveform.dtype)
        ], dim=1)

    def _fade_extend(self, segment: torch.Tensor, target_length: int, side: str = 'right') -> torch.Tensor:
        repeated = segment.repeat(1, (target_length // segment.shape[1]) + 2)
        repeated = repeated[:, :target_length]

        fade_window = torch.hann_window(2 * repeated.shape[1] + 1)

        if side == 'right':
            fade_in = fade_window[:repeated.shape[1]]
            fade_out = fade_window[-repeated.shape[1]:]
            return repeated * fade_out.unsqueeze(0)
        else:
            fade_in = fade_window[:repeated.shape[1]]
            return repeated * fade_in.unsqueeze(0)

    def extract_features(self, waveform: torch, original_sample_rate: int = 16000, mode='train') -> Dict:
        features = {}
        processed_audio = self.process_audio(waveform, original_sample_rate, mode)
        features['mel'] = self.compute_mel(processed_audio)
        features['wav2vec_input'] = self.process_for_wav2vec(processed_audio)
        return features


class ASVspoofDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, metadata_csv: pd.DataFrame, target_duration: int, processor: ASVSpoofProcessor,
                 mode: str = 'train') -> None:
        self.root_dir = root_dir
        self.metadata_csv = metadata_csv
        self.target_duration = target_duration
        self.processor = processor
        self.mode = mode

    def __len__(self) -> int:
        return len(self.metadata_csv)

    def __getitem__(self, idx):

        row = self.metadata_csv.iloc[idx]
        path = os.path.join(self.root_dir, row['file'])
        waveform, original_sample_rate = torchaudio.load(path, normalize=True)
        label = 1 if self.metadata_csv.iloc[idx]['label'] == 'spoof' else 0
        features_waveform = self.processor.extract_features(waveform, original_sample_rate, self.mode)

        return {"features": features_waveform, "labels": torch.tensor(label, dtype=torch.long)}


def calculate_eer(y_true, y_scores_positive_class):
    """
    Calcola l'Equal Error Rate (EER).
    y_true: Etichette binarie vere (0 o 1).
    y_scores_positive_class: Score del modello per la classe positiva (es. 'spoof').
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores_positive_class, pos_label=1)  # Assumiamo 1 = spoof
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_value = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer_value * 100  # Riportato come percentuale


def CreateCSVASVSpoof(pathKey):
    data = []
    label = {'bonafide': 'bona-fide', 'spoof': 'spoof'}
    with open(pathKey,'r') as csvfile:
        for line in csvfile.readlines():
            df = {}
            split_line = line.split(' ')
            name_audio = split_line[1]
            label_audio = label[split_line[5]]
            df.update({'file': name_audio+".wav", 'label': label_audio})
            data.append(df)
    return pd.DataFrame(data)


def convert_flac_to_wav_inplace(input_dir: Path):
    input_dir = Path(input_dir)

    for flac_file in input_dir.rglob("*.flac"):
        try:
            print(f"Converting: {flac_file}")
            audio = AudioSegment.from_file(flac_file, format="flac")

            wav_path = flac_file.with_suffix('.wav')
            audio.export(wav_path, format="wav")

            flac_file.unlink()  # rimuove il file .flac originale
            print(f" Converted and replaced: {flac_file.name}")
        except Exception as e:
            print(f" Failed to convert {flac_file}: {e}")


def create_balanced_dataset(df_original, output_csv, total_samples, folders):

    # Raggruppa i file per tipo e cartella
    file_dict = {
        'spoof': defaultdict(list),
        'bona-fide': defaultdict(list)
    }

    for idx, row in df_original.iterrows():
        label = row['label']
        file_name = row['file']

        # Verifica in quale cartella esiste il file
        for folder in folders:
            full_path = os.path.join(folder, file_name)
            if os.path.exists(full_path):
                file_dict[label][folder].append(full_path)
                break

    # Calcola il numero di campioni per classe
    len_spoof = sum(x for x in [len(lista) for lista in file_dict['spoof'].values()])
    len_real = sum(x for x in [len(lista) for lista in file_dict['bona-fide'].values()])
    n_per_class = total_samples // 2
    # Campiona i file per ogni classe
    spoof_files = sample_class_files('spoof', min(n_per_class, len_spoof), file_dict, folders)
    real_files = sample_class_files('bona-fide', min(n_per_class, len_real), file_dict, folders)

    # Crea il nuovo dataframe
    balanced_data = []
    for file_path in spoof_files:
        balanced_data.append({'file': file_path, 'label': 'spoof'})

    for file_path in real_files:
        balanced_data.append({'file': file_path, 'label': 'bona-fide'})

    # Mescola il dataset
    random.shuffle(balanced_data)

    # Salva il nuovo CSV
    balanced_df = pd.DataFrame(balanced_data)
    balanced_df.to_csv(output_csv, index=False)

    print(f"Creato dataset bilanciato con {len(balanced_df)} campioni")
    print(f"- Spoof: {len(spoof_files)}")
    print(f"- Real: {len(real_files)}")
    return balanced_df

def sample_class_files(target, n_samples, dict_file, folders):
    sampled_files = []
    while len(sampled_files) < n_samples:
        # Seleziona una cartella casuale
        fold = random.choice(folders)

        # Se non ci sono file in questa cartella, passa alla prossima
        if not dict_file[target][fold]:
            continue

        # Seleziona un file casuale dalla cartella
        path = random.choice(dict_file[target][fold])

        # Se il file è già stato selezionato, salta
        if path in sampled_files:
            continue

        # Aggiungi il file ai campionati
        sampled_files.append(path)
        # Rimuovi il file dalla lista per evitare duplicati
        dict_file[target][fold].remove(path)
    return sampled_files


if __name__ == "__main__":
    Asvdf = CreateCSVASVSpoof("C:\\Users\dmc\PycharmProjects\CASA-FVAB\\trial_metadata.txt")








