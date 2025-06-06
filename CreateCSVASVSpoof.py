import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from pydub import AudioSegment
from torch.utils.data import DataLoader
from DeepLearningModelRESNET import AudioConfig, AudioProcessor, DeepfakeDataset, DeepfakeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report


class ASVSpoofProcessor(AudioProcessor):

    def __init__(self, config):
        super(ASVSpoofProcessor, self).__init__(config)

    def extract_features(self, waveform: torch.Tensor, target_sample_rate=16000, target_duration=5):
        current_num_samples = waveform.shape[0]
        target_num_samples = int(target_duration * target_sample_rate)
        if current_num_samples == target_num_samples:
            # Già della lunghezza corretta
            processed_waveform = waveform
        elif current_num_samples > target_num_samples:
            # Audio più lungo: applica il ritaglio centrale (center cropping)
            start_sample = (current_num_samples - target_num_samples) // 2
            processed_waveform = waveform[:, start_sample: start_sample + target_num_samples]
        else:  # current_num_samples < target_num_samples
            # Audio più corto: applica il padding con silenzio alla fine
            padding_needed = target_num_samples - current_num_samples
            processed_waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return super(ASVSpoofProcessor, self).extract_features(processed_waveform)



# ----------------------------------------------------------------------------
# FUNZIONE PER CALCOLARE L'EQUAL ERROR RATE (EER)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# FUNZIONE PRINCIPALE DI TEST
# ----------------------------------------------------------------------------
def testOnASVspoof(model,
                   asv_csv_path,
                   audio_root_dir,
                   audioProcessor,
                   batch_size=64,
                   label_mapping=None,

                    ):
    """
    Valuta un modello PyTorch su un sottoinsieme di ASVspoof.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    model.to(device)
    model.eval()  # Imposta il modello in modalità valutazione

    current_label_mapping = label_mapping if label_mapping is not None else {'bona-fide': 0, 'spoof': 1}
    print(f"Utilizzo del mapping etichette: {current_label_mapping}")
    try:

        idx_to_label_name = {v: k for k, v in current_label_mapping.items()}
        report_target_names = [idx_to_label_name[i] for i in sorted(idx_to_label_name.keys())]
    except Exception:
        report_target_names = None  # Fallback

    # Crea Dataset e DataLoader
    eval_dataset = DeepfakeDataset(
        root_dir=audio_root_dir,
        metadata_df=asv_csv_path,
        processor=audioProcessor,
        augment=False
    )

    if len(eval_dataset) == 0:
        print("Il dataset di valutazione è vuoto! Controlla i percorsi e il CSV.")
        return None

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    all_true_labels = []
    all_predicted_classes = []
    all_positive_class_scores = []  # Score per la classe 'spoof' (per EER/AUC)

    print(f"Inizio valutazione su {len(eval_dataset)} campioni...")
    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for i, (batch_features, batch_true_labels) in enumerate(eval_dataloader):

            model_outputs = model(batch_features['mel'].to(device))  # Output raw del modello (logits)


            if model_outputs.ndim > 1 and model_outputs.shape[1] == 2:
                probabilities = torch.softmax(model_outputs, dim=1)

                scores_for_spoof_class = probabilities[:, current_label_mapping.get('spoof', 1)]
                _, predicted_batch_classes = torch.max(probabilities, dim=1)

            elif model_outputs.ndim == 1 or (model_outputs.ndim == 2 and model_outputs.shape[1] == 1):
                model_outputs = model_outputs.squeeze()  # Assicura sia 1D
                scores_for_spoof_class = torch.sigmoid(model_outputs)  # Probabilità
                predicted_batch_classes = (scores_for_spoof_class > 0.5).long()
            else:
                raise ValueError(
                    f"Forma dell'output del modello ({model_outputs.shape}) non gestita. "
                    "Adatta la sezione di elaborazione dell'output."
                )

            all_true_labels.extend(batch_true_labels.cpu().numpy())
            all_predicted_classes.extend(predicted_batch_classes.cpu().numpy())
            all_positive_class_scores.extend(scores_for_spoof_class.cpu().numpy())

            if (i + 1) % (max(1, len(eval_dataloader) // 10)) == 0:  # Stampa progresso circa 10 volte
                print(f"  Elaborato batch {i + 1}/{len(eval_dataloader)}")

    print("Valutazione completata. Calcolo metriche...")

    # Converti liste in array NumPy
    y_true_np = np.array(all_true_labels)
    y_pred_classes_np = np.array(all_predicted_classes)
    y_scores_spoof_np = np.array(all_positive_class_scores)

    # Calcola Metriche
    accuracy = accuracy_score(y_true_np, y_pred_classes_np)

    if len(np.unique(y_true_np)) < 2:  # Controlla se c'è almeno una istanza per ogni classe
        print("Trovata solo una classe nelle etichette vere. AUC e EER potrebbero essere non definiti o fuorvianti.")
        auc = float('nan')
        eer = float('nan')
    else:
        try:
            auc = roc_auc_score(y_true_np, y_scores_spoof_np)
        except ValueError as e:
            print(f"Errore nel calcolo AUC: {e}")
            auc = float('nan')
        try:
            # Assicurati che y_true_np contenga 0 per 'bonafide' e 1 per 'spoof' per EER
            eer = calculate_eer(y_true_np, y_scores_spoof_np)
        except Exception as e:
            print(f"Errore nel calcolo EER: {e}")
            eer = float('nan')

    print("\n--- Risultati della Valutazione ---")
    print(f"Accuratezza: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"EER: {eer:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true_np, y_pred_classes_np, target_names=report_target_names, digits=4))

    results = {"accuracy": accuracy, "auc": auc, "eer": eer}
    return results


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
            print(f"✓ Converted and replaced: {flac_file.name}")
        except Exception as e:
            print(f"✗ Failed to convert {flac_file}: {e}")

if __name__ == '__main__':
    eval_path = Path('B:/4835108/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac')
    pathKey = './trial_metadata.txt'

    #convert_flac_to_wav_inplace(eval_path)

    df = CreateCSVASVSpoof(pathKey)
    df.to_csv("ASVSpoofData.csv", index=False, columns=['file', 'label'])
    modelPath = 'best_hardened_detector.pth'
    modelArchitecture = DeepfakeClassifier()
    state_dict = torch.load(modelPath, map_location=torch.device('cuda'), weights_only=False)
    modelArchitecture.load_state_dict(state_dict['model_state_dict'])
    modelArchitecture.to(device="cuda")
    config = AudioConfig()
    processor = ASVSpoofProcessor(config)
    testOnASVspoof(
        model=modelArchitecture,
        asv_csv_path=df[:152956],
        audio_root_dir=eval_path,
        audioProcessor=processor
    )









