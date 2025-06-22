from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from torch.utils.data import DataLoader
from dataAudio import AudioConfig
from ASVSpoofDataset import ASVspoofDataset, calculate_eer, CreateCSVASVSpoof, ASVSpoofProcessor

import torch

from models.LightCNNRNN import LightCNNRNN


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
    except RuntimeError:
        report_target_names = None  # Fallback

    # Crea Dataset e DataLoader
    eval_dataset = ASVspoofDataset(
        root_dir=audio_root_dir,
        metadata_csv=asv_csv_path,
        processor=audioProcessor,
        target_duration=5,
        mode="eval"
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
            model_outputs = model(batch_features.to(device))  # Output raw del modello (logits)

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


if __name__ == '__main__':
    eval_path = Path('B:/4835108/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac')
    pathKey = 'C:/Users/dmc/PycharmProjects/CASA-FVAB/trial_metadata.txt'

    # convert_flac_to_wav_inplace(eval_path)

    df = CreateCSVASVSpoof(pathKey)
    df.to_csv("ASVSpoofData.csv", index=False, columns=['file', 'label'])
    config = AudioConfig()
    processor = ASVSpoofProcessor(config=config, target=5)
    cnn_rnn = LightCNNRNN().to(device="cuda")
    cnn_rnn.load_state_dict(torch.load("C:/Users/dmc/PycharmProjects/CASA-FVAB/models/best_model.pth",
                                       map_location="cuda"))
    prova = df[:152956]
    spoof = prova[prova['label'] == 'spoof']
    bona = prova[prova['label'] == 'bona-fide']
    final = pd.concat([bona, spoof[:5536]])
    testOnASVspoof(cnn_rnn, final, eval_path, processor)
