from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader
from ASVSpoofDataset import calculate_eer
from dataAudio import AudioProcessor, AudioConfig, DeepfakeDataset
import torch
from IntermediateCrossFusion import collate_fn_skip_none, MultiViewCollaborativeNet
def main():
    csv_path = Path('C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv')
    audio_path = Path('C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio')
    config = AudioConfig()
    audio_processor = AudioProcessor(config, wav2vec_model_name="C:\\Users\dmc\PycharmProjects\CASA-FVAB\wav2vec2-xlsr")
    df_audio = pd.read_csv(csv_path)

    test_dataset = DeepfakeDataset(audio_path, df_audio, audio_processor)

    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_skip_none)

    model = MultiViewCollaborativeNet(wav2vec_model_name="C:\\Users\dmc\PycharmProjects\CASA-FVAB\wav2vec2-xlsr")
    model.load_state_dict(torch.load('../best_collaborative_model2.pth',
                                     map_location='cuda'))
    model.to(device="cuda" if torch.cuda.is_available() else "cpu")

    test_in_the_wild(model, val_loader)


def test_in_the_wild(model, val_loader):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_scores_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x_wav2vec = x['wav2vec_input'].to(device="cuda")
            x_mel = x['mel'].to(device="cuda" if torch.cuda.is_available() else "cpu")
            logits = model(x_wav2vec, x_mel)
            if logits['final_logits'].ndim > 1 and logits['final_logits'].shape[1] == 2:
                probabilities = torch.softmax(logits['final_logits'], dim=1)
                scores_for_spoof_class = probabilities[:, 1]
                _, predicted_batch_classes = torch.max(probabilities, dim=1)
            else:
                model_outputs = logits['final_logits'].squeeze()
                scores_for_spoof_class = torch.sigmoid(model_outputs)  # Probabilità
                predicted_batch_classes = (scores_for_spoof_class > 0.5).long()

            all_true_labels.extend(y.cpu().numpy())
            all_pred_labels.extend(predicted_batch_classes.cpu().numpy())
            all_scores_labels.extend(scores_for_spoof_class.cpu().numpy())

        y_true_np = np.array(all_true_labels)
        y_pred_classes_np = np.array(all_pred_labels)
        y_scores_spoof_np = np.array(all_scores_labels)

        # Calcola Metriche
        accuracy = accuracy_score(y_true_np, y_pred_classes_np)

        if len(np.unique(y_true_np)) < 2:  # Controlla se c'è almeno una istanza per ogni classe
            print(
                "Trovata solo una classe nelle etichette vere. AUC e EER potrebbero essere non definiti o fuorvianti.")
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
        print(classification_report(y_true_np, y_pred_classes_np, digits=4))

if __name__ == "__main__":
    main()