from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from torch import optim

from DeepLearningModel import DeepfakeDataset, AudioProcessor, AudioConfig
from split_dataset import train_speaker, test_speaker
from light_cnn import network_29layers_v2, resblock


DIR_PATH = Path('C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio\chunkedDf.csv')


class ModifiedLightCNN(network_29layers_v2):
    """
    Versione modificata della LightCNN che restituisce l'output
    dell'ultimo blocco convoluzionale, prima dei layer Fully Connected.
    """
    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        # Passaggi presi dal forward originale, ma fermati prima di fc1
        x = self.features(x)
        # L'output di self.features è l'output dell'ultimo blocco conv
        return x


class LightCNNRNN(nn.Module):
    def __init__(self, cnn_output_channel=128, cnn_final_height=4, rnn_hidden_size=128):
        super().__init__()

        # CNN Part
        self.cnn = ModifiedLightCNN(block=resblock, layers=[1, 2, 3, 4], num_classes=1)

        # RNN Part
        GRU_input_size = cnn_output_channel * cnn_final_height

        self.rnn = nn.GRU(GRU_input_size, rnn_hidden_size,
                          bidirectional=True, dropout=0.3,
                          num_layers=2, batch_first=True)
        # Multy Layer Perceptron
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size*2, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(rnn_hidden_size, 2)
        )


    def forward(self, x):
        # 1. Passa attraverso la CNN
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.flatten(x, 2)

        # 3. Passo di Forward della GRU
        rnn_output, _ = self.rnn(x)
        last_time_step_output = rnn_output[:, -1, :]
        # 4. Passo di forward dell'MLP
        classifier_output = self.classifier(last_time_step_output)

        return classifier_output


def compute_loss_weight(sample: pd.DataFrame) -> torch.Tensor:
    count_bonafide = (sample['label'] == 'bona-fide').sum()
    count_spoof = (sample['label'] == 'spoof').sum()
    total = count_bonafide + count_spoof

    weight_bonafide = total / (2.0 * count_bonafide)
    weight_spoof = total / (2.0 * count_spoof)

    weights = torch.tensor([weight_bonafide, weight_spoof], dtype=torch.float)
    return weights


def get_batch_accuracy(output, label):
    pred = output.argmax(dim=1)
    correct = (pred == label).sum().item()
    return correct


def train(model, train_loader, criterion, device, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for x, y in train_loader:
        mel_spectrograms = x['mel'].to(device)
        labels = y.to(device)
        output = model(mel_spectrograms)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += get_batch_accuracy(output, labels)
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, valid_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in valid_loader:
            mel_spectrograms = x['mel'].to(device)
            labels = y.to(device)
            mel_spectrograms = mel_spectrograms.unsqueeze(1)
            output = model(mel_spectrograms)
            loss = criterion(output, labels)

            total_loss += loss.item() * len(labels)
            total_correct += get_batch_accuracy(output, labels)
            total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples



def prepare_loader(samples: pd.DataFrame, train_speakers: pd.DataFrame, valid_speakers: pd.DataFrame,
                   batch_size: int = 64):
    config = AudioConfig()
    processor = AudioProcessor(config)

    train_speakers = samples[samples['speaker'].isin(train_speakers)].reset_index(drop=True)
    valid_speakers = samples[samples['speaker'].isin(valid_speakers)].reset_index(drop=True)
    weights = compute_loss_weight(train_speakers)

    train_dataset = DeepfakeDataset(Path("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio"), train_speakers, processor)
    valid_dataset = DeepfakeDataset(Path("C:\\Users\dmc\PycharmProjects\CASA-FVAB\processed_audio"), valid_speakers, processor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, weights


def main():
    df = pd.read_csv(DIR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightCNNRNN().to(device)
    torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loader, valid_loader, pos_weight = prepare_loader(df, train_speaker, valid_speakers=test_speaker, batch_size=128)
    loss_fn = nn.CrossEntropyLoss()

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_epoch = -1

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = validate(model, valid_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n Best model at epoch {best_epoch} with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

