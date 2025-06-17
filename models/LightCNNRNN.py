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
        self.classifier = nn.Linear(rnn_hidden_size * 2, 1)

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
    spoof = sample[sample['label'] == 'spoof']
    bona_fide = sample[sample['label'] == 'bona-fide']

    pos_weight = len(bona_fide) / len(spoof)

    return torch.tensor(pos_weight)


def get_batch_accuracy(output, label):
    probabilities = torch.sigmoid(output)

    pred = (probabilities > 0.5).long()

    correct = (pred == label).sum().item()

    return correct

def train(model: LightCNNRNN, train_loader: torch.utils.data.DataLoader, criterion: torch.nn.BCEWithLogitsLoss,
          device: torch.device, optimizer: torch.optim.Optimizer):
    model.train()
    accuracy = 0.0
    loss = 0.0
    total_samples = 0
    for x, y in train_loader:
        # Prepara i dati
        mel_spectrograms = x['mel'].to(device)
        labels = y.to(device)
        # Forward pass
        output = model(mel_spectrograms)
        # Calcola la loss
        batch_loss = criterion(output, labels.float().unsqueeze(1))
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Aggiorna le metriche
        loss += batch_loss.item() * len(labels)  # Pesa la loss per la dimensione del batch
        accuracy += get_batch_accuracy(output, labels.unsqueeze(1))
        total_samples += len(labels)

    epoch_loss = loss / total_samples
    epoch_accuracy = accuracy / total_samples
    print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))


def validate(model, valid_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in valid_loader:
            mel_spectrograms = x['mel'].to(device)
            labels = y.to(device)

            output = model(mel_spectrograms)
            loss = criterion(output, labels.float().unsqueeze(1))

            total_loss += loss.item() * len(labels)
            total_correct += get_batch_accuracy(output, labels.unsqueeze(1))
            total_samples += len(labels)

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    print('Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))



def prepare_loader(samples: pd.DataFrame, train_speakers: pd.DataFrame, valid_speakers: pd.DataFrame,
                   batch_size: int = 64):
    config = AudioConfig()
    processor = AudioProcessor(config)

    train_speakers = samples[samples['speaker'].isin(train_speakers)].reset_index(drop=True)
    valid_speakers = samples[samples['speaker'].isin(valid_speakers)].reset_index(drop=True)
    weights = compute_loss_weight(train_speakers)

    train_dataset = DeepfakeDataset(Path("./processed_audio"), train_speakers, processor)
    valid_dataset = DeepfakeDataset(Path("./processed_audio"), valid_speakers, processor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, weights


def main():
    df = pd.read_csv(DIR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightCNNRNN()
    model.to(device)
    torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, valid_loader, pos_weight = prepare_loader(df, train_speaker, valid_speakers=test_speaker, batch_size=256)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_epochs = 20

    for i in range(num_epochs):
        print("Epoch : {}".format(i + 1))
        train(model, train_loader, loss_fn, device, optimizer)
        validate(model, valid_loader, loss_fn, device)
    torch.save(model, "LightCNNRNN.pth")


if __name__ == "__main__":
    main()

