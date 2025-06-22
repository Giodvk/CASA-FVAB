import pandas as pd
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForPreTraining
import torch.nn as nn
from split_dataset import train_speaker, test_speaker


processor = AutoProcessor.from_pretrained('facebook/wav2vec2-large-xlsr-53')


class Wav2VecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, df, speakers, processor):
        self.data_dir = data_dir
        self.metadata = df[df['speaker'].isin(speakers)][['file', 'label']].reset_index(drop=True)
        self.processor = processor

    def __getitem__(self, idx: int):
        label_map = {'spoof': 1, 'bonafide': 0}
        if idx > len(self):
            raise IndexError("Index out of bounds")

        file = self.metadata.iloc(idx)['file']
        label = self.metadata.iloc[idx]['label']

        try:
            waveform, sample_rate = torchaudio.load(self.data_dir + file, sr=None)
        except Exception as e:
            print(f"Errore nel caricamento del file {self.data_dir + file}")
            return None, None

        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        normalized_waveform = self.processor(waveform.squeeze(0).numpy(),
                                             sampling_rate=16000, return_tensors="pt").input_values
        label_num = label_map[label]

        return normalized_waveform.unsqueeze(0), torch.tensor(label_num, dtype=torch.long)

    def __len__(self):
        return len(self.metadata)




class Wav2VecClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):

        super(Wav2VecClassifier, self).__init__()

        self.wav2vec_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        for param in self.wav2vec2_model.parameters():
            param.required_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        wav_embeddings = self.wav2vec_model(x).last_hidden_state
        pooled = wav_embeddings.mean(dim=1)
        return self.classifier(pooled)


def train(model, train_loader, loss_fn, optimizer, device):
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_acc += (output.argmax(dim=1) == y).sum().item()
        total_samples += len(y)

    return total_loss / total_samples, total_acc / total_samples


def evaluate(model, valid_loader, loss_fn, device):
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()

            total_loss += loss.item() * len(y)
            total_acc += (output.argmax(dim=1) == y).sum().item()
            total_samples += len(y)

    return total_loss / total_samples, total_acc / total_samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav2vec_model = Wav2VecClassifier()
    data_frame = pd.read_csv("C:\\Users\dmc\PycharmProjects\CASA-FVAB\meta.csv")
    train_dataset = Wav2VecDataset("C:\\Users\dmc\PycharmProjects\CASA-FVAB\\release_in_the_wild",
                                   data_frame, train_speaker, processor)
    valid_dataset = Wav2VecDataset("C:\\Users\dmc\PycharmProjects\CASA-FVAB\\release_in_the_wild",
                                   data_frame, test_speaker, processor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    wav2vec_model.to(device)
    num_epoch = 50
    optimizer = torch.optim.Adam(wav2vec_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    patience = 0
    max_patience = 20
    best_val_loss = float('inf')

    for epoch in range(1, num_epoch + 1):
          print("Epoch {}/{}".format(epoch, num_epoch))
          train_loss, train_acc = train(wav2vec_model, train_loader, loss_fn, optimizer, device)
          print(f"Train loss : {train_loss:.4f}, Train accuracy : {train_acc:.4f}")

          val_loss, val_acc = evaluate(wav2vec_model, valid_loader, loss_fn, device)
          print(f"Val loss : {val_loss:.4f}, Val accuracy : {val_acc:.4f}")

          if val_loss < best_val_loss:
              best_val_loss = val_loss
              torch.save(wav2vec_model.state_dict(), '..saved_models/wav2vec_model.pth')
              print("Saved best model")
          else:
              patience += 1
              print(f"Patience has been increased {patience}/{max_patience}")
          if patience > max_patience:
              print("Early stopping activated")
              break


if __name__ == "__main__":
    main()

