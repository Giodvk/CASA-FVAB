import torch
import torch.nn as nn
from keras.src.utils.jax_layer_test import num_classes

from DeepLearningModelRESNET import DeepfakeDataset, AudioProcessor, AudioConfig
from DataBalancingDeepSeek import train_speaker, test_speaker
from light_cnn import LightCNN_29Layers


class ModifiedLightCNN(LightCNN_29Layers):
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
    def __init__(self, cnn_output_channel = 128, cnn_final_height= 8, cnn_final_width = 8, rnn_hidden_size = 128):
        super().__init__()

        #CNN Part
        self.cnn = ModifiedLightCNN(num_classes = 1)

        # RNN Part
        GRU_input_size = cnn_output_channel * cnn_final_height * cnn_final_width

        self.rnn = nn.GRU(GRU_input_size, rnn_hidden_size,
                          bidirectional=True, dropout=0.3,
                          num_layers=2, batch_first=True)
        # Multy Layer Perceptron
        self.classifier = nn.Linear(rnn_hidden_size * 2, 2)

    def forward(self, x):
        # 1. Passa attraverso la CNN
        # x shape: [B, 1, H_in, W_in] (es: [64, 1, 64, 512])
        x = self.cnn.forward(x)
        # x shape: [B, C, H_out, W_out] (es: [64, 128, 4, 32])

        # 2. Permuto per la RNN [Batch, Sequence_lenght, Features]
        x = x.permute(0, 3, 1, 2)
        x = torch.Flatten(x, start_dim=2)

        # 3. Passo di Forward della GRU
        rnn_output, _ = self.rnn(x)
        # rnn_output shape: [B, W_out, rnn_hidden_size * 2] (es: [64, 32, 256])
        last_time_step_output = rnn_output[:, -1, :]

        # 4. Passo di forward dell'MLP
        classifier_output = self.classifier(last_time_step_output)

        return classifier_output



def main():
    train_df = DeepfakeDataset(train_speaker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_conf = AudioConfig()
    processor = AudioProcessor(audio_conf)
