import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
    def forward(self, x):
        seq_len = x.size(0)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(x.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float().to(x.device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(1)  # (seq_len, batch, features)
        return x

class MusicTransformer(nn.Module):
    def __init__(self, latent_dim=128):
        super(MusicTransformer, self).__init__()
        
        # Feature Extractor
        print(1)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        print(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        print(3)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=latent_dim)
        print(4)
        # Transformer Encoder
        self.transformer = nn.Transformer(d_model=latent_dim, nhead=8, num_encoder_layers=6)
        print(5)
        # Output Layer
        self.fc_out = nn.Linear(latent_dim, 1)

    def forward(self, x):
        # Convolutional Layers
        print("phase 4.1")
        x = F.relu(self.conv1(x))
        print("phase 4.2")
        x = F.relu(self.conv2(x))
        print("phase 4.3")

        # Prepare for Transformer
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        print("phase 4.4")
        x = self.positional_encoding(x)
        print("phase 4.5")
        x = self.transformer(x, x)
        print("phase 4.6")

        # Output Layer
        x = self.fc_out(x.permute(1, 2, 0))  # (batch, features, seq_len)
        return x

class NeuralVocoder(nn.Module):
    def __init__(self):
        super(NeuralVocoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.post_processing = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.post_processing(x)
        return x
