import numpy as np
import torch.nn as nn

# μ-law encoding and decoding
def mu_law(x, quantization_channels=256):
    mu = quantization_channels - 1
    safe_x = np.clip(x, -1, 1)
    magnitude = np.log1p(mu * np.abs(safe_x)) / np.log1p(mu)
    signal = np.sign(safe_x) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)

def inv_mu_law(x, quantization_channels=256):
    mu = quantization_channels - 1
    y = 2 * (x.astype(np.float32) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(y) - 1)
    return np.sign(y) * magnitude

# Wave augmentation for data diversity
def wave_augmentation(x, factor_range=(0.9, 1.1)):
    factor = np.random.uniform(*factor_range)
    return x * factor

# Loss function combining reconstruction and perceptual loss
class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.perceptual_loss = nn.L1Loss()

    def forward(self, output, target):
        reconstruction_loss = self.mse(output, target)
        style_loss = self.perceptual_loss(output.mean(dim=-1), target.mean(dim=-1))
        return reconstruction_loss + 0.1 * style_loss
