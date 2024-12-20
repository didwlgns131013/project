import torch
from torch.utils.data import DataLoader
from data import MultiMusicDataset
from models import MusicTransformer, NeuralVocoder
from utils import MixedLoss

def evaluate_model(model, vocoder, dataloader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (input_data, target_data) in enumerate(dataloader):
            # Forward Pass
            transformed = model(input_data)
            reconstructed = vocoder(transformed)

            # Compute Loss
            loss = loss_fn(reconstructed, target_data)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    # Parameters
    test_data_path = [r"/content/drive/MyDrive/기계학습/음악/Kpop-1.mp3"]
    batch_size = 16

    # Dataset and DataLoader
    test_dataset = MultiMusicDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load Pretrained Model and Vocoder
    model = MusicTransformer(latent_dim=128).cuda()
    vocoder = NeuralVocoder().cuda()
    model.load_state_dict(torch.load("music_transformer.pth"))
    vocoder.load_state_dict(torch.load("neural_vocoder.pth"))

    # Loss Function
    loss_fn = MixedLoss()

    # Evaluation
    evaluate_model(model, vocoder, test_dataloader, loss_fn)
