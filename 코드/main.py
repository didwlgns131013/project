import torch
from torch.utils.data import DataLoader
from data import MultiMusicDataset, merge_kpop_with_carol
from models import MusicTransformer, NeuralVocoder
from utils import MixedLoss
from train import train_model

def main():
    # Step 1: Define paths and parameters
    print("Starting data processing...")
    kpop_mp3_paths = [
        r"/content/drive/MyDrive/기계학습/음악/Kpop-1.mp3",
        r"/content/drive/MyDrive/기계학습/음악/Kpop-2.mp3", 
        r"/content/drive/MyDrive/기계학습/음악/Kpop-3.mp3",
        r"/content/drive/MyDrive/기계학습/음악/Kpop-4.mp3", 
        r"/content/drive/MyDrive/기계학습/음악/Kpop-5.mp3", 
        r"/content/drive/MyDrive/기계학습/음악/Kpop-6.mp3"
    ]  # List of K-pop MP3 file paths
    carol_mp3_path = r"/content/drive/MyDrive/기계학습/음악/Carol.mp3"  # Carol MP3 file path
    output_pt_file = "merged_data.pt"  # Output .pt file for dataset
    print("Data processing completed.")

    # Step 2: Merge the K-pop and Carol audio files into a dataset
    print("Data merge processign...")
    merge_kpop_with_carol(kpop_mp3_paths, carol_mp3_path, output_pt_file)
    print("Data merge completed.")

    # Step 3: Load the dataset
    print("phase 1")
    data_paths = [output_pt_file]  # In this example, only one .pt file is used
    print("phase 2")
    dataset = MultiMusicDataset(data_paths)
    print("phase 3")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("phase 4")

    # Step 4: Define the model and optimizer
    model = MusicTransformer(latent_dim=128).cuda()
    print("phase 5")
    vocoder = NeuralVocoder().cuda()
    print("phase 6")
    loss_fn = MixedLoss()
    print("phase 7")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Step 5: Train the model
    print("Starting model training...")
    train_model(model, vocoder, dataloader, optimizer, loss_fn, epochs=50)
    print("Model training completed.")

    # Step 6: Save the trained model
    torch.save(model.state_dict(), "music_transformer.pth")
    torch.save(vocoder.state_dict(), "neural_vocoder.pth")

if __name__ == "__main__":
    main()
