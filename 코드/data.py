import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pydub import AudioSegment
import librosa

# MultiMusicDataset: 여러 데이터셋을 통합 관리
class MultiMusicDataset(Dataset):
    def __init__(self, data_paths):
        self.data = []
        for path in data_paths:
            dataset = torch.load(path, weights_only = True)  # Load each dataset
            self.data.extend(dataset)
    
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return len(self.data)

# WaveData: Raw audio 데이터를 다루기 위한 클래스
class WaveData:
    def __init__(self, paths):
        self.data = [np.load(path, weights_only = True) for path in paths]
        
        for path, data in zip(paths, self.data):
            print(f"{path} length: {len(data)}")
        
    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(DatasetWrapper(self.data),
                          batch_size=batch_size,
                          shuffle=shuffle)

# DatasetWrapper: 데이터 증강 및 μ-law 인코딩 처리
class DatasetWrapper(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = min([len(d) for d in self.data])
        self.data_mu = [np.array(list(map(self.mu_law, data))) for data in self.data]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        out = dict()
        out_aug = dict()
        
        for i, (data, data_mu) in enumerate(zip(self.data, self.data_mu)):
            try:
                out[i] = data_mu[index]
                out_aug[i] = self.mu_law(self.wave_augmentation(data[index]))
            except IndexError:
                idx = index - len(data)
                out[i] = data_mu[idx]
                out_aug[i] = self.mu_law(self.wave_augmentation(data[idx]))
        
        return out, out_aug

    @staticmethod
    def mu_law(x, quantization_channels=256):
        mu = quantization_channels - 1
        safe_x = np.clip(x, -1, 1)
        magnitude = np.log1p(mu * np.abs(safe_x)) / np.log1p(mu)
        signal = np.sign(safe_x) * magnitude
        return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)

    @staticmethod
    def wave_augmentation(x):
        factor = np.random.uniform(0.9, 1.1)
        return x * factor

# Function to merge multiple K-pop files with a single Carol file
def merge_kpop_with_carol(kpop_paths, carol_path, output_file):
    """
    kpop_paths: K-pop MP3 파일 경로 리스트
    carol_path: 캐롤 MP3 파일 경로
    output_file: 저장할 .pt 파일 이름
    """
    data = []

    # Load and process Carol audio
    print("carol load")
    carol_audio = AudioSegment.from_file(carol_path, format="mp3")
    print(f"Carol audio length: {len(carol_audio)} ms")

    # Limit file length to 30 seconds
    carol_audio = carol_audio[:30 * 1000]

    # Convert Carol audio to NumPy array
    samples = np.array(carol_audio.get_array_of_samples()).astype(float)

    # Resample in chunks to avoid memory issues
    chunk_size = 16000 * 10  # 10 seconds
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    resampled_chunks = []
    print("carol resample")
    for chunk in chunks:
        resampled_chunk = librosa.resample(chunk, orig_sr=carol_audio.frame_rate, target_sr=16000)
        resampled_chunks.append(resampled_chunk)

    # Combine resampled chunks
    resampled_audio = np.concatenate(resampled_chunks)
    print(f"Resampled Carol audio shape: {resampled_audio.shape}")

    for kpop_path in kpop_paths:
        # Load and process each K-pop audio file
        kpop_audio = AudioSegment.from_file(kpop_path, format="mp3")
        print(f"K-pop audio length: {len(kpop_audio)} ms")

        # Limit file length to 30 seconds
        kpop_audio = kpop_audio[:30 * 1000]

        # Convert K-pop audio to NumPy array
        kpop_samples = np.array(kpop_audio.get_array_of_samples()).astype(float)

        # Resample in chunks to avoid memory issues
        kpop_chunks = [kpop_samples[i:i + chunk_size] for i in range(0, len(kpop_samples), chunk_size)]
        kpop_resampled_chunks = []
        for chunk in kpop_chunks:
            kpop_resampled_chunk = librosa.resample(chunk, orig_sr=kpop_audio.frame_rate, target_sr=16000)
            kpop_resampled_chunks.append(kpop_resampled_chunk)

        # Combine resampled chunks
        kpop_resampled = np.concatenate(kpop_resampled_chunks)

        # Ensure audio length is the same
        min_length = min(len(kpop_resampled), len(resampled_audio))
        if len(kpop_resampled) > min_length:
            print(f"Trimming K-pop audio from {len(kpop_resampled)} to {min_length}")
            kpop_resampled = kpop_resampled[:min_length]
        if len(resampled_audio) > min_length:
            print(f"Trimming Carol audio from {len(resampled_audio)} to {min_length}")
            resampled_audio_trimmed = resampled_audio[:min_length]
        else:
            resampled_audio_trimmed = resampled_audio

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(kpop_resampled, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")  # Add channel dimension
        target_tensor = torch.tensor(resampled_audio_trimmed, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        # Append as (input, target) tuple
        data.append((input_tensor, target_tensor))

    # Save data to .pt file
    torch.save(data, output_file)
    print(f"Data saved to {output_file}")