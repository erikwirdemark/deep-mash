import librosa
import torch
import matplotlib.pyplot as plt
import random
from IPython.display import Audio, display
from pathlib import Path

from deepmash.data_processing.constants import *
from deepmash.data_processing.common import StemsDataset, ToLogMel

def zero_pad_or_clip(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if len(x) >= target_len:
        return x[:target_len]
    pad_len = target_len - len(x)
    return torch.cat([x, torch.zeros(pad_len)], dim=0)

def ensure_same_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    min_len = min(len(t) for t in tensors)
    return [t[:min_len] for t in tensors]

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

# --- plotting utils

def display_melspec(vocals: torch.Tensor, non_vocals: torch.Tensor, track_name: str):
    fig, axs = plt.subplots(2,1, figsize=(16, 6))
    fig.suptitle(track_name)
    axs[0].set_title("vocals"); axs[1].set_title("non-vocals")
    im0 = librosa.display.specshow(vocals.numpy(), sr=TARGET_SR, hop_length=HOP_SIZE, n_fft=WINDOW_SIZE, fmin=F_MIN, fmax=F_MAX,
                             x_axis="time", y_axis="mel", ax=axs[0])
    im1 = librosa.display.specshow(non_vocals.numpy(), sr=TARGET_SR, hop_length=HOP_SIZE, n_fft=WINDOW_SIZE, fmin=F_MIN, fmax=F_MAX,
                             x_axis="time", y_axis="mel", ax=axs[1])
    fig.tight_layout()
    fig.colorbar(im0, ax=axs, format="%+.0f db")
    plt.show()

def display_processed_chunk(p: Path):
    track_name = p.name
    vocals = torch.load(p/"vocals.pt").squeeze()
    non_vocals = torch.load(p/"non_vocals.pt").squeeze()

    if vocals.ndim == 1 and non_vocals.ndim == 1:
        vocals_np, non_vocals_np = vocals.numpy(), non_vocals.numpy()
        fig, axs = plt.subplots(2,1, figsize=(14,6))
        fig.suptitle(track_name)
        axs[0].set_title("vocals"); axs[1].set_title("non-vocals")
        librosa.display.waveshow(vocals_np, ax=axs[0])
        librosa.display.waveshow(non_vocals_np, ax=axs[1])
        fig.tight_layout()

        display(Audio(vocals_np, rate=TARGET_SR))
        display(Audio(non_vocals_np, rate=TARGET_SR))
        
        vocals_mel = ToLogMel()(vocals)
        non_vocals_mel = ToLogMel()(non_vocals)
        display_melspec(vocals_mel, non_vocals_mel, track_name)
        
        print(f"vocals: shape={vocals_mel.shape}, mean={vocals_mel.mean()}")
        print(f"non-vocals: shape={non_vocals_mel.shape}, mean={non_vocals_mel.mean()}")
    
    elif vocals.ndim == 2 and non_vocals.ndim == 2:
        display_melspec(vocals, non_vocals, track_name)
    
    else:
        print(f"Unexpected tensor shapes: vocals {vocals.shape}, non-vocals {non_vocals.shape}")

def display_random_chunk(dataset: StemsDataset):
    all_paths = list(p.parent for p in dataset.processed_root.glob(r"*/*.pt"))
    p = random.choice(all_paths)
    display_processed_chunk(p)
