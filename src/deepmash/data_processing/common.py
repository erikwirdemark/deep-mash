from pathlib import Path
from typing import Generator
from dataclasses import dataclass
import abc
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader, Subset
import stempeg

from deepmash.data_processing.constants import *

def _ensure_same_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    min_len = min(len(t) for t in tensors)
    return [t[:min_len] for t in tensors]

def mix_stems(stems: list[torch.Tensor], peak_val=0.98) -> torch.Tensor:
    stems = _ensure_same_length(stems)
    mixed: torch.Tensor = sum(stems) # type: ignore
    max_val = mixed.abs().max()
    if max_val > 0:  # normalize to max peak_val to avoid clipping
        mixed = mixed / max_val * peak_val
    return mixed
    
def zero_pad_or_clip(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if len(x) >= target_len:
        return x[:target_len]
    pad_len = target_len - len(x)
    return torch.cat([x, torch.zeros(pad_len)], dim=0)

def load_audio(path: Path|str, sr:int|float, to_mono=True) -> torch.Tensor:
    y, sr = torchaudio.load(path)
    if to_mono and y.shape[0] > 1: 
        y = y.mean(dim=0, keepdim=True)
    y = AF.resample(y, orig_freq=sr, new_freq=TARGET_SR)
    return y

def load_stem_audio(path: Path|str, target_sr:int|float, to_mono=True) -> torch.Tensor:
    stems, sr = stempeg.read_stems(str(path), sample_rate=target_sr)
    stems_tensor = torch.from_numpy(stems).to(torch.float32)
    if to_mono and stems_tensor.ndim == 3: # (stem, n_samples, channel)
        stems_tensor = stems_tensor.mean(dim=2) 
    return stems_tensor

def get_chunks(vocals: torch.Tensor, non_vocals: torch.Tensor) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    chunk_frames = CHUNK_DURATION_SEC * TARGET_SR
    min_chunk_frames = MIN_CHUNK_DURATION_SEC * TARGET_SR
    
    vocals, non_vocals = _ensure_same_length([vocals, non_vocals])
    
    for i, start in enumerate(range(0, len(vocals), chunk_frames)):
        vocals_chunk = vocals[start:start+chunk_frames]
        non_vocals_chunk = non_vocals[start:start+chunk_frames]

        # if this is the final chunk: discard if way too short, zero-pad if slightly too short
        if len(vocals_chunk) < min_chunk_frames: continue
        vocals_chunk = zero_pad_or_clip(vocals_chunk, chunk_frames)
        non_vocals_chunk = zero_pad_or_clip(non_vocals_chunk, chunk_frames)
        
        yield vocals_chunk, non_vocals_chunk

class ToLogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_melspec = AT.MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX)
        self.to_db = AT.AmplitudeToDB()
    def forward(self, x: torch.Tensor):
        return self.to_db(self.to_melspec(x))
    
@dataclass 
class StemsSample:
    vocals: torch.Tensor
    non_vocals: torch.Tensor

class StemsDataset(Dataset):
    processed_root: Path
    chunk_folders: list[Path]
    preprocess_transform: nn.Module|None
    runtime_transform: nn.Module|None
    
    def __len__(self) -> int:
        return len(self.chunk_folders)

    def __getitem__(self, idx) -> StemsSample:
        folder = self.chunk_folders[idx]
        vocals = torch.load(folder/"vocals.pt").squeeze()
        non_vocals = torch.load(folder/"non-vocals.pt").squeeze()
        
        if self.runtime_transform is not None:
            vocals = self.runtime_transform(vocals)
            non_vocals = self.runtime_transform(non_vocals)

        return StemsSample(vocals=vocals, non_vocals=non_vocals)

def collate_stems_batch(batch: list[StemsSample]) -> StemsSample:
    vocals = torch.stack([sample.vocals for sample in batch], dim=0)
    non_vocals = torch.stack([sample.non_vocals for sample in batch], dim=0)
    return StemsSample(vocals=vocals, non_vocals=non_vocals)

# split by tracks so chunks from same track are in same split
def get_dataloaders(dataset: StemsDataset, batch_size: int, val_split: float=0.1, test_split: float=0.1, random_seed: int=42) -> tuple[DataLoader, DataLoader, DataLoader]:
    track_names = sorted(set(p.name.split(".chunk")[0] for p in dataset.chunk_folders))
    train_and_val_track_names, test_track_names = train_test_split(track_names, test_size=test_split, random_state=random_seed)
    train_track_names, val_track_names = train_test_split(train_and_val_track_names, test_size=val_split/(1 - test_split), random_state=random_seed)
    
    train_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in train_track_names]
    val_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in val_track_names]
    test_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in test_track_names]
    
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_stems_batch)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_stems_batch)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_stems_batch)
    
    return train_loader, val_loader, test_loader