from __future__ import annotations
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
import abc
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader, Subset

# ignore annoying "pkg_resources is deprecated as an API" coming from this import
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    import stempeg

from deepmash.utils.utils import zero_pad_or_clip, ensure_same_length

def get_vocal_rms(vocals: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(vocals**2)).item()

# TODO: maybe filter on fraction of above-threshold frames instead of global RMS
def has_enough_vocal_energy(vocals: torch.Tensor, threshold) -> bool:
    return get_vocal_rms(vocals) >= threshold

def mix_stems(stems: list[torch.Tensor], peak_val=0.98) -> torch.Tensor:
    stems = ensure_same_length(stems)
    mixed: torch.Tensor = sum(stems) # type: ignore
    max_val = mixed.abs().max()
    if max_val > 0:  # normalize to max peak_val to avoid clipping
        mixed = mixed / max_val * peak_val
    return mixed
    
def load_audio(target_sr, path: Path|str, to_mono=True) -> torch.Tensor:
    y, sr = torchaudio.load(path)
    if to_mono and y.shape[0] > 1: 
        y = y.mean(dim=0, keepdim=True)
    y = AF.resample(y, orig_freq=sr, new_freq=target_sr)
    return y

# For loading the multichannel stem-format files used in MUSDB18
def load_stem_audio(path: Path|str, target_sr:int|float, to_mono=True) -> torch.Tensor:
    stems, sr = stempeg.read_stems(str(path), sample_rate=target_sr)
    stems_tensor = torch.from_numpy(stems).to(torch.float32)
    if to_mono and stems_tensor.ndim == 3: # (stem, n_samples, channel)
        stems_tensor = stems_tensor.mean(dim=2) 
    return stems_tensor

def get_chunks(config, vocals: torch.Tensor, non_vocals: torch.Tensor) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    chunk_frames = config.audio.chunk_duration_sec * config.audio.target_sample_rate
    min_chunk_frames = config.audio.min_chunk_duration_sec * config.audio.target_sample_rate

    vocals, non_vocals = ensure_same_length([vocals, non_vocals])

    for i, start in enumerate(range(0, len(vocals), chunk_frames)):
        vocals_chunk = vocals[start:start+chunk_frames]
        non_vocals_chunk = non_vocals[start:start+chunk_frames]

        # if this is the final chunk: discard if way too short, zero-pad if slightly too short
        if len(vocals_chunk) < min_chunk_frames: continue
        vocals_chunk = zero_pad_or_clip(vocals_chunk, chunk_frames)
        non_vocals_chunk = zero_pad_or_clip(non_vocals_chunk, chunk_frames)
        
        yield vocals_chunk, non_vocals_chunk

class ToLogMel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to_melspec = AT.MelSpectrogram(
            sample_rate=config.audio.target_sample_rate,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_size,
            f_min=config.audio.f_min,
            f_max=config.audio.f_max
        )
        self.to_db = AT.AmplitudeToDB()
    def forward(self, x: torch.Tensor):
        return self.to_db(self.to_melspec(x))
    
@dataclass 
class StemsSample:
    vocals: torch.Tensor
    non_vocals: torch.Tensor
    
    def to(self, device: torch.device) -> StemsSample:
        return StemsSample(
            vocals=self.vocals.to(device),
            non_vocals=self.non_vocals.to(device)
        )

class StemsDataset(Dataset, abc.ABC):
    processed_root: Path
    chunk_folders: list[Path]
    preprocess_transform: nn.Module|None
    runtime_transform: nn.Module|None
    
    def __len__(self) -> int:
        return len(self.chunk_folders)

    def __getitem__(self, idx) -> StemsSample:
        folder = self.chunk_folders[idx]
        vocals = torch.load(folder/"vocals.pt").squeeze()
        non_vocals = torch.load(folder/"non_vocals.pt").squeeze()
        
        if self.runtime_transform is not None:
            vocals = self.runtime_transform(vocals)
            non_vocals = self.runtime_transform(non_vocals)

        return StemsSample(vocals=vocals, non_vocals=non_vocals)

def collate_stems_batch(batch: list[StemsSample]) -> StemsSample:
    vocals = torch.stack([sample.vocals for sample in batch], dim=0)
    non_vocals = torch.stack([sample.non_vocals for sample in batch], dim=0)
    return StemsSample(vocals=vocals, non_vocals=non_vocals)

# Split by tracks to ensure chunks from same track are in same split
def get_dataloaders(dataset: StemsDataset, config: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = config.batch_size
    random_seed = config.seed
    num_workers = config.num_workers
    val_split = config.val_split
    test_split = config.test_split

    track_names = sorted(set(p.name.split(".chunk")[0] for p in dataset.chunk_folders))
    train_and_val_track_names, test_track_names = train_test_split(track_names, test_size=test_split, random_state=random_seed)
    train_track_names, val_track_names = train_test_split(train_and_val_track_names, test_size=val_split/(1 - test_split), random_state=random_seed)
    
    train_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in train_track_names]
    val_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in val_track_names]
    test_indices = [i for i, p in enumerate(dataset.chunk_folders) if p.name.split(".chunk")[0] in test_track_names]
    
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_stems_batch, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_stems_batch, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_stems_batch, num_workers=num_workers)

    return train_loader, val_loader, test_loader