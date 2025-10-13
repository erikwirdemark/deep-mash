import os
from pathlib import Path
from typing import Generator
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import torchaudio.functional as AF

from deepmash.data_processing.constants import *

INPUT_ROOT = Path("datasets") / Path("gtzan-stems")

def ensure_same_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    min_len = min(len(t) for t in tensors)
    return [t[:min_len] for t in tensors]

def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1: return x.unsqueeze(0)
    if x.ndim != 2: raise ValueError(f"Input tensor has {x.ndim} dimensions, expected 1 or 2.")
    return x

def mix_stems(stems: list[torch.Tensor], peak_val=0.98) -> torch.Tensor:
    stems = ensure_same_length(stems)
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

def load_audio(path: Path|str, sr:int|float, frame_offset=0, num_frames=-1):
    y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    if y.shape[0] > 1: y = y.mean(dim=0, keepdim=True)   # to mono 
    y = AF.resample(y, orig_freq=sr, new_freq=TARGET_SR) # resample
    return y # (1, sr*duration)

def get_chunks(vocals: torch.Tensor, non_vocals: torch.Tensor) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    chunk_frames = CHUNK_DURATION_SEC * TARGET_SR
    min_chunk_frames = MIN_CHUNK_DURATION_SEC * TARGET_SR
    
    vocals, non_vocals = ensure_same_length([vocals, non_vocals])
    
    for i, start in enumerate(range(0, len(vocals), chunk_frames)):
        vocals_chunk = vocals[start:start+chunk_frames]
        non_vocals_chunk = non_vocals[start:start+chunk_frames]

        # if this is the final chunk: discard if way too short, zero-pad if slightly too short
        if len(vocals_chunk) < min_chunk_frames: continue
        vocals_chunk = zero_pad_or_clip(vocals_chunk, chunk_frames)
        non_vocals_chunk = zero_pad_or_clip(non_vocals_chunk, chunk_frames)
        
        yield vocals_chunk, non_vocals_chunk

def get_gtzan_track_folders(root: Path|str):
    return sorted(p for p in Path(root).glob("*/*") if p.is_dir())

class ToLogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_melspec = AT.MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX)
        self.to_db = AT.AmplitudeToDB()
    def forward(self, x: torch.Tensor):
        return self.to_db(self.to_melspec(x))
    
class GTZANStemsDataset(Dataset):
    def __init__(
        self, 
        root_dir: Path|str=INPUT_ROOT,
        preprocess=True,
        preprocess_transform: nn.Module|None=None,
        runtime_transform: nn.Module|None=None,
        device: str="cpu"
    ):
        self.root = Path(root_dir)
        self.processed_root = self.root.parent/(self.root.name+"-processed") if preprocess else self.root
                
        if self.root == self.processed_root and preprocess:
            raise ValueError("`preprocess` is True but root_dir seems to be processed already.")
        if self.root != self.processed_root and not preprocess:
            raise ValueError("`preprocess` is False but root_dir seems to be unprocessed.")
        
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.device = device
        
        if preprocess:
            print(f"Preprocessing GTZAN stems from {str(self.root)} to {str(self.processed_root)} ...")
            self._preprocess()
    
    def _preprocess(self):
        """
        Assuming input files like "`self.root`/blues/blues.000001/{drums|bass|other|vocals}.wav":
        1. load as tensors
        2. convert to mono if in stereo
        3. resample (default 16kHz)
        4. mix all non-vocal stems together and discard originals
        5. chunk into `CHUNK_DURATION_SEC` (default 10s) segments, zero-pad last chunk if needed
        6. (TODO) apply optional `preprocess_transform` (e.g. mel-spectrogram), make sure shapes are correct
        7. save as `self.processed_root`/blues.000001.chunk{1|2|...}/{non-vocals|vocals}.pt
        """
        os.makedirs(self.processed_root, exist_ok=True)
        track_folders = get_gtzan_track_folders(self.root)
        
        for track_folder in tqdm(track_folders):
            
            all_stem_paths = list(track_folder.glob("*.wav"))
            assert {p.stem for p in all_stem_paths} == {"drums", "bass", "other", "vocals"}, f"Not all stems exist for {str(track_folder)}"
            vocals_path = [p for p in all_stem_paths if p.stem == "vocals"][0]
            non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]
            
            vocals = load_audio(vocals_path, sr=TARGET_SR).squeeze(0)
            non_vocals = mix_stems([load_audio(p, sr=TARGET_SR).squeeze(0) for p in non_vocals_paths])
            
            for i, (vocals_chunk, non_vocals_chunk) in enumerate(get_chunks(vocals, non_vocals)):
                chunk_folder = self.processed_root / f"{track_folder.name}.chunk{i+1}"
                os.makedirs(chunk_folder, exist_ok=True)
                torch.save(vocals_chunk, chunk_folder/"vocals.pt")
                torch.save(non_vocals_chunk, chunk_folder/"non-vocals.pt")
            
