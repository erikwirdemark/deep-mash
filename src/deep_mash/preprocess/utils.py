
import torch
import torchaudio
from pathlib import Path
from typing import Generator
import torchaudio.functional as AF

# ---------- Utility functions ----------
def has_enough_vocal_energy(vocals_chunk, threshold):
    return vocals_chunk.abs().mean().item() > threshold

def ensure_same_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    min_len = min(len(t) for t in tensors)
    return [t[:min_len] for t in tensors]

def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Input tensor has {x.ndim} dimensions, expected 1 or 2.")
    return x

def mix_stems(stems: list[torch.Tensor], peak_val=0.98) -> torch.Tensor:
    stems = ensure_same_length(stems)
    mixed: torch.Tensor = sum(stems)  # type: ignore
    max_val = mixed.abs().max()
    if max_val > 0:
        mixed = mixed / max_val * peak_val
    return mixed

def zero_pad_or_clip(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if len(x) >= target_len:
        return x[:target_len]
    pad_len = target_len - len(x)
    return torch.cat([x, torch.zeros(pad_len)], dim=0)

def load_audio(path: Path | str, sr: int | float, frame_offset=0, num_frames=-1):
    y, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)  # convert to mono
    y = AF.resample(y, orig_freq=sr, new_freq=sr)
    return y  # (1, sr*duration)

def get_chunks(config, vocals: torch.Tensor, non_vocals: torch.Tensor) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    chunk_frames = config.audio.chunk_duration_sec * config.audio.target_sample_rate
    min_chunk_frames = config.audio.min_chunk_duration_sec * config.audio.target_sample_rate

    vocals, non_vocals = ensure_same_length([vocals, non_vocals])

    for i, start in enumerate(range(0, len(vocals), chunk_frames)):
        vocals_chunk = vocals[start:start + chunk_frames]
        non_vocals_chunk = non_vocals[start:start + chunk_frames]

        if len(vocals_chunk) < min_chunk_frames:
            continue
        vocals_chunk = zero_pad_or_clip(vocals_chunk, chunk_frames)
        non_vocals_chunk = zero_pad_or_clip(non_vocals_chunk, chunk_frames)

        # Only yield if vocals have enough energy
        if not has_enough_vocal_energy(vocals_chunk, threshold=config.audio.vocal_energy_threshold):
            continue

        yield vocals_chunk, non_vocals_chunk

def get_gtzan_track_folders(root: Path | str):
    return sorted(p for p in Path(root).glob("*/*") if p.is_dir())
