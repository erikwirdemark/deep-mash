import os
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
import librosa 
import soundfile

INPUT_ROOT = "datasets/gtzan-stems"
OUTPUT_ROOT = "datasets/gtzan-stems-processed"

TARGET_SR = 16000
CHUNK_DURATION_SEC = 10

def mix_stems(stems: list[np.ndarray], peak_val=0.98) -> np.ndarray:
    min_len = min(len(stem) for stem in stems)
    stems = [stem[:min_len] for stem in stems]
    mixed = sum(stems)
    max_val = np.abs(mixed).max()
    if max_val > 0: # normalize to max peak_val to avoid clipping
        mixed = mixed / max_val * peak_val
    return mixed # type: ignore

def load_audio(path: Path|str, sr=TARGET_SR):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y

def write_audio(path: Path|str, y: np.ndarray, sr=TARGET_SR) -> None:
    soundfile.write(path, y, sr)

def get_all_track_folders(root=INPUT_ROOT):
    return sorted(p for p in Path(root).glob("*/*") if p.is_dir())

def preprocess_gtzan_stems(input_path=INPUT_ROOT, output_path=OUTPUT_ROOT, sr=TARGET_SR):
    """
    Assuming input files like "`input_path`/blues/blues.000001/{drums|bass|other|vocals}.wav",
    create outputs like "`output_path`/blues.000001/{non-vocals|vocals}.wav", by:
      1. converting inputs to mono if in stereo
      2. resampling to `sr` (default 16kHz)
      3. mixing all non-vocal stems together
    """
    os.makedirs(output_path, exist_ok=True)
    track_folders = get_all_track_folders(input_path)
    for track_folder in tqdm(track_folders):
        track_id = track_folder.name                       # ie country.002
        out_folder = Path(output_path).joinpath(track_id)  # ie gtzan-stems-processed/country.002
        
        all_stem_paths = list(track_folder.glob("*.wav"))
        if {p.stem for p in all_stem_paths} != {"drums", "bass", "other", "vocals"}:
            raise ValueError(f"Not all stems exist for {str(track_folder)}")
        
        vocals_path = [p for p in all_stem_paths if p.stem == "vocals"][0]
        non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]

        vocals = load_audio(vocals_path)
        non_vocals = mix_stems([load_audio(p) for p in non_vocals_paths])

        os.makedirs(out_folder, exist_ok=True)
        write_audio(out_folder.joinpath("vocals.wav"), vocals)
        write_audio(out_folder.joinpath("non-vocals.wav"), non_vocals)