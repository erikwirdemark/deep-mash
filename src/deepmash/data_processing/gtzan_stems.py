import os
from pathlib import Path
from tqdm.notebook import tqdm
import torch
import torch.nn as nn

from deepmash.data_processing.constants import *
from deepmash.data_processing.common import (
    has_enough_vocal_energy,
    load_audio,
    mix_stems,
    get_chunks,
    StemsDataset
)

INPUT_ROOT = Path("datasets") / Path("gtzan-stems")

EXCLUDED_GENRES = ["classical"]

def get_gtzan_track_folders(root: Path|str):
    return sorted(p for p in Path(root).glob("*/*") if p.is_dir())
    
class GTZANStemsDataset(StemsDataset):
    def __init__(
        self, 
        root_dir: Path|str=INPUT_ROOT,
        already_preprocessed: bool=True,
        preprocess_transform: nn.Module|None=None,
        runtime_transform: nn.Module|None=None,
    ):
        self.root = Path(root_dir)
        self.processed_root = self.root.parent/(self.root.name+"-processed")
        if already_preprocessed and not self.processed_root.exists():
            raise ValueError(f"already_preprocessed is True but {self.processed_root} does not exist")
        
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform

        if not already_preprocessed:
            print(f"Preprocessing stems from {str(self.root)} to {str(self.processed_root)} ...")
            self._preprocess()
        
        self.chunk_folders = list(sorted(p for p in self.processed_root.glob("*") if p.is_dir()))
    
    def _preprocess(self):
        """
        Assuming input files like "`self.root`/blues/blues.000001/{drums|bass|other|vocals}.wav":
        1. load as tensors
        2. convert to mono if in stereo
        3. resample (default 16kHz)
        4. mix all non-vocal stems together and discard originals
        5. chunk into `CHUNK_DURATION_SEC` (default 10s) segments, zero-pad last chunk if needed
        6. apply optional `preprocess_transform` (e.g. mel-spectrogram), make sure shapes are correct
        7. save as `self.processed_root`/blues.000001.chunk{1|2|...}/{non_vocals|vocals}.pt
        """
        os.makedirs(self.processed_root, exist_ok=True)
        track_folders = get_gtzan_track_folders(self.root)

        for track_folder in tqdm(track_folders):
            genre = track_folder.parent.name
            if genre in EXCLUDED_GENRES:
                continue

            all_stem_paths = list(track_folder.glob("*.wav"))
            assert {p.stem for p in all_stem_paths} == {"drums", "bass", "other", "vocals"}, f"Not all stems exist for {str(track_folder)}"
            vocals_path = [p for p in all_stem_paths if p.stem == "vocals"][0]
            non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]
            
            try:
                vocals = load_audio(vocals_path, sr=TARGET_SR).squeeze(0)
                non_vocals = mix_stems([load_audio(p, sr=TARGET_SR).squeeze(0) for p in non_vocals_paths])
            except Exception as e:
                print(f"Error loading {str(track_folder)}: {e}")
                continue
            
            for i, (vocals_chunk, non_vocals_chunk) in enumerate(get_chunks(vocals, non_vocals)):
                if not has_enough_vocal_energy(vocals_chunk):
                    continue
                
                if self.preprocess_transform is not None:
                    vocals_chunk = self.preprocess_transform(vocals_chunk)
                    non_vocals_chunk = self.preprocess_transform(non_vocals_chunk)
                    
                chunk_folder = self.processed_root / f"{track_folder.name}.chunk{i+1}"
                os.makedirs(chunk_folder, exist_ok=True)
                torch.save(vocals_chunk, chunk_folder/"vocals.pt")
                torch.save(non_vocals_chunk, chunk_folder/"non_vocals.pt")