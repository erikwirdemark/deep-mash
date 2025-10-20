import os
from pathlib import Path
from typing import Literal
from tqdm.notebook import tqdm
import torch
import torch.nn as nn

from deepmash.data_processing.constants import *
from deepmash.data_processing.common import (
    has_enough_vocal_energy,
    load_stem_audio,
    mix_stems,
    get_chunks,
    StemsDataset
)

INPUT_ROOT = Path("datasets") / Path("musdb18")

STEM_INDS = {
    "mixture": 0,
    "drums":   1,
    "bass":    2,
    "other":   3,
    "vocals":  4,
}

class MUSDB18Dataset(StemsDataset):
    def __init__(
        self, 
        root_dir: Path|str=INPUT_ROOT,
        split: Literal["train", "test"] = "train",
        already_preprocessed: bool=True,
        preprocess_transform: nn.Module|None=None,
        runtime_transform: nn.Module|None=None,
    ):
        self.split = split
        self.root = Path(root_dir) / Path(split)
        self.processed_root = Path(root_dir).parent/(Path(root_dir).name+"-processed")/Path(split)
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
        Assuming input files like `self.root`/title.stem.mp4 (5-channel stem-format):
        1. load as (5, n_samples) tensors, resample to 16kHz
        2. mix channels 1,2,3 -> non_vocals, select channel 4 -> vocals, discard channel 0
        3. chunk into CHUNK_DURATION_SEC chunks, apply transform if provided
        4. save as `self.processed_root`/title.chunk{1|2|...}/{vocals|non_vocals.pt}
        """
        os.makedirs(self.processed_root, exist_ok=True)
        track_files = list(self.root.glob("*.mp4"))
        
        for track_file in tqdm(track_files):
            stems = load_stem_audio(track_file, target_sr=TARGET_SR)
            assert stems.shape[0] == 5, f"Expected 5 stems in {str(track_file)}, got {stems.shape[0]}"
            
            vocals = stems[STEM_INDS["vocals"]]
            non_vocals = mix_stems([stems[STEM_INDS["drums"]], stems[STEM_INDS["bass"]], stems[STEM_INDS["other"]]])
            
            for i, (vocals_chunk, non_vocals_chunk) in enumerate(get_chunks(vocals, non_vocals)):
                if not has_enough_vocal_energy(vocals_chunk):
                    continue

                if self.preprocess_transform is not None:
                    vocals_chunk = self.preprocess_transform(vocals_chunk)
                    non_vocals_chunk = self.preprocess_transform(non_vocals_chunk)
                
                chunk_folder = self.processed_root / Path(f"{track_file.stem}.chunk{i+1}")
                os.makedirs(chunk_folder, exist_ok=True)
                
                torch.save(vocals_chunk, chunk_folder / Path("vocals.pt"))
                torch.save(non_vocals_chunk, chunk_folder / Path("non_vocals.pt"))