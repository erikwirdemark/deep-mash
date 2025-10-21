import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as AT
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from .utils import get_gtzan_track_folders, load_audio, mix_stems, get_chunks
from pathlib import Path

# Load configuration
config = OmegaConf.load('/Users/erikwirdemark/deep-mash/config/default.yaml')

# ---------- Mel transform ----------

class ToLogMel(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.to_melspec = AT.MelSpectrogram(
            sample_rate=config.audio.target_sample_rate,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_size,
            f_min=config.audio.f_min,
            f_max=config.audio.f_max,
        )
        self.to_db = AT.AmplitudeToDB()

    def forward(self, x: torch.Tensor):
        return self.to_db(self.to_melspec(x))


# ---------- Main dataset with preprocessing ----------

class GTZANStemsDataset(Dataset):
    def __init__(
        self,
        config: OmegaConf,
        preprocess=True,
        preprocess_transform: nn.Module | None = None,
        runtime_transform: nn.Module | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.root = Path(config.data.input_root)
        self.originals_root = Path(config.data.originals_root)
        self.processed_root = Path(config.data.processed_root)

        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        if device == "cuda":
            print('Using CUDA device for dataset tensors.')
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            device = "cpu"
        self.device = device

        if preprocess:
            print(f"Preprocessing GTZAN stems from {self.root} and originals from {self.originals_root}")
            self._preprocess()

        # After preprocessing, load the chunk list
        self.chunk_dirs = sorted([p for p in self.processed_root.glob("*") if p.is_dir()])

    def _preprocess(self):
        """
        Assuming input files like "`self.root`/blues/blues.000001/{drums|bass|other|vocals}.wav":
        1. load as tensors
        2. convert to mono if in stereo
        3. resample (default 16kHz)
        4. mix all non-vocal stems together and discard originals
        5. chunk into `CHUNK_DURATION_SEC` (default 10s) segments, zero-pad last chunk if needed
        6. apply optional `preprocess_transform` (e.g. mel-spectrogram), make sure shapes are correct
        7. save as `self.processed_root`/blues.000001.chunk{1|2|...}/{non-vocals|vocals}.pt
        """
        os.makedirs(self.processed_root, exist_ok=True)
        track_folders = get_gtzan_track_folders(self.root)

        for track_folder in tqdm(track_folders):
            all_stem_paths = list(track_folder.glob("*.wav"))
            assert {p.stem for p in all_stem_paths} == {"drums", "bass", "other", "vocals"}, \
                f"Missing stems for {track_folder}"

            vocals_path = [p for p in all_stem_paths if p.stem == "vocals"][0]
            non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]

            track_name = track_folder.name
            genre = track_folder.parent.name  # e. g. "blues"
            orig_path = self.originals_root / f"{genre}" / f"{track_name}.wav"
            
            # Load and mix stems
            try:
                vocals = load_audio(path=vocals_path, sr=config.audio.target_sample_rate)
                non_vocals = mix_stems([load_audio(path=p, sr=config.audio.target_sample_rate) for p in non_vocals_paths])
                original = load_audio(path=orig_path, sr=config.audio.target_sample_rate)
            except Exception as e:
                print(f"Skipping track {track_name} due to loading error: {e}")
                continue
            
            vocals = vocals.squeeze(0)
            non_vocals = non_vocals.squeeze(0)
            original = original.squeeze(0)

            # Generate aligned chunks
            for i, ((vocals_chunk, non_vocals_chunk), (orig_chunk, _)) in enumerate(
                zip(get_chunks(config=config, vocals=vocals, non_vocals=non_vocals), get_chunks(config=config, vocals=original, non_vocals=original))
            ):
                if self.preprocess_transform is not None:
                    with torch.no_grad():
                        vocals_chunk = self.preprocess_transform(vocals_chunk.unsqueeze(0))  # (1, T)
                        non_vocals_chunk = self.preprocess_transform(non_vocals_chunk.unsqueeze(0))
                        orig_chunk = self.preprocess_transform(orig_chunk.unsqueeze(0))
                        
                chunk_folder = self.processed_root / f"{track_name}.chunk{i+1}"
                os.makedirs(chunk_folder, exist_ok=True)
                torch.save(vocals_chunk, chunk_folder / "vocals.pt")
                torch.save(non_vocals_chunk, chunk_folder / "non-vocals.pt")
                torch.save(orig_chunk, chunk_folder / "original.pt")

    def __len__(self):
        return len(self.chunk_dirs)

    def __getitem__(self, idx):
        chunk_dir = self.chunk_dirs[idx]
        vocals = torch.load(chunk_dir / "vocals.pt", map_location='cpu', weights_only=False)
        non_vocals = torch.load(chunk_dir / "non-vocals.pt", map_location='cpu', weights_only=False)
        original = torch.load(chunk_dir / "original.pt", map_location='cpu', weights_only=False)

        if self.runtime_transform:
            vocals = self.runtime_transform(vocals)
            non_vocals = self.runtime_transform(non_vocals)
            original = self.runtime_transform(original)

        return {
            "vocals": vocals.to(self.device),
            "non_vocals": non_vocals.to(self.device),
            "original": original.to(self.device),
            "chunk_name": chunk_dir.name,
        }
    
        