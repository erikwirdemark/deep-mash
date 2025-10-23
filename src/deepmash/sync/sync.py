import librosa
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch 
from IPython.display import Audio, display

from deepmash.data_processing.common import load_audio, mix_stems
from deepmash.utils.plotting_utils import display_waveforms

config: DictConfig = OmegaConf.load("../../../config/default.yaml") # type: ignore

def mashup_baseline(vocals: torch.Tensor, non_vocals: torch.Tensor) -> np.ndarray:
    mixed = mix_stems([vocals, non_vocals])
    return mixed.cpu().numpy()

def mashup(config: DictConfig, track_folder: Path, plot=True):
    track_name = track_folder

    all_stem_paths = list(track_folder.glob("*.wav"))
    assert {p.stem for p in all_stem_paths} == {"drums", "bass", "other", "vocals"}, f"Not all stems exist for {str(track_folder)}"
    vocals_path = [p for p in all_stem_paths if p.stem == "vocals"][0]
    non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]

    vocals = load_audio(path=vocals_path, target_sr=config.audio.target_sample_rate).squeeze(0)
    non_vocals = mix_stems([load_audio(path=p, target_sr=config.audio.target_sample_rate).squeeze(0) for p in non_vocals_paths])

    mashup_audio = mashup_baseline(vocals, non_vocals)
    
    if plot:
        display_waveforms(vocals, non_vocals, track_name=track_name)
        display(Audio(mashup_audio, rate=config.audio.target_sample_rate))
    
    return mashup_audio

if __name__ == "__main__":
    # all folders with .wav files
    all_track_folders = [p.parent for p in Path("../data/gtzan-stems").glob("**/*.wav")]
    track_folder = all_track_folders[0]
    mashup_audio = mashup(config=config, track_folder=track_folder, plot=True)