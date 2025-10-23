import librosa
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch 
from IPython.display import Audio, display
import torch.nn.functional as F

from deepmash.data_processing.common import load_audio, mix_stems
from deepmash.utils.plotting_utils import display_waveforms
import soundfile as sf


def mashup_baseline(vocals: torch.Tensor, non_vocals: torch.Tensor) -> np.ndarray:

    mixed = mix_stems([vocals, non_vocals])
    return mixed.cpu().numpy()

def mashup(config: DictConfig, vocals_path: Path, track_folder: Path, plot=True):
    track_name = track_folder

    all_stem_paths = list(track_folder.glob("*.wav"))
    assert {p.stem for p in all_stem_paths} == {"drums", "bass", "other", "vocals"}, f"Not all stems exist for {str(track_folder)}"
    non_vocals_paths = [p for p in all_stem_paths if p.stem != "vocals"]

    vocals = load_audio(path=vocals_path, target_sr=config.audio.target_sample_rate).squeeze(0)
    non_vocals = mix_stems([load_audio(path=p, target_sr=config.audio.target_sample_rate).squeeze(0) for p in non_vocals_paths])

    try:
        sr = config.audio.target_sample_rate
        # ensure numpy 1D arrays for librosa
        v_np = vocals.cpu().numpy() if isinstance(vocals, torch.Tensor) else np.array(vocals)
        nv_np = non_vocals.cpu().numpy() if isinstance(non_vocals, torch.Tensor) else np.array(non_vocals)
        v_np = v_np.flatten()
        nv_np = nv_np.flatten()

        # estimate tempo (BPM)
        t_v = float(librosa.feature.rhythm.tempo(y=v_np, sr=sr, aggregate=None).mean()) if v_np.size > 0 else 0.0
        t_nv = float(librosa.feature.rhythm.tempo(y=nv_np, sr=sr, aggregate=None).mean()) if nv_np.size > 0 else 0.0

        if t_v > 0 and t_nv > 0:
            rate = t_v / t_nv
            if not np.isclose(rate, 1.0, atol=0.01):
                stretched = librosa.effects.time_stretch(nv_np, rate=rate)
                non_vocals = torch.from_numpy(stretched).to(non_vocals.device if isinstance(non_vocals, torch.Tensor) else 'cpu')
    except Exception:
        # keep original non_vocals if something goes wrong
        pass
    desired_len = non_vocals.shape[-1]
    v_len = vocals.shape[-1]
    if v_len > desired_len:
        vocals = vocals[..., :desired_len]
    elif v_len < desired_len:
        pad = desired_len - v_len
        vocals = F.pad(vocals, (0, pad))
    mashup_audio = mashup_baseline(vocals, non_vocals)

    # save to wav file
    vocals_name = (str(vocals_path)).split('/')[-2]
    instr_name = (str(track_name)).split('/')[-1]
    output_path = Path(config.data.output) / f"{vocals_name}_{instr_name}_mashup.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mashup_audio, config.audio.target_sample_rate)
    
    if plot:
        display_waveforms(vocals, non_vocals, track_name=track_name)
        display(Audio(mashup_audio, rate=config.audio.target_sample_rate))
    print(f"Saved mashup to {str(output_path)}")
    return mashup_audio
