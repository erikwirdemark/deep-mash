from pathlib import Path
import pickle
from typing import List
import torch
from deep_mash.preprocess.gtzanstems_dataset import ToLogMel
from deep_mash.preprocess.utils import load_audio
from deep_mash.model.base import DualEncoderModel
import torch.nn as nn
import torch.nn.functional as F


def compute_embedding_for_audio(model: DualEncoderModel, target_sr: int, audio_path: str | Path, which: str = "vocal", preprocess_transform: nn.Module = ToLogMel(), device: str | None = None) -> torch.Tensor:
    """Compute a single embedding for an audio file using the model's vocal or instr encoder.

    Returns a CPU tensor of shape (embedding_dim,).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load waveform (returns tensor shape (1, samples))
    audio = load_audio(audio_path, sr=target_sr)  # (1, N)
    with torch.no_grad():
        mel = preprocess_transform(audio)  # expected shape: (1, n_mels, time) or (n_mels, time)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # (1,1,n_mels,time) expected by encoder
        elif mel.dim() == 2:
            mel = mel.unsqueeze(0).unsqueeze(1)
        mel = mel.float().to(device)
        if which.startswith("vocal") or which == "v":
            emb = model.vocal_encoder(mel)  # (1, D)
        else:
            emb = model.instr_encoder(mel)  # (1, D)
        emb = emb.squeeze(0).cpu()  # (D,)
    return emb

def query_saved_embeddings(model, query_audio: str | Path, config: dict, top_k: int = 5, preprocess_transform: nn.Module = ToLogMel(), device: str | None = None) -> List[tuple[str, float]]:
    """Load saved embeddings and return top_k matches for the query audio file.

    Returns list of (name, score) sorted descending by cosine similarity.
    which_query: which encoder to use for the query audio ('vocal' or 'instr')
    which_catalogue: which saved catalogue to search ('instr' or 'vocal')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load catalogue embeddings and names
    emb_file = f"{config.query.embeddings_prefix}_{config.query.which_catalogue}_embs.pt"
    names_file = f"{config.query.embeddings_prefix}_names.pkl"
    if not Path(emb_file).exists() or not Path(names_file).exists():
        raise FileNotFoundError(f"Embedding files not found: {emb_file} or {names_file}")
    catalogue_embs = torch.load(emb_file, map_location="cpu")  # [N, D]
    with open(names_file, "rb") as f:
        names = pickle.load(f)
    if catalogue_embs.ndim != 2:
        raise ValueError("Catalogue embeddings should be a 2D tensor [N,D]")

    # Compute query embedding
    q_emb = compute_embedding_for_audio(model, query_audio, which=config.query.which_query, preprocess_transform=preprocess_transform, device=device, target_sr=config.audio.target_sample_rate)  # (D,) CPU
    # Ensure both normalized
    catalogue = F.normalize(catalogue_embs, dim=1)  # [N,D]
    query = F.normalize(q_emb.unsqueeze(0), dim=1)  # [1,D]
    sims = (catalogue @ query.T).squeeze(1).numpy()  # [N]
    top_idx = sims.argsort()[::-1][:top_k]
    results = [(names[i], float(sims[i])) for i in top_idx]
    print("Top matches:")
    for name, score in results:
        print(f"{name}: {score:.4f}")
    return results
