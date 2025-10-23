from pathlib import Path
import pickle
from typing import List
import torch
from deepmash.data_processing.common import ToLogMel, load_audio
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as T


def compute_embedding_for_audio(model, target_sr: int, audio_path: str | Path, preprocess_transform: nn.Module, device: str | None = None) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load waveform (returns tensor shape (1, samples))
    audio = load_audio(path=audio_path, target_sr=target_sr)  # (1, N)
    with torch.no_grad():
        mel = preprocess_transform(audio)
        query_emb = model.compute_vocal_embedding(mel.to(device))  # (D,)
    return query_emb

def query_saved_embeddings(
        model, 
        query_audio: str | Path, 
        catalogue: str | Path,
        preprocess_transform: nn.Module, 
        target_sr: int = 16000,
        top_k: int = 5, 
        device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Compute query embedding
    query_emb = compute_embedding_for_audio(
        model=model,
        target_sr=target_sr,
        audio_path=query_audio,
        preprocess_transform=preprocess_transform,
        device=device
    )

    # Load saved embeddings and names
    non_vocal_embs = torch.load(catalogue + "_non_vocal_embs.pt")
    with open(catalogue + "_names.pkl", 'rb') as f:
        names = pickle.load(f)
    if isinstance(non_vocal_embs, list):
        non_vocal_embs = torch.stack(non_vocal_embs, dim=0)  # (N, D)
    non_vocal_embs = non_vocal_embs.to(device)  # (N, D)
    query_emb = query_emb.to(device)

    q = F.normalize(query_emb, dim=1)  # (1, D)
    catalogue = F.normalize(non_vocal_embs, dim=1)   # (N, D)

    sims = (q @ catalogue.T).squeeze(0)

    # Get top-k indices
    if sims.numel() == 0:
        print("Catalogue is empty — no matches found.")
        return []

    top_k = min(top_k, sims.numel())
    vals, idxs = torch.topk(sims, k=top_k)

    results = [(names[i], float(vals[j].item())) for j, i in enumerate(idxs.cpu().tolist())]

    # Print nicely
    print("Top matches:")
    i = 1
    for name, score in results:
        print(f"[{i}] \t{name}: \t{score*100:.4f}% match")
        i += 1
    

    return results
    
