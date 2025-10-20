import torch

def zero_pad_or_clip(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if len(x) >= target_len:
        return x[:target_len]
    pad_len = target_len - len(x)
    return torch.cat([x, torch.zeros(pad_len)], dim=0)

def ensure_same_length(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    min_len = min(len(t) for t in tensors)
    return [t[:min_len] for t in tensors]

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")
