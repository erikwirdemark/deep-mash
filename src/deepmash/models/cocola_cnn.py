from typing import override
import lightning as L
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

from deepmash.data_processing.common import StemsSample

class BilinearSimilarity(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(data=torch.Tensor(self.dim, self.dim))
        self.w.data.normal_(0, 0.05)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ out[i,j] = similarity(x[i], y[j]) """
        return x @ self.w @ y.t()
    
    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ out[i] = similarity(x[i], y[i]) (ie the diagonal of the full pairwise matrix, but this is more efficient) """
        return ((x @ self.w) * y).sum(dim=-1)

class EfficientNetEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p

        self.encoder = nn.Sequential(
            EfficientNet.from_name("efficientnet-b0", include_top=False, in_channels=1),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )
        self.projection = nn.Linear(1280, self.embedding_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (B, 1, N_MELS=64, n_samples) """
        if x.ndim == 3: x = x.unsqueeze(1)  # add channel dim
        
        embeddings = self.encoder(x)
        projected = self.projection(embeddings)
        return projected

def get_accuracy(similarity: torch.Tensor) -> float:
    preds = similarity.argmax(dim=1)
    labels = torch.arange(similarity.size(0), device=similarity.device)
    return (preds == labels).float().mean().item()

class CocolaCNN(L.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = config.learning_rate
        self.embedding_dim = config.embedding_dim
        self.dropout_p = config.dropout_p

        self.encoder = EfficientNetEncoder(embedding_dim=self.embedding_dim, dropout_p=self.dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh() # to [-1, 1]
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)
    
    @override
    def forward(self, batch: StemsSample) -> torch.Tensor:
        """ 
        Returns a (B, B) similarity matrix S, where S[i, j] is the models unnormalized
        log-probability that vocals[i] belongs to non_vocals[j].
        """
        vocals, non_vocals = batch.vocals, batch.non_vocals  # (B, N_MELS=64, n_samples)
        vocal_embeddings = self.tanh(self.layer_norm(self.encoder(vocals)))
        non_vocal_embeddings = self.tanh(self.layer_norm(self.encoder(non_vocals)))
        similarity = self.similarity(vocal_embeddings, non_vocal_embeddings)
        return similarity
    
    def compute_embeddings(self, batch: StemsSample) -> torch.Tensor:
        vocals, non_vocals = batch.vocals, batch.non_vocals  # (B, N_MELS=64, n_samples)
        vocal_embeddings = self.tanh(self.layer_norm(self.encoder(vocals)))
        non_vocal_embeddings = self.tanh(self.layer_norm(self.encoder(non_vocals)))
        v_emb = vocal_embeddings.detach().cpu()
        i_emb = non_vocal_embeddings.detach().cpu()
        names = batch.name
        return v_emb, i_emb, names
    
    def compute_vocal_embedding(self, vocal) -> torch.Tensor:
        vocal_embedding = self.tanh(self.layer_norm(self.encoder(vocal)))
        return vocal_embedding

    def _step(self, batch: StemsSample) -> tuple[torch.Tensor, float]:
        similarity = self(batch)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        accuracy = get_accuracy(similarity)
        return loss, accuracy

    def _log(self, name: str, value: float):
        self.log(name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    @override
    def training_step(self, batch: StemsSample, batch_idx: int) -> torch.Tensor:
        loss, accuracy = self._step(batch)
        self._log("train_accuracy", accuracy)
        self._log("train_loss", loss.item())
        return loss

    @override
    def validation_step(self, batch: StemsSample, batch_idx: int) -> None:
        loss, accuracy = self._step(batch)
        self._log("val_accuracy", accuracy)
        self._log("val_loss", loss.item())
    
    @override 
    def test_step(self, batch: StemsSample, batch_idx: int) -> None:
        loss, accuracy = self._step(batch)
        self._log("test_accuracy", accuracy)
        self._log("test_loss", loss.item())
    
    @override
    def configure_optimizers(self):
        # use AdamW (>Adam), testing fused=True optimization because think its usually faster 
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, fused=False)