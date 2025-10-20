import lightning as L
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (B, 1, N_MELS=64, n_samples) """
        if x.ndim == 3: x = x.unsqueeze(1)  # add channel dim
        
        print(f"input shape: {x.shape}")
        embeddings = self.encoder(x)
        print(f"encoded embedding shape: {embeddings.shape}")
        projected = self.projection(embeddings)
        return projected

class CocolaCNN(L.LightningModule):
    def __init__(self, learning_rate: float = 0.001, embedding_dim: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        
        self.encoder = EfficientNetEncoder(embedding_dim=self.embedding_dim, dropout_p=self.dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh() # to [-1, 1]
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)
    
    def forward(self, batch: StemsSample) -> torch.Tensor:
        vocals, non_vocals = batch.vocals, batch.non_vocals  # (B, N_MELS=64, n_samples)
        vocal_embeddings = self.tanh(self.layer_norm(self.encoder(vocals)))
        non_vocal_embeddings = self.tanh(self.layer_norm(self.encoder(non_vocals)))
        similarity = self.similarity(vocal_embeddings, non_vocal_embeddings)
        return similarity
