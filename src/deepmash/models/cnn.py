from typing import override
import lightning as L
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from deepmash.data_processing.common import StemsSample

# -------------------------------------------------------------------------
# Utility: Compute Accuracy
# -------------------------------------------------------------------------

def get_accuracy(similarity: torch.Tensor) -> float:
    preds = similarity.argmax(dim=1)
    labels = torch.arange(similarity.size(0), device=similarity.device)
    return (preds == labels).float().mean().item()

def topk_accuracies(similarity: torch.Tensor, ks=(1, 5)) -> dict:
    """
    Computes Top-k accuracy for k in ks, handling cases where k > batch size.
    """
    labels = torch.arange(similarity.size(0), device=similarity.device)
    max_k = min(max(ks), similarity.size(1))  # Avoid k out of range
    _, topk_indices = similarity.topk(max_k, dim=1)

    results = {}
    for k in ks:
        k = min(k, similarity.size(1))  # Clip k
        correct = topk_indices[:, :k].eq(labels.unsqueeze(1)).any(dim=1)
        results[f"top_{k}"] = correct.float().mean().item()
    return results

# -------------------------------------------------------------------------
# CNN Model
# -------------------------------------------------------------------------

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

class CNNEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str = "resnet18",
        embedding_dim: int = 512, 
        dropout_p: float = 0.1,
        weights: str = "DEFAULT"
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.model_name = model_name

        # Get the CNN backbone from torchvision.models
        if hasattr(models, model_name):
            cnn_backbone = getattr(models, model_name)(weights=weights)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        
        # Handle different model architectures
        if model_name.startswith('resnet') or model_name.startswith('resnext'):
            # Remove the final fully connected layer
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            # Get the output feature dimension
            if hasattr(cnn_backbone, 'fc'):
                if model_name in ['resnet18', 'resnet34']:
                    feature_dim = 512
                else:
                    feature_dim = 2048
            else:
                # For newer versions of torchvision
                feature_dim = cnn_backbone.fc.in_features
                
        elif model_name.startswith('efficientnet'):
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            feature_dim = cnn_backbone.classifier[1].in_features
            
        elif model_name.startswith('densenet'):
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            feature_dim = cnn_backbone.classifier.in_features
            
        elif model_name.startswith('vgg'):
            self.features = cnn_backbone.features
            # For VGG, we need to compute the feature dimension
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, 64, 100)  # Adjust based on your input size
                feature_dim = self.features(dummy_input).view(1, -1).size(1)
                
        elif model_name.startswith('mobilenet'):
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            feature_dim = cnn_backbone.classifier[1].in_features
            
        elif model_name.startswith('convnext'):
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            feature_dim = cnn_backbone.classifier[2].in_features
            
        else:
            # Generic fallback - try to get features and compute dimension
            self.features = nn.Sequential(*list(cnn_backbone.children())[:-1])
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, 64, 100)
                feature_dim = self.features(dummy_input).view(1, -1).size(1)

        self.encoder = nn.Sequential(
            self.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )
        
        self.projection = nn.Linear(feature_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (B, 1, N_MELS=64, n_samples) """
        if x.ndim == 3: 
            x = x.unsqueeze(1)  # add channel dim
        
        # Handle single channel input for models expecting 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        features = self.encoder(x)
        projected = self.projection(features)
        return projected

class CNN(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = config.model_name
        self.learning_rate = config.learning_rate
        self.embedding_dim = config.embedding_dim
        self.dropout_p = config.dropout_p
        self.weights = config.weights
        self.weight_decay = config.weight_decay
        self.scheduler_patience = config.scheduler_patience
        self.scheduler_factor = config.scheduler_factor

        self.encoder = CNNEncoder(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim, 
            dropout_p=self.dropout_p,
            weights=self.weights
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh() # to [-1, 1]
        
        self.similarity = BilinearSimilarity(dim=self.embedding_dim) # denna skriver över configurationen
    
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
        names = batch.name
        vocal_embeddings = self.tanh(self.layer_norm(self.encoder(vocals)))
        non_vocal_embeddings = self.tanh(self.layer_norm(self.encoder(non_vocals)))
        return vocal_embeddings, non_vocal_embeddings, names

    def _step(self, batch: StemsSample) -> tuple[torch.Tensor, dict]:
        similarity = self(batch)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        accs = topk_accuracies(similarity, ks=(1, 5))
        return loss, accs
    
    def _log_metrics(self, prefix: str, accs: dict):
        for k, v in accs.items():
            self.log(f"{prefix}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _log(self, name: str, value: float):
        self.log(name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    @override
    def training_step(self, batch: StemsSample, batch_idx: int) -> torch.Tensor:
        loss, accs = self._step(batch)
        self.log("train_loss", loss.item(), prog_bar=True)
        self._log_metrics("train", accs)
        return loss

    @override
    def validation_step(self, batch: StemsSample, batch_idx: int) -> None:
        loss, accs = self._step(batch)
        self.log("val_loss", loss.item(), prog_bar=True)
        self._log_metrics("val", accs)
    
    @override 
    def test_step(self, batch: StemsSample, batch_idx: int) -> None:
        loss, accs = self._step(batch)
        self.log("test_loss", loss.item(), prog_bar=True)
        self._log_metrics("test", accs)
        self.print(f"[TEST] Loss: {loss.item():.4f} | "
                   f"Top-1: {accs['top_1']:.3f} | Top-5: {accs['top_5']:.3f}")
    
    @override
    def configure_optimizers(self):
        # Use AdamW with weight decay and fused optimization
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            fused=False
        )
        
        # Reduce learning rate when validation loss plateaus
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # Monitor validation loss
                patience=self.scheduler_patience,
                factor=self.scheduler_factor,
            ),
            'monitor': 'val_loss',    # Metric to monitor
            'interval': 'epoch',      # Check after each epoch
            'frequency': 1            # Check every epoch
        }
        
        return [optimizer], [scheduler]