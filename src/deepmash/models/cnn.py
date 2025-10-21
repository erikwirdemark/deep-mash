import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

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
            x = x.repeat(1, 3, 1, 1)  # repeat single channel to 3 channels
            
        features = self.encoder(x)
        projected = self.projection(features)
        return projected

class CNN(L.LightningModule):
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        embedding_dim: int = 512, 
        dropout_p: float = 0.1,
        model_name: str = "resnet18",
        weights: str = "DEFAULT",
        similarity_type: str = "bilinear",  # "bilinear" or "contrastive"
        temperature: float = 0.1  # for contrastive similarity
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.model_name = model_name
        self.weights = weights
        self.similarity_type = similarity_type
        self.temperature = temperature
        
        self.encoder = CNNEncoder(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim, 
            dropout_p=self.dropout_p,
            weights=self.weights
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh() # to [-1, 1]
        
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)

    
    def forward(self, batch: StemsSample) -> torch.Tensor:
        vocals, non_vocals = batch.vocals, batch.non_vocals  # (B, N_MELS=64, n_samples)
        vocal_embeddings = self.tanh(self.layer_norm(self.encoder(vocals)))
        non_vocal_embeddings = self.tanh(self.layer_norm(self.encoder(non_vocals)))
        similarity = self.similarity(vocal_embeddings, non_vocal_embeddings)
        return similarity
    
    def get_pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise similarity for positive pairs (diagonal elements)"""
        x_emb = self.tanh(self.layer_norm(self.encoder(x)))
        y_emb = self.tanh(self.layer_norm(self.encoder(y)))
        return self.similarity.pairwise(x_emb, y_emb)