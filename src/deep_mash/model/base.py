import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT

class MusicEncoder(nn.Module):
    def __init__(self, embedding_dim=128, cnn_name='resnet18', weights='ResNet18_Weights.IMAGENET1K_V1', n_heads=4, n_layers=2):
        super(MusicEncoder, self).__init__()

        # --- Pretrained CNN feature extractor ---
        base_cnn = getattr(models, cnn_name)(weights=weights)
        self.feature_extractor = nn.Sequential(*list(base_cnn.children())[:-2])
        self.cnn_out_channels = 512  # resnet18 output channels

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_out_channels,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Projection to embedding ---
        self.fc = nn.Linear(self.cnn_out_channels, embedding_dim)

    def forward(self, x):
        """
        Accept multiple mel input shapes and convert to [B,3,H,W] for the pretrained CNN.
        Supported shapes:
          - [B, 1, n_mels, time]
          - [B, n_mels, time]
          - [n_mels, time]
        """
        # Ensure tensor on same device and float
        x = x.float()

        # Handle different input dimensionalities robustly
        if x.dim() == 4:
            # [B, C, H, W]
            B, C, H, W = x.shape
            if C == 1:
                x = x.repeat(1, 3, 1, 1)
            elif C == 3:
                pass  # already RGB-like
            else:
                # Fallback: expand or truncate to 3 channels
                if C > 3:
                    x = x[:, :3, :, :]
                else:
                    reps = 3 // C + (1 if 3 % C != 0 else 0)
                    x = x.repeat(1, reps, 1, 1)[:, :3, :, :]
        elif x.dim() == 3:
            # [B, n_mels, time] -> add channel dim and repeat to 3 channels
            x = x.unsqueeze(1)  # [B,1,H,W]
            x = x.repeat(1, 3, 1, 1)
        elif x.dim() == 2:
            # [n_mels, time] -> add batch and channel dims
            x = x.unsqueeze(0).unsqueeze(1)  # [1,1,H,W]
            x = x.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported input shape {x.shape} for MusicEncoder")

        feats = self.feature_extractor(x)  # [B, C, H', W']
        B, C, H, W = feats.shape

        # Collapse frequency dimension
        feats = feats.mean(dim=2)           # [B, C, time]
        feats = feats.transpose(1, 2)       # [B, time, C]

        feats = self.transformer(feats)     # temporal modeling
        pooled = feats.mean(dim=1)          # global average pooling

        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalize
        return emb

class DualEncoderModel(nn.Module):
    def __init__(self, config):
        super(DualEncoderModel, self).__init__()
        self.config = config
        embedding_dim = config.model.embedding_dim
        self.vocal_encoder = MusicEncoder(embedding_dim)
        self.instr_encoder = MusicEncoder(embedding_dim)

    def forward(self, vocals, non_vocals):
        v_emb = self.vocal_encoder(vocals)
        i_emb = self.instr_encoder(non_vocals)
        return v_emb, i_emb
