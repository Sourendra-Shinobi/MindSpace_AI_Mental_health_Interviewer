"""
Feature Fusion layer.

Late fusion: concatenates Wav2Vec deep embeddings (768-d) with
eGeMAPS acoustic embeddings (128-d), then projects through a
shared linear layer with LayerNorm.
"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """
    Concatenation-based late fusion with learned projection.

    Combines the deep branch (Wav2Vec attention-pooled embedding)
    and the acoustic branch (eGeMAPS MLP projection) into a single
    fused representation.

    Uses LayerNorm (not BatchNorm) because:
    - The two branches have different distributional properties
    - LayerNorm is per-sample, stable with small batch sizes (16-32)
    - LayerNorm is invariant to batch size (important for inference)

    Input:  ([batch, deep_dim], [batch, acoustic_dim])
    Output: [batch, output_dim]
    """

    def __init__(
        self,
        deep_dim: int = 768,
        acoustic_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            deep_dim: Wav2Vec attention pooling output dimension.
            acoustic_dim: eGeMAPS projection output dimension.
            output_dim: Fused representation dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        concat_dim = deep_dim + acoustic_dim

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Store dims for ablation support
        self.deep_dim = deep_dim
        self.acoustic_dim = acoustic_dim
        self.output_dim = output_dim

    def forward(
        self,
        deep_embed: torch.Tensor,
        acoustic_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse deep and acoustic embeddings.

        Args:
            deep_embed: [batch, deep_dim] from Wav2Vec attention pooling.
            acoustic_embed: [batch, acoustic_dim] from eGeMAPS branch.

        Returns:
            Fused representation [batch, output_dim].
        """
        # Concatenate along feature dimension
        fused = torch.cat([deep_embed, acoustic_embed], dim=-1)  # [B, 896]
        return self.fusion(fused)
