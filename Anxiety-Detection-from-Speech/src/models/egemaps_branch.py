"""
eGeMAPS acoustic feature branch.

Projects the 88-dimensional eGeMAPS vector into a 128-d embedding
space for fusion with the Wav2Vec deep branch. Intentionally
lightweight — eGeMAPS features are already clinically curated
and well-conditioned.
"""

import torch
import torch.nn as nn


class EgemapsBranch(nn.Module):
    """
    MLP projection for eGeMAPS acoustic features.

    Single linear layer + ReLU + Dropout projection from the raw
    88-d eGeMAPS space into 128-d fusion-ready embeddings.

    Input:  [batch, 88]
    Output: [batch, 128]
    """

    def __init__(
        self,
        input_dim: int = 88,
        proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: eGeMAPS feature dimension (88 for eGeMAPSv02 Functionals).
            proj_dim: Output projection dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project eGeMAPS features into embedding space.

        Args:
            features: [batch, input_dim] eGeMAPS feature vectors.

        Returns:
            [batch, proj_dim] projected embeddings.
        """
        return self.projection(features)
