"""
Weighted Layer Aggregation module.

Learns scalar weights for each transformer layer and computes a
weighted sum. This replaces the naive approach of using only the
final layer — intermediate layers (especially 6–10) carry more
emotion/prosody-relevant features (Zhang et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WeightedLayerAggregation(nn.Module):
    """
    Learnable weighted sum across all transformer hidden states.

    Given N hidden states (one per transformer layer), learn a scalar
    weight for each layer, apply softmax normalization, and compute
    a weighted sum. This adds only N parameters (e.g., 12 for
    wav2vec2-base).

    Input: Tuple of N tensors, each [batch, T, hidden_size]
    Output: Single tensor [batch, T, hidden_size]
    """

    def __init__(self, num_layers: int = 12):
        """
        Args:
            num_layers: Number of transformer layers to aggregate.
        """
        super().__init__()
        self.num_layers = num_layers

        # Learnable scalar weight per layer, initialized equally
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute weighted sum of hidden states across layers.

        Args:
            hidden_states: Tuple of N tensors [batch, T, hidden_size].

        Returns:
            Weighted sum tensor [batch, T, hidden_size].
        """
        assert len(hidden_states) == self.num_layers, (
            f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
        )

        # Softmax-normalize the weights so they sum to 1
        weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]

        # Stack: [num_layers, batch, T, hidden_size]
        stacked = torch.stack(hidden_states, dim=0)

        # Reshape weights for broadcasting: [num_layers, 1, 1, 1]
        weights = weights.view(-1, 1, 1, 1)

        # Weighted sum across layer dimension
        weighted = (weights * stacked).sum(dim=0)  # [batch, T, hidden_size]

        return weighted

    def get_layer_attention(self) -> torch.Tensor:
        """
        Get the current learned layer importance distribution.

        Useful for analysis — after training, check which layers
        the model considers most important for anxiety detection.

        Returns:
            Softmax-normalized weights [num_layers].
        """
        with torch.no_grad():
            return F.softmax(self.layer_weights, dim=0)
