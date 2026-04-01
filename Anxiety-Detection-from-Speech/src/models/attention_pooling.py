"""
Attention Pooling module.

Collapses the temporal dimension using learned additive attention,
focusing on the most anxiety-salient frames rather than averaging
all frames equally. Anxiety markers (disfluencies, pitch spikes,
tremor) are frame-sparse — attention pooling learns to upweight them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Additive attention pooling over the time dimension.

    Computes attention scores for each frame, normalizes via softmax,
    and produces a weighted context vector. Adds only (hidden_dim + 1)
    parameters — 769 for wav2vec2-base (hidden_dim=768).

    Input:  [batch, T_frames, hidden_dim]
    Output: [batch, hidden_dim]
    """

    def __init__(self, hidden_dim: int = 768):
        """
        Args:
            hidden_dim: Dimension of input hidden states (matches Wav2Vec output).
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention-weighted context vector.

        Formula:
            e_t = tanh(W_a · h_t + b_a)     → [batch, T, 1]
            α_t = softmax(e_t, dim=1)         → [batch, T, 1]
            c   = Σ_t (α_t · h_t)            → [batch, hidden_dim]

        Args:
            hidden_states: [batch, T, hidden_dim] from weighted layer aggregation.
            attention_mask: Optional [batch, T] mask for Wav2Vec output frames.
                           If the waveform attention mask is provided, we need
                           to account for Wav2Vec's CNN downsampling.

        Returns:
            Context vector [batch, hidden_dim].
        """
        # Compute attention scores
        energy = torch.tanh(self.attention(hidden_states))  # [B, T, 1]

        # Mask padded positions if mask is provided
        if attention_mask is not None:
            # attention_mask may be for raw waveform, but hidden_states
            # is shorter due to CNN downsampling. Create a frame-level mask.
            if attention_mask.shape[1] != hidden_states.shape[1]:
                # Downsample mask to match hidden states length
                frame_mask = self._downsample_mask(
                    attention_mask, hidden_states.shape[1]
                )
            else:
                frame_mask = attention_mask

            # Set energy to -inf for padded frames
            energy = energy.masked_fill(
                frame_mask.unsqueeze(-1) == 0, float("-inf")
            )

        # Softmax over time dimension
        alpha = F.softmax(energy, dim=1)  # [B, T, 1]

        # Replace any NaN from all-masked softmax with zeros
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum
        context = (alpha * hidden_states).sum(dim=1)  # [B, hidden_dim]

        return context

    def _downsample_mask(
        self, mask: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """
        Downsample the raw waveform attention mask to match the frame
        length after Wav2Vec's CNN encoder (stride product = 320).

        Args:
            mask: [batch, T_samples] waveform-level mask.
            target_length: T_frames (after CNN downsampling).

        Returns:
            [batch, T_frames] frame-level mask.
        """
        # Use average pooling to approximate the downsampling
        mask_float = mask.float().unsqueeze(1)  # [B, 1, T_samples]
        kernel_size = mask.shape[1] // target_length
        if kernel_size < 1:
            kernel_size = 1
        pooled = F.avg_pool1d(
            mask_float, kernel_size=kernel_size, stride=kernel_size
        )
        pooled = pooled.squeeze(1)  # [B, T']

        # Trim or pad to exact target length
        if pooled.shape[1] > target_length:
            pooled = pooled[:, :target_length]
        elif pooled.shape[1] < target_length:
            pad = torch.zeros(
                pooled.shape[0],
                target_length - pooled.shape[1],
                device=pooled.device,
            )
            pooled = torch.cat([pooled, pad], dim=1)

        # Threshold: if any real sample exists in that frame window, mark as 1
        return (pooled > 0.5).long()

    def get_attention_weights(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get attention weights for visualization/analysis.

        Returns:
            Attention weights [batch, T, 1] after softmax.
        """
        with torch.no_grad():
            energy = torch.tanh(self.attention(hidden_states))
            if attention_mask is not None and attention_mask.shape[1] != hidden_states.shape[1]:
                frame_mask = self._downsample_mask(attention_mask, hidden_states.shape[1])
                energy = energy.masked_fill(frame_mask.unsqueeze(-1) == 0, float("-inf"))
            elif attention_mask is not None:
                energy = energy.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
            alpha = F.softmax(energy, dim=1)
            return torch.nan_to_num(alpha, nan=0.0)

