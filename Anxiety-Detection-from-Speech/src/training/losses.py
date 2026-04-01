"""
Loss functions for anxiety classification.

Includes:
- WeightedBCELoss: BCEWithLogitsLoss with computed pos_weight for class imbalance
- FocalLoss: Focuses on hard examples, useful when easy negatives dominate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with Logits, weighted for class imbalance.

    Automatically computes pos_weight from the training set's class
    distribution: pos_weight = num_negative / num_positive.
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Args:
            pos_weight: Weight for positive class. Compute from data as
                        (num_non_anxious / num_anxious).
        """
        super().__init__()
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output [batch, 1] (before sigmoid).
            targets: Binary labels [batch] or [batch, 1].

        Returns:
            Scalar loss.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

    @classmethod
    def from_label_counts(cls, num_positive: int, num_negative: int):
        """Create from class counts."""
        pos_weight = num_negative / max(num_positive, 1)
        return cls(pos_weight=pos_weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Reduces the loss contribution from easy examples, focusing
    training on hard misclassifications. Useful when the model
    quickly learns the majority class but struggles with edge cases.

    FL(p) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (0–1).
            gamma: Focusing parameter. Higher → more focus on hard examples.
                   gamma=0 → standard BCE. gamma=2 is the default recommendation.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output [batch, 1] (before sigmoid).
            targets: Binary labels [batch] or [batch, 1].

        Returns:
            Scalar focal loss.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulation
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss
        return loss.mean()
