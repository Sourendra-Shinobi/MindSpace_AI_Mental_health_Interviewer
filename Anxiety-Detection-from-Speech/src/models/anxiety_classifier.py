"""
Full end-to-end Anxiety Classifier: Dual-Branch Fusion Architecture.

Assembles all components:
    Wav2Vec 2.0 + LoRA → Weighted Layer Aggregation → Attention Pooling
                                                          ↓
                                        Fusion ← eGeMAPS MLP Branch
                                          ↓
                                   Classification Head
                                          ↓
                                    Anxiety Logit
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .wav2vec_lora import Wav2VecLoRA
from .layer_aggregation import WeightedLayerAggregation
from .attention_pooling import AttentionPooling
from .egemaps_branch import EgemapsBranch
from .fusion import FusionLayer


class AnxietyClassifier(nn.Module):
    """
    Dual-Branch Fusion model for speech anxiety detection.

    Deep Branch:
        Raw waveform → Wav2Vec 2.0 + LoRA → Weighted Layer Aggregation
        → Attention Pooling → [768]

    Acoustic Branch:
        eGeMAPS features → MLP Projection → [128]

    Fusion + Classification:
        Concat([768, 128]) → Fusion Layer → [256] → FC Head → [1] logit

    Total trainable parameters: ~200–300K
    (LoRA adapters + layer weights + attention pooling + eGeMAPS MLP
     + fusion + FC head)
    """

    def __init__(
        self,
        wav2vec_model: str = "facebook/wav2vec2-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        num_transformer_layers: int = 12,
        hidden_size: int = 768,
        egemaps_dim: int = 88,
        egemaps_proj_dim: int = 128,
        egemaps_dropout: float = 0.2,
        fusion_output_dim: int = 256,
        fusion_dropout: float = 0.3,
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.3,
    ):
        """
        Args:
            wav2vec_model: HuggingFace model name for Wav2Vec 2.0.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: LoRA dropout.
            lora_target_modules: LoRA target modules (default: ["q_proj", "v_proj"]).
            num_transformer_layers: Number of transformer layers (12 for base).
            hidden_size: Transformer hidden dimension (768 for base).
            egemaps_dim: eGeMAPS feature dimension (88).
            egemaps_proj_dim: eGeMAPS projection dimension (128).
            egemaps_dropout: Dropout for eGeMAPS MLP.
            fusion_output_dim: Fusion layer output dimension (256).
            fusion_dropout: Dropout for fusion layer.
            classifier_hidden: FC head hidden dimension (64).
            classifier_dropout: Dropout for FC head.
        """
        super().__init__()

        # === Deep Branch ===
        self.wav2vec = Wav2VecLoRA(
            model_name=wav2vec_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        self.layer_aggregation = WeightedLayerAggregation(
            num_layers=num_transformer_layers,
        )

        self.attention_pooling = AttentionPooling(
            hidden_dim=hidden_size,
        )

        # === Acoustic Branch ===
        self.egemaps_branch = EgemapsBranch(
            input_dim=egemaps_dim,
            proj_dim=egemaps_proj_dim,
            dropout=egemaps_dropout,
        )

        # === Fusion ===
        self.fusion = FusionLayer(
            deep_dim=hidden_size,
            acoustic_dim=egemaps_proj_dim,
            output_dim=fusion_output_dim,
            dropout=fusion_dropout,
        )

        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        egemaps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: dual-branch → fusion → logit.

        Args:
            waveform: [batch, T_samples] raw audio waveform.
            egemaps: [batch, 88] eGeMAPS feature vectors.
            attention_mask: Optional [batch, T_samples] for padded waveforms.

        Returns:
            Logits tensor [batch, 1] (apply sigmoid for probability).
        """
        # --- Deep Branch ---
        # Wav2Vec 2.0 + LoRA → all transformer hidden states
        hidden_states = self.wav2vec(waveform, attention_mask)
        # Tuple of 12 tensors, each [B, T_frames, 768]

        # Weighted layer aggregation → [B, T_frames, 768]
        aggregated = self.layer_aggregation(hidden_states)

        # Attention pooling → [B, 768]
        deep_embed = self.attention_pooling(aggregated, attention_mask)

        # --- Acoustic Branch ---
        # eGeMAPS MLP → [B, 128]
        acoustic_embed = self.egemaps_branch(egemaps)

        # --- Fusion ---
        # Concat + project → [B, 256]
        fused = self.fusion(deep_embed, acoustic_embed)

        # --- Classification ---
        # FC head → [B, 1]
        logits = self.classifier(fused)

        return logits

    def predict(
        self,
        waveform: torch.Tensor,
        egemaps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: forward pass + sigmoid + thresholding.

        Args:
            waveform: [batch, T_samples] raw waveform.
            egemaps: [batch, 88] eGeMAPS features.
            attention_mask: Optional mask.
            threshold: Decision threshold for binary classification.

        Returns:
            Dict with 'logits', 'probabilities', 'predictions'.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(waveform, egemaps, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        }

    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable parameter counts by component."""
        components = {
            "wav2vec_lora": sum(
                p.numel() for p in self.wav2vec.parameters() if p.requires_grad
            ),
            "layer_aggregation": sum(
                p.numel()
                for p in self.layer_aggregation.parameters()
                if p.requires_grad
            ),
            "attention_pooling": sum(
                p.numel()
                for p in self.attention_pooling.parameters()
                if p.requires_grad
            ),
            "egemaps_branch": sum(
                p.numel()
                for p in self.egemaps_branch.parameters()
                if p.requires_grad
            ),
            "fusion": sum(
                p.numel() for p in self.fusion.parameters() if p.requires_grad
            ),
            "classifier": sum(
                p.numel() for p in self.classifier.parameters() if p.requires_grad
            ),
        }
        components["total"] = sum(components.values())
        return components

    def print_model_summary(self):
        """Print a summary of trainable parameters by component."""
        params = self.get_trainable_params()
        print("\n" + "=" * 55)
        print("  Anxiety Classifier — Trainable Parameter Summary")
        print("=" * 55)
        for name, count in params.items():
            if name != "total":
                print(f"  {name:25s}: {count:>10,}")
        print("-" * 55)
        print(f"  {'TOTAL':25s}: {params['total']:>10,}")
        print("=" * 55 + "\n")
