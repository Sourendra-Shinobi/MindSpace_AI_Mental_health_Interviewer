"""
Wav2Vec 2.0 backbone with LoRA adapters.

Loads the pre-trained Wav2Vec 2.0 model from HuggingFace, injects
LoRA adapters into the attention layers (Q and V projections),
freezes all base weights, and returns all transformer hidden states
for weighted layer aggregation.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from peft import LoraConfig, get_peft_model
from typing import Optional, Tuple

class Wav2VecLoRA(nn.Module):
    """
    Wav2Vec 2.0 with LoRA adapters for parameter-efficient fine-tuning.

    Architecture:
        Raw waveform → CNN Feature Encoder (FROZEN) → Transformer (FROZEN + LoRA)
        → All 12 hidden states (skipping CNN output at index 0)

    The LoRA adapters are injected into the Q and V projection matrices
    of each transformer layer's self-attention. Only ~150K parameters
    are trainable (0.3% of the model).

    NOTE on hidden_states:
        Wav2Vec2Model.output_hidden_states returns a TUPLE of 13 tensors:
          - Index 0: CNN feature encoder output (projected + positional encoding)
          - Index 1-12: Outputs of transformer layers 1-12
        We return indices 1-12 only (the transformer layer outputs).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout for LoRA layers.
            lora_target_modules: Which attention modules to apply LoRA to.
        """
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        # Load pre-trained Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)

        # Get model config info
        self.hidden_size = self.wav2vec.config.hidden_size
        self.num_transformer_layers = self.wav2vec.config.num_hidden_layers

        # Freeze ALL base weights before applying LoRA
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # HF PEFT bugfix: Wav2Vec2Model's get_input_embeddings raises 
        # NotImplementedError, causing PEFT's gradient checkpointing setup to crash.
        # We override it with `feature_extractor` so PEFT's hook registration succeeds.
        # (Using feature_extractor instead of feature_projection prevents tuple grad crashes).
        self.wav2vec.get_input_embeddings = lambda *args, **kwargs: self.wav2vec.feature_extractor
        
        if hasattr(self.wav2vec.config, "gradient_checkpointing"):
            self.wav2vec.config.gradient_checkpointing = False

        # Apply LoRA adapters
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.wav2vec = get_peft_model(self.wav2vec, lora_config)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through Wav2Vec 2.0 with LoRA.

        Args:
            input_values: Raw waveform tensor [batch, T_samples].
            attention_mask: Optional mask [batch, T_samples] (1=real, 0=pad).

        Returns:
            Tuple of 12 hidden state tensors, each [batch, T_frames, hidden_size].
            (Transformer layer outputs only, CNN output excluded.)
        """
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states is a tuple of 13 tensors:
        # Index 0 = CNN encoder output (pre-transformer)
        # Index 1..12 = transformer layer outputs
        all_hidden_states = outputs.hidden_states

        # Return only transformer layer outputs (skip index 0)
        transformer_hidden_states = all_hidden_states[1:]

        return transformer_hidden_states

    def get_trainable_params(self) -> int:
        """Count trainable parameters (LoRA adapters only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def print_trainable_parameters(self):
        """Print trainable vs total parameters."""
        trainable = self.get_trainable_params()
        total = self.get_total_params()
        print(
            f"  Wav2Vec+LoRA: {trainable:,} trainable / {total:,} total "
            f"({100 * trainable / total:.2f}%)"
        )

    def merge_lora_weights(self):
        """
        Merge LoRA adapter weights into the base model for inference.

        After merging, the model runs at the same speed as the original
        Wav2Vec 2.0 with no LoRA overhead. Use this before ONNX export
        or production deployment.
        """
        self.wav2vec = self.wav2vec.merge_and_unload()
