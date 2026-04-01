"""
Verification script — test imports, model construction, and forward pass.
Run from the project root: python verify.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import without error."""
    print("Testing imports...")

    from src.preprocessing.audio_utils import load_audio, normalize_audio, trim_silence
    print("  ✓ preprocessing.audio_utils")

    from src.preprocessing.segmentation import segment_audio, pad_audio, segment_or_pad
    print("  ✓ preprocessing.segmentation")

    from src.preprocessing.egemaps_extractor import EgemapsExtractor
    print("  ✓ preprocessing.egemaps_extractor")

    from src.data.augmentation import AudioAugmentor
    print("  ✓ data.augmentation")

    from src.data.collate import collate_fn
    print("  ✓ data.collate")

    from src.models.wav2vec_lora import Wav2VecLoRA
    print("  ✓ models.wav2vec_lora")

    from src.models.layer_aggregation import WeightedLayerAggregation
    print("  ✓ models.layer_aggregation")

    from src.models.attention_pooling import AttentionPooling
    print("  ✓ models.attention_pooling")

    from src.models.egemaps_branch import EgemapsBranch
    print("  ✓ models.egemaps_branch")

    from src.models.fusion import FusionLayer
    print("  ✓ models.fusion")

    from src.models.anxiety_classifier import AnxietyClassifier
    print("  ✓ models.anxiety_classifier")

    from src.training.losses import WeightedBCELoss, FocalLoss
    print("  ✓ training.losses")

    from src.training.metrics import compute_metrics, find_optimal_threshold
    print("  ✓ training.metrics")

    from src.inference.predictor import AnxietyPredictor
    print("  ✓ inference.predictor")

    print("\n✓ All imports passed!\n")


def test_components():
    """Test individual model components with dummy data."""
    import torch
    print("Testing model components...")

    # Layer Aggregation
    from src.models.layer_aggregation import WeightedLayerAggregation
    agg = WeightedLayerAggregation(num_layers=12)
    dummy_hidden = tuple(torch.randn(2, 50, 768) for _ in range(12))
    out = agg(dummy_hidden)
    assert out.shape == (2, 50, 768), f"LayerAgg shape mismatch: {out.shape}"
    print(f"  ✓ WeightedLayerAggregation: {tuple(out.shape)}")

    # Check learned weights
    weights = agg.get_layer_attention()
    assert weights.shape == (12,), f"Layer weights shape: {weights.shape}"
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights don't sum to 1"
    print(f"  ✓ Layer weights sum to 1.0: {weights.sum().item():.6f}")

    # Attention Pooling
    from src.models.attention_pooling import AttentionPooling
    pool = AttentionPooling(hidden_dim=768)
    out = pool(torch.randn(2, 50, 768))
    assert out.shape == (2, 768), f"AttentionPool shape mismatch: {out.shape}"
    print(f"  ✓ AttentionPooling: {tuple(out.shape)}")

    # eGeMAPS Branch
    from src.models.egemaps_branch import EgemapsBranch
    branch = EgemapsBranch(input_dim=88, proj_dim=128, dropout=0.2)
    out = branch(torch.randn(2, 88))
    assert out.shape == (2, 128), f"EgemapsBranch shape mismatch: {out.shape}"
    print(f"  ✓ EgemapsBranch: {tuple(out.shape)}")

    # Fusion
    from src.models.fusion import FusionLayer
    fusion = FusionLayer(deep_dim=768, acoustic_dim=128, output_dim=256)
    out = fusion(torch.randn(2, 768), torch.randn(2, 128))
    assert out.shape == (2, 256), f"Fusion shape mismatch: {out.shape}"
    print(f"  ✓ FusionLayer: {tuple(out.shape)}")

    # Losses
    from src.training.losses import WeightedBCELoss, FocalLoss
    bce = WeightedBCELoss(pos_weight=2.0)
    loss = bce(torch.randn(4, 1), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert loss.dim() == 0, "Loss should be scalar"
    print(f"  ✓ WeightedBCELoss: {loss.item():.4f}")

    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(torch.randn(4, 1), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    print(f"  ✓ FocalLoss: {loss.item():.4f}")

    # Metrics
    from src.training.metrics import compute_metrics
    import numpy as np
    metrics = compute_metrics(
        np.array([1, 0, 1, 0, 1]),
        np.array([0.8, 0.2, 0.6, 0.4, 0.9]),
    )
    print(f"  ✓ Metrics: AUC={metrics['auc_roc']:.3f}, UAR={metrics['uar']:.3f}")

    print("\n✓ All component tests passed!\n")


def test_full_model():
    """Test end-to-end model construction and forward pass."""
    import torch
    print("Testing full AnxietyClassifier (requires downloading wav2vec2-base ~360MB)...")
    print("  This will download the model from HuggingFace on first run.\n")

    from src.models.anxiety_classifier import AnxietyClassifier

    model = AnxietyClassifier(
        wav2vec_model="facebook/wav2vec2-base",
        lora_r=8,
        lora_alpha=16,
    )

    # Print parameter summary
    model.print_model_summary()

    # Forward pass with dummy data
    batch_size = 2
    audio_len = 16000 * 3  # 3 seconds at 16kHz
    waveform = torch.randn(batch_size, audio_len)
    egemaps = torch.randn(batch_size, 88)
    mask = torch.ones(batch_size, audio_len, dtype=torch.long)

    print("  Running forward pass...")
    with torch.no_grad():
        logits = model(waveform, egemaps, mask)

    assert logits.shape == (batch_size, 1), f"Output shape mismatch: {logits.shape}"
    probs = torch.sigmoid(logits)
    print(f"  ✓ Output shape: {tuple(logits.shape)}")
    print(f"  ✓ Logits: {logits.flatten().tolist()}")
    print(f"  ✓ Probabilities: {probs.flatten().tolist()}")

    # Check gradient flow
    model.train()
    logits = model(waveform, egemaps, mask)
    loss = logits.mean()
    loss.backward()

    # Verify LoRA params have gradients
    lora_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            lora_grads += 1
    print(f"  ✓ Parameters with gradients: {lora_grads}")

    print("\n✓ Full model test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Speech Anxiety Detection — Verification")
    print("=" * 60 + "\n")

    # Phase 1: Import check (always runs)
    test_imports()

    # Phase 2: Component tests (always runs)
    test_components()

    # Phase 3: Full model test (downloads wav2vec2-base)
    if "--full" in sys.argv:
        test_full_model()
    else:
        print("Skipping full model test (requires wav2vec2-base download).")
        print("Run with --full flag to include: python verify.py --full\n")

    print("=" * 60)
    print("  ✓ All verification passed!")
    print("=" * 60)
