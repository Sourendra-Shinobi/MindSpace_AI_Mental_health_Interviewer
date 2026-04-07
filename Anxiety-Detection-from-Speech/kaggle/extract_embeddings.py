"""
Kaggle Notebook 3 — Extract Speaker-Level Embeddings

Loads the trained binary classifier checkpoint (best_phase2.pt),
runs all participants through the backbone, extracts:
  - 768-d Wav2Vec embedding (from attention pooling)
  - 88-d raw eGeMAPS features
  - Concatenated 856-d feature vector per speaker

=== KAGGLE USAGE ===
Cell 1: !pip install -q transformers peft opensmile librosa soundfile
Cell 2: Paste or upload this file, then run.

=== INPUT ===
  /kaggle/working/checkpoints/best_phase2.pt    (from Notebook 2)
  /kaggle/working/labels.csv                     (from Notebook 1)
  /kaggle/input/your-dataset/*_P/ folders

=== OUTPUT (download these 3 files) ===
  /kaggle/working/embeddings/embeddings.npy     [N_speakers × 856]
  /kaggle/working/embeddings/phq_labels.npy     [N_speakers]
  /kaggle/working/embeddings/speaker_ids.npy    [N_speakers]
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
class ExtractConfig:
    # Paths
    DATA_DIR = "/kaggle/input/anxiety-dataset"
    LABELS_CSV = "/kaggle/working/labels.csv"
    CHECKPOINT_PATH = "/kaggle/working/checkpoints/best_phase2.pt"
    EGEMAPS_DIR = "/kaggle/working/egemaps"
    OUTPUT_DIR = "/kaggle/working/embeddings"

    # Audio
    SR = 16000
    SEGMENT_SEC = 10.0
    HOP_SEC = 5.0
    MIN_SEC = 3.0
    PARTICIPANT_ONLY = True

    # Model (must match training)
    WAV2VEC_MODEL = "facebook/wav2vec2-base"
    LORA_R = 8
    LORA_ALPHA = 16
    EGEMAPS_DIM = 88
    FUSION_DIM = 256

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4


# ─────────────────────────────────────────────
# AUDIO UTILITIES (same as kaggle_gpu_train.py)
# ─────────────────────────────────────────────
def load_audio(path: str, sr: int = ExtractConfig.SR) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def normalize(wav: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wav))
    return (wav / peak).astype(np.float32) if peak > 1e-6 else wav


def zero_pad(wav: np.ndarray, min_samples: int) -> np.ndarray:
    if len(wav) >= min_samples:
        return wav
    return np.concatenate([wav, np.zeros(min_samples - len(wav), dtype=np.float32)])


def get_participant_speech(wav: np.ndarray, transcript_path: Optional[str], sr: int) -> np.ndarray:
    if not transcript_path or not Path(transcript_path).exists():
        return wav
    try:
        df = pd.read_csv(transcript_path, sep="\t")
        rows = df[df["speaker"] == "Participant"]
        chunks = []
        for _, row in rows.iterrows():
            s = int(float(row["start_time"]) * sr)
            e = min(int(float(row["stop_time"]) * sr), len(wav))
            if e > s:
                chunks.append(wav[s:e])
        if chunks:
            return np.concatenate(chunks).astype(np.float32)
    except Exception:
        pass
    return wav


def make_segments(wav: np.ndarray, sr: int, seg_sec: float, hop_sec: float, min_sec: float) -> List[np.ndarray]:
    seg_samples = int(seg_sec * sr)
    hop_samples = int(hop_sec * sr)
    min_samples = int(min_sec * sr)
    if len(wav) <= seg_samples:
        return [zero_pad(wav, min_samples)]
    segments = []
    start = 0
    while start < len(wav):
        end = start + seg_samples
        chunk = wav[start:end]
        if len(chunk) >= min_samples:
            segments.append(zero_pad(chunk, seg_samples))
        start += hop_samples
    return segments if segments else [zero_pad(wav[:seg_samples], seg_samples)]


def extract_egemaps(wav: np.ndarray, sr: int) -> np.ndarray:
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_signal(wav, sr)
        return df.values.flatten().astype(np.float32)
    except Exception:
        return np.zeros(ExtractConfig.EGEMAPS_DIM, dtype=np.float32)


# ─────────────────────────────────────────────
# MODEL (same architecture as kaggle_gpu_train.py)
# ─────────────────────────────────────────────
class Wav2VecLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(ExtractConfig.WAV2VEC_MODEL)
        for p in self.backbone.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=ExtractConfig.LORA_R,
            lora_alpha=ExtractConfig.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def forward(self, wav, mask):
        out = self.backbone(
            input_values=wav, attention_mask=mask,
            output_hidden_states=True, return_dict=True,
        )
        return out.hidden_states[1:]


class WeightedLayerAggregation(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states):
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (w * torch.stack(hidden_states, dim=0)).sum(dim=0)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask=None):
        e = torch.tanh(self.attn(h))
        if mask is not None and mask.shape[1] != h.shape[1]:
            target = h.shape[1]
            m = mask.float().unsqueeze(1)
            k = max(mask.shape[1] // target, 1)
            pooled = F.avg_pool1d(m, kernel_size=k, stride=k).squeeze(1)
            if pooled.shape[1] > target:
                pooled = pooled[:, :target]
            elif pooled.shape[1] < target:
                pooled = torch.cat([pooled, torch.zeros(pooled.shape[0], target - pooled.shape[1], device=pooled.device)], dim=1)
            mask = (pooled > 0.5).long()
        if mask is not None:
            e = e.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        alpha = torch.nan_to_num(F.softmax(e, dim=1), nan=0.0)
        return (alpha * h).sum(dim=1)


class EmbeddingExtractor(nn.Module):
    """
    Same architecture as AnxietyClassifier but exposes the 768-d
    embedding BEFORE the classification head.
    """
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2VecLoRA()
        self.agg = WeightedLayerAggregation(12)
        self.pool = AttentionPooling(768)
        # Acoustic branch (not used for extraction — eGeMAPS is raw)
        self.egemaps = nn.Sequential(nn.Linear(ExtractConfig.EGEMAPS_DIM, 128), nn.ReLU(), nn.Dropout(0.2))
        # Fusion + head (loaded for weight compatibility)
        self.fusion = nn.Sequential(nn.Linear(768 + 128, ExtractConfig.FUSION_DIM), nn.LayerNorm(ExtractConfig.FUSION_DIM), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(ExtractConfig.FUSION_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def extract_wav2vec_embedding(self, wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract the 768-d embedding from the deep branch (before fusion)."""
        hidden_states = self.wav2vec(wav, mask)
        aggregated = self.agg(hidden_states)
        embedding = self.pool(aggregated, mask)  # [batch, 768]
        return embedding


def load_model(checkpoint_path: str, device: str) -> EmbeddingExtractor:
    """Load model from checkpoint, mapping trained weights."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    model = EmbeddingExtractor()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    checkpoint_state = checkpoint.get("model_state_dict", checkpoint)

    # Map keys: checkpoint uses AnxietyClassifier names which match EmbeddingExtractor
    model_state = model.state_dict()
    loaded = 0
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    print(f"  Loaded {loaded} weight tensors from checkpoint")
    return model


# ─────────────────────────────────────────────
# DATA DISCOVERY
# ─────────────────────────────────────────────
def discover_participants(data_dir: str) -> List[dict]:
    root = Path(data_dir)
    participants = []
    audio_files = sorted(root.glob("*_AUDIO.wav"))
    if not audio_files:
        audio_files = sorted(root.glob("*/*_AUDIO.wav"))
    for audio_path in audio_files:
        try:
            pid = int(audio_path.stem.replace("_AUDIO", ""))
        except ValueError:
            continue
        transcript = audio_path.parent / f"{pid}_TRANSCRIPT.csv"
        participants.append({
            "participant_id": pid,
            "audio_path": str(audio_path),
            "transcript_path": str(transcript) if transcript.exists() else None,
        })
    return participants


# ─────────────────────────────────────────────
# MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Speaker Embedding Extraction — Notebook 3")
    print("=" * 60)

    cfg = ExtractConfig
    device = cfg.DEVICE

    # Support local runs
    data_dir = cfg.DATA_DIR
    if not Path(data_dir).exists():
        for candidate in ["configs/dataset", "dataset"]:
            if Path(candidate).exists():
                data_dir = candidate
                break

    labels_csv = cfg.LABELS_CSV
    if not Path(labels_csv).exists():
        for candidate in ["labels.csv", "configs/dataset/labels.csv"]:
            if Path(candidate).exists():
                labels_csv = candidate
                break

    checkpoint_path = cfg.CHECKPOINT_PATH
    if not Path(checkpoint_path).exists():
        for candidate in ["checkpoints/wave2vec/best_phase2.pt", "checkpoints/best_phase2.pt"]:
            if Path(candidate).exists():
                checkpoint_path = candidate
                break

    output_dir = Path(cfg.OUTPUT_DIR)
    if not Path(cfg.OUTPUT_DIR).parent.exists():
        output_dir = Path("embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Data dir:       {data_dir}")
    print(f"  Labels CSV:     {labels_csv}")
    print(f"  Checkpoint:     {checkpoint_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Device:         {device}")
    print()

    # ── 1. Load labels ────────────────────────────────────────────────
    print("--- Step 1: Loading labels ---")
    if not Path(labels_csv).exists():
        print(f"ERROR: Labels CSV not found: {labels_csv}")
        print("Run kaggle_dataprep.py (Notebook 1) first.")
        return

    labels_df = pd.read_csv(labels_csv)
    phq_scores = {}
    for _, row in labels_df.iterrows():
        phq_scores[int(row["participant_id"])] = int(row["phq8_score"])
    print(f"  {len(phq_scores)} participants with PHQ-8 labels")

    # ── 2. Discover participants ──────────────────────────────────────
    print("\n--- Step 2: Discovering participant folders ---")
    participants = discover_participants(data_dir)
    print(f"  Found {len(participants)} participant folder(s)")

    # Filter to labeled participants only
    labeled = [p for p in participants if p["participant_id"] in phq_scores]
    print(f"  With PHQ-8 labels: {len(labeled)}")

    if not labeled:
        print("ERROR: No labeled participants found. Check dataset folder paths.")
        return

    # ── 3. Load model ─────────────────────────────────────────────────
    print("\n--- Step 3: Loading trained model ---")
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run kaggle_gpu_train.py (Notebook 2) first.")
        return
    model = load_model(checkpoint_path, device)

    # ── 4. Extract embeddings ─────────────────────────────────────────
    print("\n--- Step 4: Extracting speaker-level embeddings ---")

    all_embeddings = []   # will become [N, 856]
    all_phq = []          # will become [N]
    all_speaker_ids = []  # will become [N]

    for p in labeled:
        pid = p["participant_id"]
        phq = phq_scores[pid]
        print(f"\n  Participant {pid} (PHQ={phq}):")

        # Load and preprocess audio
        wav = load_audio(p["audio_path"])
        if cfg.PARTICIPANT_ONLY:
            wav = get_participant_speech(wav, p["transcript_path"], cfg.SR)
        wav = normalize(wav)
        duration = len(wav) / cfg.SR
        print(f"    Speech duration: {duration:.1f}s")

        # Create segments
        segments = make_segments(wav, cfg.SR, cfg.SEGMENT_SEC, cfg.HOP_SEC, cfg.MIN_SEC)
        print(f"    Segments: {len(segments)}")

        # Extract eGeMAPS for full participant speech (one vector per speaker)
        # Try loading pre-computed eGeMAPS from Notebook 1
        egemaps_path = Path(cfg.EGEMAPS_DIR) / f"{pid}_egemaps.npy"
        if not egemaps_path.exists():
            # Fallback: check local paths
            for candidate_dir in ["egemaps", "configs/dataset/egemaps"]:
                candidate = Path(candidate_dir) / f"{pid}_egemaps.npy"
                if candidate.exists():
                    egemaps_path = candidate
                    break

        if egemaps_path.exists():
            speaker_egemaps = np.load(str(egemaps_path)).astype(np.float32)
        else:
            # Extract on the fly
            speaker_egemaps = extract_egemaps(wav, cfg.SR)

        print(f"    eGeMAPS: shape={speaker_egemaps.shape}")

        # Run all segments through model to get 768-d embeddings
        segment_embeddings = []

        with torch.no_grad():
            for i in range(0, len(segments), cfg.BATCH_SIZE):
                batch_wavs = segments[i : i + cfg.BATCH_SIZE]

                # Prepare tensors
                max_len = max(len(w) for w in batch_wavs)
                padded_wavs = []
                masks = []
                for w in batch_wavs:
                    padded = np.concatenate([w, np.zeros(max_len - len(w), dtype=np.float32)])
                    mask = np.zeros(max_len, dtype=np.int64)
                    mask[:len(w)] = 1
                    padded_wavs.append(padded)
                    masks.append(mask)

                wav_tensor = torch.tensor(np.array(padded_wavs), dtype=torch.float32).to(device)
                mask_tensor = torch.tensor(np.array(masks), dtype=torch.long).to(device)

                # Extract 768-d embedding
                emb = model.extract_wav2vec_embedding(wav_tensor, mask_tensor)  # [B, 768]
                segment_embeddings.append(emb.cpu().numpy())

        # Average segment embeddings → one 768-d vector per speaker
        segment_embeddings = np.concatenate(segment_embeddings, axis=0)  # [num_segments, 768]
        speaker_wav2vec = segment_embeddings.mean(axis=0)                # [768]
        print(f"    Wav2Vec embedding: averaged over {segment_embeddings.shape[0]} segments → shape={speaker_wav2vec.shape}")

        # Concatenate: [768 wav2vec + 88 egemaps] = [856]
        speaker_full = np.concatenate([speaker_wav2vec, speaker_egemaps])
        print(f"    Full feature vector: shape={speaker_full.shape}")

        all_embeddings.append(speaker_full)
        all_phq.append(phq)
        all_speaker_ids.append(pid)

    # ── 5. Save outputs ───────────────────────────────────────────────
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    phq_array = np.array(all_phq, dtype=np.float32)
    ids_array = np.array(all_speaker_ids, dtype=np.int32)

    np.save(str(output_dir / "embeddings.npy"), embeddings_array)
    np.save(str(output_dir / "phq_labels.npy"), phq_array)
    np.save(str(output_dir / "speaker_ids.npy"), ids_array)

    print(f"\n--- Step 5: Saved outputs ---")
    print(f"  embeddings.npy:   shape={embeddings_array.shape}  ({embeddings_array.nbytes / 1024:.1f} KB)")
    print(f"  phq_labels.npy:   shape={phq_array.shape}")
    print(f"  speaker_ids.npy:  shape={ids_array.shape}")
    print(f"  Output dir:       {output_dir}")

    print(f"\n  PHQ score distribution:")
    print(f"    Min: {phq_array.min():.0f}")
    print(f"    Max: {phq_array.max():.0f}")
    print(f"    Mean: {phq_array.mean():.1f}")
    print(f"    Std: {phq_array.std():.1f}")

    print("\n" + "=" * 60)
    print("  Extraction complete!")
    print("  Download these 3 files for local regressor training:")
    print(f"    {output_dir / 'embeddings.npy'}")
    print(f"    {output_dir / 'phq_labels.npy'}")
    print(f"    {output_dir / 'speaker_ids.npy'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
