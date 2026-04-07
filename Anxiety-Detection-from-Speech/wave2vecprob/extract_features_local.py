"""
Local Feature Extraction — Frozen Wav2Vec2 + eGeMAPS

Extracts 856-d speaker-level feature vectors from 5 local DAIC-WOZ audio files
using frozen Wav2Vec2-base (768-d) and openSMILE eGeMAPS (88-d).

No fine-tuning, no LoRA, no checkpoint required.
Runs locally on CPU or GPU.

=== USAGE ===
  pip install transformers opensmile librosa soundfile torch pandas numpy
  python extract_features_local.py

=== OUTPUT ===
  embeddings/X.npy            [N_speakers x 856]
  embeddings/y.npy            [N_speakers]       PHQ-8 scores
  embeddings/speaker_ids.npy  [N_speakers]       participant IDs
"""

import gc
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = Path("configs/dataset")
OUTPUT_DIR = Path("embeddings")

# Audio settings
SR = 16000
SEGMENT_SEC = 10.0
HOP_SEC = 5.0
MIN_SEC = 3.0
PARTICIPANT_ONLY = True
MAX_AUDIO_SEC = 600.0    # cap at 10 min to avoid OOM (covers most interviews)
EGEMAPS_MAX_SEC = 90.0   # cap eGeMAPS input to avoid opensmile crash on very long audio

# Model
WAV2VEC_MODEL = "facebook/wav2vec2-base"
HIDDEN_SIZE = 768
NUM_LAYERS = 12
EGEMAPS_DIM = 88

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


# ─────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────
def load_audio(path: str, sr: int = SR, max_sec: float = MAX_AUDIO_SEC) -> np.ndarray:
    """
    Load audio file using soundfile (memory-efficient) then resample.
    Caps to max_sec to prevent OOM on very long recordings.
    """
    info = sf.info(path)
    max_frames = int(max_sec * info.samplerate)
    data, file_sr = sf.read(path, frames=max_frames, dtype="float32", always_2d=False)
    # Convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample to target SR if needed
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return data.astype(np.float32)


def normalize(wav: np.ndarray) -> np.ndarray:
    """Peak normalization to [-1, 1]."""
    peak = np.max(np.abs(wav))
    return (wav / peak).astype(np.float32) if peak > 1e-6 else wav


def zero_pad(wav: np.ndarray, min_samples: int) -> np.ndarray:
    """Zero-pad waveform to minimum length."""
    if len(wav) >= min_samples:
        return wav
    return np.concatenate([wav, np.zeros(min_samples - len(wav), dtype=np.float32)])


def get_participant_speech(
    wav: np.ndarray, transcript_path: Optional[str], sr: int
) -> np.ndarray:
    """Extract only participant speech from DAIC-WOZ interview audio."""
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
    except Exception as ex:
        print(f"    Warning: Could not parse transcript: {ex}")
    return wav


def make_segments(
    wav: np.ndarray,
    sr: int,
    seg_sec: float,
    hop_sec: float,
    min_sec: float,
) -> List[np.ndarray]:
    """Split long audio into overlapping segments."""
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


def extract_egemaps(wav: np.ndarray, sr: int, max_sec: float = EGEMAPS_MAX_SEC) -> np.ndarray:
    """
    Extract 88-d eGeMAPSv02 features using openSMILE.
    Clips audio to max_sec to avoid opensmile internal crashes on long signals.
    """
    try:
        import opensmile

        # Clip audio to avoid opensmile "Unknown exception" on very long signals
        max_samples = int(max_sec * sr)
        wav_clip = wav[:max_samples] if len(wav) > max_samples else wav

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_signal(wav_clip, sr)
        return df.values.flatten().astype(np.float32)
    except ImportError:
        print("ERROR: opensmile not installed! pip install opensmile")
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)
    except Exception as e:
        print(f"    eGeMAPS extraction error: {e} — using zero vector")
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)


# ─────────────────────────────────────────────
# FROZEN WAV2VEC2 FEATURE EXTRACTOR
# ─────────────────────────────────────────────
class FrozenWav2VecExtractor(nn.Module):
    """
    Frozen Wav2Vec2-base used purely as a feature extractor.
    No LoRA, no fine-tuning, no checkpoint needed.

    Architecture:
      Raw waveform → Conv Feature Encoder (frozen)
                   → 12 Transformer layers (frozen)
                   → Weighted Layer Aggregation (uniform weights)
                   → Mean Pooling over time
                   → 768-d embedding per segment
    """

    def __init__(self, model_name: str = WAV2VEC_MODEL, num_layers: int = NUM_LAYERS):
        super().__init__()
        print(f"  Loading {model_name} from HuggingFace (frozen, no LoRA)...")
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze ALL parameters — no training at all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Uniform layer weights (not trained)
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers), requires_grad=False
        )

    @torch.no_grad()
    def forward(
        self, wav: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract 768-d embedding from raw waveform.

        Args:
            wav:  [batch, T_samples] raw waveform
            mask: [batch, T_samples] attention mask (1=real, 0=pad)

        Returns:
            [batch, 768] speaker embedding
        """
        out = self.backbone(
            input_values=wav,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 12 transformer hidden states (skip the CNN output at index 0)
        hidden_states = out.hidden_states[1:]  # tuple of 12 × [B, T_frames, 768]

        # Weighted layer aggregation (uniform → simple average of all layers)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        stacked = torch.stack(hidden_states, dim=0)  # [12, B, T, 768]
        aggregated = (w * stacked).sum(dim=0)  # [B, T, 768]

        # Mean pooling over time dimension
        if mask is not None:
            # Downsample mask to match transformer frame rate
            T_frames = aggregated.shape[1]
            m = mask.float().unsqueeze(1)  # [B, 1, T_samples]
            k = max(mask.shape[1] // T_frames, 1)
            pooled_mask = F.avg_pool1d(m, kernel_size=k, stride=k).squeeze(1)
            # Align lengths
            if pooled_mask.shape[1] > T_frames:
                pooled_mask = pooled_mask[:, :T_frames]
            elif pooled_mask.shape[1] < T_frames:
                pad = torch.zeros(
                    pooled_mask.shape[0],
                    T_frames - pooled_mask.shape[1],
                    device=pooled_mask.device,
                )
                pooled_mask = torch.cat([pooled_mask, pad], dim=1)
            frame_mask = (pooled_mask > 0.5).float()

            # Masked mean
            aggregated = aggregated * frame_mask.unsqueeze(-1)
            embedding = aggregated.sum(dim=1) / (
                frame_mask.sum(dim=1, keepdim=True) + 1e-9
            )
        else:
            embedding = aggregated.mean(dim=1)

        return embedding  # [B, 768]


# ─────────────────────────────────────────────
# PARTICIPANT DISCOVERY
# ─────────────────────────────────────────────
def discover_participants(data_dir: Path) -> List[dict]:
    """Find all participant folders with audio + PHQ labels."""
    # Load PHQ labels from all split CSVs
    phq_scores = {}
    for csv_name in [
        "train_split_Depression_AVEC2017.csv",
        "dev_split_Depression_AVEC2017.csv",
        "test_split_Depression_AVEC2017.csv",
        "full_test_split.csv",
    ]:
        csv_path = data_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Handle variations in column names across different splits
                pid_col = "Participant_ID" if "Participant_ID" in df.columns else "participant_ID"
                if "PHQ8_Score" in df.columns:
                    phq_col = "PHQ8_Score"
                elif "PHQ_Score" in df.columns:
                    phq_col = "PHQ_Score"
                else:
                    continue # Skip if no PHQ score column

                pid = int(row[pid_col])
                phq_scores[pid] = int(row[phq_col])

    print(f"Loaded PHQ labels for {len(phq_scores)} participants from CSVs")
    # Find participant folders with audio
    participants = []
    for folder in sorted(data_dir.glob("*_P")):
        pid_str = folder.name.replace("_P", "")
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)

        audio_path = folder / f"{pid}_AUDIO.wav"
        transcript_path = folder / f"{pid}_TRANSCRIPT.csv"

        if audio_path.exists() and pid in phq_scores:
            participants.append(
                {
                    "participant_id": pid,
                    "audio_path": str(audio_path),
                    "transcript_path": (
                        str(transcript_path) if transcript_path.exists() else None
                    ),
                    "phq_score": phq_scores[pid],
                }
            )

    return participants


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Local Feature Extraction")
    print("Frozen Wav2Vec2 (768-d) + eGeMAPS (88-d) → 856-d")
    print("=" * 60)

    # Discover data
    print("\n--- Step 1: Discovering participants ---")
    participants = discover_participants(DATA_DIR)

    if not participants:
        print("ERROR: No participant folders found with audio!")
        print(f"Searched in: {DATA_DIR.resolve()}")
        print("Expected: <pid>_P/<pid>_AUDIO.wav folders")
        return

    print(f"  Found {len(participants)} participants with audio + labels:")
    for p in participants:
        print(f"    {p['participant_id']}: PHQ={p['phq_score']}  audio={Path(p['audio_path']).name}")

    # Load model
    print("\n--- Step 2: Loading frozen Wav2Vec2 model ---")
    model = FrozenWav2VecExtractor().to(DEVICE).eval()
    print(f"Device: {DEVICE}")
    print(f"Trainable params: 0 (fully frozen)")

    # Extract features
    print("\n--- Step 3: Extracting features per participant ---")
    all_embeddings = []
    all_phq = []
    all_pids = []

    for p in participants:
        pid = p["participant_id"]
        phq = p["phq_score"]
        print(f"\n  Participant {pid} (PHQ={phq}):")

        try:
            # Load and preprocess audio (capped at MAX_AUDIO_SEC to prevent OOM)
            wav = load_audio(p["audio_path"])
            original_duration = len(wav) / SR
            print(f"    Full audio: {original_duration:.1f}s (capped at {MAX_AUDIO_SEC:.0f}s)")

            if PARTICIPANT_ONLY and p["transcript_path"]:
                wav = get_participant_speech(wav, p["transcript_path"], SR)
                print(f"    Participant-only speech: {len(wav)/SR:.1f}s")

            wav = normalize(wav)

            # Extract eGeMAPS (clipped to EGEMAPS_MAX_SEC internally)
            egemaps_vec = extract_egemaps(wav, SR)
            print(f"    eGeMAPS: shape={egemaps_vec.shape}")

            # Segment audio
            segments = make_segments(wav, SR, SEGMENT_SEC, HOP_SEC, MIN_SEC)
            print(f"    Segments: {len(segments)} x {SEGMENT_SEC}s")

            # Free the full wav from memory now that segments are made
            del wav
            gc.collect()

            # Run frozen Wav2Vec2 on all segments
            segment_embeddings = []
            with torch.no_grad():
                for i in range(0, len(segments), BATCH_SIZE):
                    batch_wavs = segments[i : i + BATCH_SIZE]
                    max_len = max(len(w) for w in batch_wavs)

                    padded_wavs = []
                    masks = []
                    for w in batch_wavs:
                        padded = np.concatenate(
                            [w, np.zeros(max_len - len(w), dtype=np.float32)]
                        )
                        mask = np.zeros(max_len, dtype=np.int64)
                        mask[: len(w)] = 1
                        padded_wavs.append(padded)
                        masks.append(mask)

                    wav_t = torch.tensor(
                        np.array(padded_wavs), dtype=torch.float32
                    ).to(DEVICE)
                    mask_t = torch.tensor(
                        np.array(masks), dtype=torch.long
                    ).to(DEVICE)

                    emb = model(wav_t, mask_t)  # [batch, 768]
                    segment_embeddings.append(emb.cpu().numpy())

                    # Free GPU memory after each batch
                    del wav_t, mask_t
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            # Free segment list
            del segments
            gc.collect()

            # Average all segment embeddings → one 768-d vector per speaker
            segment_embeddings = np.concatenate(segment_embeddings, axis=0)
            speaker_wav2vec = segment_embeddings.mean(axis=0)  # [768]
            print(
                f"    Wav2Vec2: averaged {segment_embeddings.shape[0]} segments → [{speaker_wav2vec.shape[0]}]"
            )

            # Concatenate: [768 wav2vec + 88 egemaps] = [856]
            full_vec = np.concatenate([speaker_wav2vec, egemaps_vec])
            print(f"    Full feature vector: [{full_vec.shape[0]}]")

            all_embeddings.append(full_vec)
            all_phq.append(phq)
            all_pids.append(pid)

        except Exception as e:
            print(f"    ERROR processing participant {pid}: {e}")
            print(f"    Skipping participant {pid} and continuing...")
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue

    # Save outputs
    print("\n--- Step 4: Saving outputs ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.array(all_embeddings, dtype=np.float32)
    y = np.array(all_phq, dtype=np.float32)
    ids = np.array(all_pids, dtype=np.int32)

    np.save(str(OUTPUT_DIR / "X.npy"), X)
    np.save(str(OUTPUT_DIR / "y.npy"), y)
    np.save(str(OUTPUT_DIR / "speaker_ids.npy"), ids)

    # Also save in the format train_regressor.py expects
    np.save(str(OUTPUT_DIR / "embeddings.npy"), X)
    np.save(str(OUTPUT_DIR / "phq_labels.npy"), y)

    print(f"X (embeddings):   shape={X.shape}  ({X.nbytes / 1024:.1f} KB)")
    print(f"y (PHQ scores):   shape={y.shape}  values={y.tolist()}")
    print(f"speaker_ids:      shape={ids.shape}  values={ids.tolist()}")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")

    print(f"\n  PHQ-8 distribution:")
    print(f"Min:  {y.min():.0f}")
    print(f"Max:  {y.max():.0f}")
    print(f"Mean: {y.mean():.1f}")
    print(f"Std:  {y.std():.1f}")

    print("\n" + "=" * 60)
    print("  Extraction complete!")
    print("  Next: python train_local.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
