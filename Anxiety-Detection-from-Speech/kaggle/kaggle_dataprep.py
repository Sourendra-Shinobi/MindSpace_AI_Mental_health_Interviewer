"""
Kaggle Notebook 1 — DAIC-WOZ Data Preparation

Scans participant folders, merges PHQ-8 labels from AVEC2017 CSVs,
generates a unified labels.csv, and pre-extracts eGeMAPS per participant.

=== KAGGLE USAGE ===
Cell 1: !pip install -q opensmile librosa soundfile
Cell 2: Paste or upload this file, then run.

=== INPUT ===
  /kaggle/input/your-dataset/
      train_split_Depression_AVEC2017.csv
      (optional) dev_split_Depression_AVEC2017.csv
      301_P/301_AUDIO.wav, 301_P/301_TRANSCRIPT.csv, ...

=== OUTPUT ===
  /kaggle/working/labels.csv
  /kaggle/working/egemaps/
      301_egemaps.npy, 302_egemaps.npy, ...
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION — Edit for your Kaggle paths
# ─────────────────────────────────────────────
class PrepConfig:
    # Root directory containing participant folders + AVEC CSV(s)
    DATA_DIR = "/kaggle/input/anxiety-dataset"

    # Output directory
    OUTPUT_DIR = "/kaggle/working"

    # Audio settings
    SR = 16000

    # Train/Val split ratio (by speaker). Test speakers = those in test CSV.
    VAL_RATIO = 0.15

    SEED = 42


# ─────────────────────────────────────────────
# STEP 1: LOAD AVEC2017 LABELS
# ─────────────────────────────────────────────
def load_avec_labels(data_dir: str) -> Dict[int, dict]:
    """
    Load PHQ-8 labels from AVEC2017 CSV files.

    The DAIC-WOZ dataset distributes labels across:
      - train_split_Depression_AVEC2017.csv  (has PHQ8_Score)
      - dev_split_Depression_AVEC2017.csv    (has PHQ8_Score)
      - test_split_Depression_AVEC2017.csv   (NO PHQ8_Score — labels hidden)

    Returns dict: {participant_id: {"phq8_score": int, "phq8_binary": int, "gender": int}}
    """
    root = Path(data_dir)
    labels = {}

    # Search for all possible label CSVs
    csv_patterns = [
        "train_split_Depression_AVEC2017.csv",
        "dev_split_Depression_AVEC2017.csv",
        "full_Depression_AVEC2017.csv",
        "labels.csv",
    ]

    for pattern in csv_patterns:
        # Search in root and one level down
        candidates = list(root.glob(pattern)) + list(root.glob(f"*/{pattern}"))
        for csv_path in candidates:
            print(f"  Found label file: {csv_path.name}")
            df = pd.read_csv(csv_path)

            # Normalize column names (AVEC2017 uses Participant_ID)
            col_map = {}
            for col in df.columns:
                low = col.lower().strip()
                if "participant" in low and "id" in low:
                    col_map[col] = "participant_id"
                elif low == "phq8_score":
                    col_map[col] = "phq8_score"
                elif low == "phq8_binary":
                    col_map[col] = "phq8_binary"
                elif low == "gender":
                    col_map[col] = "gender"
            df = df.rename(columns=col_map)

            if "participant_id" not in df.columns:
                print(f"    Skipping {csv_path.name} — no participant ID column found")
                continue

            if "phq8_score" not in df.columns:
                print(f"    {csv_path.name} has no PHQ8_Score (test set — labels hidden)")
                continue

            count = 0
            for _, row in df.iterrows():
                pid = int(row["participant_id"])
                score = row.get("phq8_score", None)
                if pd.isna(score):
                    continue
                score = int(score)
                labels[pid] = {
                    "phq8_score": score,
                    "phq8_binary": 1 if score >= 10 else 0,
                    "gender": int(row.get("gender", -1)) if not pd.isna(row.get("gender", None)) else -1,
                }
                count += 1
            print(f"    Loaded {count} participants with PHQ-8 scores")

    return labels


# ─────────────────────────────────────────────
# STEP 2: DISCOVER PARTICIPANT FOLDERS
# ─────────────────────────────────────────────
def discover_participants(data_dir: str) -> List[dict]:
    """
    Find all participant folders in flat or nested layout.
    Returns list of {participant_id, audio_path, transcript_path}.
    """
    root = Path(data_dir)
    participants = []

    # Find audio files in flat or nested layout
    audio_files = sorted(root.glob("*_AUDIO.wav"))
    if not audio_files:
        audio_files = sorted(root.glob("*/*_AUDIO.wav"))

    if not audio_files:
        print(f"ERROR: No *_AUDIO.wav files found under {root}")
        return participants

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
# STEP 3: EXTRACT PARTICIPANT-ONLY SPEECH
# ─────────────────────────────────────────────
def get_participant_speech(audio_path: str, transcript_path: Optional[str], sr: int) -> np.ndarray:
    """Load audio and extract only participant's speech using transcript timestamps."""
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav = wav.astype(np.float32)

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
        print(f"    Warning: transcript parsing failed ({ex}), using full audio")

    return wav


# ─────────────────────────────────────────────
# STEP 4: EXTRACT eGeMAPS PER PARTICIPANT
# ─────────────────────────────────────────────
def extract_egemaps_for_participant(wav: np.ndarray, sr: int) -> np.ndarray:
    """Extract 88-d eGeMAPSv02 feature vector from participant speech."""
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_signal(wav, sr)
        return df.values.flatten().astype(np.float32)
    except Exception as ex:
        print(f"    Warning: eGeMAPS extraction failed ({ex}), using zeros")
        return np.zeros(88, dtype=np.float32)


# ─────────────────────────────────────────────
# STEP 5: SPEAKER-LEVEL TRAIN/VAL SPLIT
# ─────────────────────────────────────────────
def split_speakers(
    participant_ids: List[int],
    labels: Dict[int, dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[int, str]:
    """
    Assign each speaker to 'train' or 'val', stratified by PHQ8_binary.
    Returns {participant_id: "train" | "val"}.
    """
    rng = np.random.RandomState(seed)

    # Separate by class
    positive = [pid for pid in participant_ids if labels.get(pid, {}).get("phq8_binary", 0) == 1]
    negative = [pid for pid in participant_ids if labels.get(pid, {}).get("phq8_binary", 0) == 0]

    rng.shuffle(positive)
    rng.shuffle(negative)

    n_val_pos = max(1, int(len(positive) * val_ratio)) if positive else 0
    n_val_neg = max(1, int(len(negative) * val_ratio)) if negative else 0

    val_speakers = set(positive[:n_val_pos] + negative[:n_val_neg])

    result = {}
    for pid in participant_ids:
        result[pid] = "val" if pid in val_speakers else "train"

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DAIC-WOZ Data Preparation — Notebook 1")
    print("=" * 60)

    data_dir = PrepConfig.DATA_DIR
    output_dir = PrepConfig.OUTPUT_DIR

    # Support local runs: check if Kaggle paths exist
    if not Path(data_dir).exists():
        # Fallback for local testing
        local_candidates = [
            "configs/dataset",
            "dataset",
            "data/raw/daic_woz",
        ]
        for candidate in local_candidates:
            if Path(candidate).exists():
                data_dir = candidate
                break

    if not Path(output_dir).exists():
        output_dir = "."

    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {output_dir}")
    print()

    # ── 1. Load labels ─────────────────────────────────────────────────
    print("--- Step 1: Loading AVEC2017 labels ---")
    labels = load_avec_labels(data_dir)
    print(f"  Total participants with PHQ-8 scores: {len(labels)}")

    if not labels:
        print("\nERROR: No PHQ-8 labels found!")
        print("Make sure train_split_Depression_AVEC2017.csv is in your dataset directory.")
        return

    scores = [v["phq8_score"] for v in labels.values()]
    print(f"  PHQ-8 range: {min(scores)} – {max(scores)}")
    print(f"  Binary split: {sum(1 for v in labels.values() if v['phq8_binary']==0)} non-depressed / "
          f"{sum(1 for v in labels.values() if v['phq8_binary']==1)} depressed")

    # ── 2. Discover participants ──────────────────────────────────────
    print("\n--- Step 2: Discovering participant folders ---")
    participants = discover_participants(data_dir)
    print(f"  Found {len(participants)} participant folder(s)")

    # Filter to only those with labels
    labeled_participants = [p for p in participants if p["participant_id"] in labels]
    unlabeled_participants = [p for p in participants if p["participant_id"] not in labels]

    print(f"  With PHQ-8 labels: {len(labeled_participants)}")
    if unlabeled_participants:
        pids = [p["participant_id"] for p in unlabeled_participants]
        print(f"  Without labels (skipped): {pids}")

    if not labeled_participants:
        print("\nERROR: No participant folders matched the label CSV!")
        print("Make sure folder names match (e.g., '303_P' for Participant_ID=303)")
        return

    # ── 3. Assign splits ──────────────────────────────────────────────
    print("\n--- Step 3: Assigning train/val splits ---")
    labeled_pids = [p["participant_id"] for p in labeled_participants]
    splits = split_speakers(labeled_pids, labels, PrepConfig.VAL_RATIO, PrepConfig.SEED)

    n_train = sum(1 for s in splits.values() if s == "train")
    n_val = sum(1 for s in splits.values() if s == "val")
    print(f"  Train speakers: {n_train}")
    print(f"  Val speakers:   {n_val}")

    # ── 4. Extract eGeMAPS + build labels.csv ─────────────────────────
    print("\n--- Step 4: Extracting eGeMAPS features ---")
    egemaps_dir = Path(output_dir) / "egemaps"
    egemaps_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in labeled_participants:
        pid = p["participant_id"]
        info = labels[pid]
        split = splits[pid]

        print(f"  Processing participant {pid} (PHQ={info['phq8_score']}, split={split})...")

        # Load participant speech
        wav = get_participant_speech(p["audio_path"], p["transcript_path"], PrepConfig.SR)
        duration = len(wav) / PrepConfig.SR
        print(f"    Speech duration: {duration:.1f}s")

        # Extract eGeMAPS
        feats = extract_egemaps_for_participant(wav, PrepConfig.SR)
        npy_path = egemaps_dir / f"{pid}_egemaps.npy"
        np.save(str(npy_path), feats)
        print(f"    eGeMAPS saved: {npy_path.name} (shape={feats.shape})")

        rows.append({
            "participant_id": pid,
            "phq8_score": info["phq8_score"],
            "phq8_binary": info["phq8_binary"],
            "gender": info["gender"],
            "speech_duration_sec": round(duration, 1),
            "egemaps_path": str(npy_path),
            "split": split,
        })

    # ── 5. Save labels.csv ────────────────────────────────────────────
    labels_df = pd.DataFrame(rows)
    labels_path = Path(output_dir) / "labels.csv"
    labels_df.to_csv(str(labels_path), index=False)

    print(f"\n--- Step 5: Saved labels.csv ---")
    print(f"  Path: {labels_path}")
    print(f"  Rows: {len(labels_df)}")
    print(f"\n  Preview:")
    print(labels_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print(f"  labels.csv:  {labels_path}")
    print(f"  eGeMAPS dir: {egemaps_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
