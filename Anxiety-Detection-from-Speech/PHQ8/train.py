"""
Local DAIC-WOZ — 25-Class PHQ-8 Classification Pipeline
Architecture: Frozen wav2vec2-base + eGeMAPSv02 → StandardScaler → PCA → SVM / XGBoost

Reads from your local dataset at configs/dataset/:
  configs/dataset/
    train_split_Depression_AVEC2017.csv   ← PHQ-8 labels (train)
    dev_split_Depression_AVEC2017.csv     ← PHQ-8 labels (dev / val)
    test_split_Depression_AVEC2017.csv    ← no labels (hidden)
    301_P/301_AUDIO.wav
    302_P/302_AUDIO.wav
    ...

Exact implementation of anxiety_detection_architecture.md:
  Stage 1  - Local AVEC2017 CSVs + participant folders
  Stage 2  - Frozen wav2vec2-base, Mean+Max pooling → 1536-d per participant
  Stage 3  - eGeMAPSv02 per 10s segment, mean-aggregated → 88-d per participant
  Stage 4  - Concatenate 1624-d, cache to hf_embeddings/, StandardScaler, PCA (95%)
  Stage 5  - SVM (RBF, class_weight=balanced) primary
              XGBoost (multi:softmax, num_class=25) secondary
  Stage 6  - weighted F1 + UAR on fixed official splits

Output:
  hf_embeddings/train_X.npy, train_y.npy
  hf_embeddings/dev_X.npy,   dev_y.npy
  scalers/scaler_hf.joblib
  scalers/pca_hf.joblib
  scalers/svm_phq_classifier.joblib
  scalers/xgb_phq_classifier.joblib   (if xgboost available)

Usage:
  python train_hf_classifier.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import joblib
from transformers import Wav2Vec2Model
import opensmile
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← the only section you may need to edit
# ─────────────────────────────────────────────────────────────────────────────
# Root folder containing participant sub-folders + AVEC CSV files
DATASET_DIR = Path("configs/dataset")

# Label CSV filenames (relative to DATASET_DIR)
TRAIN_CSV   = "train_split.csv"
DEV_CSV     = "dev_split.csv"

# Output directories (created automatically)
CACHE_DIR   = Path("hf_embeddings")
SCALERS_DIR = Path("scalers")

# ── Audio ─────────────────────────────────────────────────────────────────────
TARGET_SR   = 16_000       # wav2vec2-base input requirement
SEGMENT_SEC = 10.0         # 10-second windows (architecture §Stage 1)
OVERLAP_SEC = 2.0          # 2-second overlap
HOP_SEC     = SEGMENT_SEC - OVERLAP_SEC
MIN_SEC     = 3.0          # discard segments shorter than 3 s

# ── Model ────────────────────────────────────────────────────────────────────
WAV2VEC_MODEL   = "facebook/wav2vec2-base"
HIDDEN_SIZE     = 768
WAV2VEC_OUT_DIM = HIDDEN_SIZE * 2      # mean(768) + max(768) = 1536
EGEMAPS_DIM     = 88                   # eGeMAPSv02 Functionals
FUSED_DIM       = WAV2VEC_OUT_DIM + EGEMAPS_DIM  # 1624
NUM_CLASSES     = 25                   # PHQ-8 scores 0–24

BATCH_SIZE      = 4        # segments per inference batch (GPU memory safe)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# STAGE 1 — Load local AVEC2017 labels
def load_labels(csv_path: Path) -> Dict[int, int]:
    """
    Parse a CSV and return {participant_id: phq_score (int 0-24)}.
    Supports both old AVEC2017 column names and new PHQ_Score column names.
    Skips rows where score is missing.
    """
    if not csv_path.exists():
        print(f"  [WARN] Label file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find participant ID column
    id_col = next((c for c in df.columns if "participant" in c and "id" in c), None)
    if id_col is None:
        print(f"  [ERROR] Cannot find participant ID column in {csv_path.name}")
        return {}

    # Find PHQ score column — supports both 'PHQ_Score' and 'PHQ8_Score'
    score_col = next(
        (c for c in df.columns if c in ("phq_score", "phq8_score")),
        None
    )
    if score_col is None:
        print(f"  [SKIP] {csv_path.name} has no PHQ score column (test set — labels hidden)")
        return {}

    labels: Dict[int, int] = {}
    for _, row in df.iterrows():
        score = row.get(score_col, None)
        if pd.isna(score):
            continue
        pid = int(row[id_col])
        labels[pid] = int(float(score))

    print(f"  Loaded {len(labels)} participants from {csv_path.name}  "
          f"| PHQ range: [{min(labels.values())} – {max(labels.values())}]")
    return labels


def discover_participant(pid: int, dataset_dir: Path) -> Optional[Dict]:
    """
    Find audio for a participant.
    Layout 1 (flat, new):   All_participants/{PID}_AUDIO.wav
    Layout 2 (nested, old): {PID}_P/{PID}_AUDIO.wav
    Transcript CSV is optional — used when present to strip interviewer turns.
    """
    # Flat layout (new dataset structure)
    flat_audio = dataset_dir / "All_participants" / f"{pid}_AUDIO.wav"
    flat_tx    = dataset_dir / "full-extended-transcript" / f"{pid}_TRANSCRIPT.csv"
    if flat_audio.exists():
        return {
            "pid": pid,
            "audio": str(flat_audio),
            "transcript": str(flat_tx) if flat_tx.exists() else None,
        }

    # Nested fallback (legacy DAIC-WOZ layout: {PID}_P/{PID}_AUDIO.wav)
    nested_audio = dataset_dir / f"{pid}_P" / f"{pid}_AUDIO.wav"
    nested_tx    = dataset_dir / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
    if nested_audio.exists():
        return {
            "pid": pid,
            "audio": str(nested_audio),
            "transcript": str(nested_tx) if nested_tx.exists() else None,
        }

    return None  # audio not found locally


# AUDIO UTILITIES (§Stage 1)
def normalize(wav: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wav))
    return (wav / peak).astype(np.float32) if peak > 1e-6 else wav.astype(np.float32)


def apply_vad(wav: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Energy-based VAD — removes silence / non-speech."""
    intervals = librosa.effects.split(wav, top_db=top_db)
    if intervals.size == 0:
        return wav
    return np.concatenate([wav[s:e] for s, e in intervals]).astype(np.float32)


def get_participant_speech(wav: np.ndarray, transcript_path: Optional[str], sr: int) -> np.ndarray:
    """
    Use TRANSCRIPT.csv to strip out interviewer (Ellie) turns.
    Falls back to full audio when transcript is missing or unreadable.
    """
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
        print(f"    [WARN] Transcript parse failed ({ex}), using full audio")
    return wav


def make_segments(wav: np.ndarray, sr: int) -> List[np.ndarray]:
    """
    Slice waveform into 10-second windows with 2-second overlap.
    Short final chunks are zero-padded to full segment length.
    """
    seg_samples = int(SEGMENT_SEC * sr)
    hop_samples = int(HOP_SEC * sr)
    min_samples = int(MIN_SEC * sr)

    if len(wav) <= seg_samples:
        pad = np.zeros(seg_samples, dtype=np.float32)
        pad[:len(wav)] = wav
        return [pad]

    segments: List[np.ndarray] = []
    start = 0
    while start < len(wav):
        chunk = wav[start : start + seg_samples]
        if len(chunk) >= min_samples:
            pad = np.zeros(seg_samples, dtype=np.float32)
            pad[:len(chunk)] = chunk
            segments.append(pad)
        start += hop_samples

    if not segments:
        pad = np.zeros(seg_samples, dtype=np.float32)
        pad[:min(len(wav), seg_samples)] = wav[:seg_samples]
        segments = [pad]

    return segments


# STAGE 2 — Frozen wav2vec2-base extractor
class FrozenWav2VecExtractor(nn.Module):
    """
    ALL 95M parameters frozen (CNN encoder + 12 Transformer blocks).
    Per-segment output: concat(MeanPool, MaxPool) → (1536,)
    Participant-level: element-wise mean across N segments → (1536,)
    """
    def __init__(self):
        super().__init__()
        print(f"  Loading {WAV2VEC_MODEL} … (first run downloads ~360 MB)")
        self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav  : (B, T) float32 waveform at 16 kHz
            mask : (B, T) long attention mask (1=speech, 0=padding)
        Returns:
            (B, 1536) concat[mean_pool, max_pool] over time
        """
        out = self.backbone(
            input_values=wav,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # hidden_states[0] = CNN output; [1:13] = 12 Transformer layers
        transformer_states = out.hidden_states[1:]          # 12 × (B, T', 768)
        stacked    = torch.stack(transformer_states, dim=0) # (12, B, T', 768)
        aggregated = stacked.mean(dim=0)                    # (B, T', 768)

        mean_pooled = aggregated.mean(dim=1)                # (B, 768)
        max_pooled  = aggregated.max(dim=1).values          # (B, 768)

        return torch.cat([mean_pooled, max_pooled], dim=-1) # (B, 1536)


def extract_wav2vec_participant(
    segments: List[np.ndarray],
    model: FrozenWav2VecExtractor,
) -> np.ndarray:
    """Process segments in mini-batches → mean across segments → (1536,)."""
    all_embs: List[np.ndarray] = []

    for b_start in range(0, len(segments), BATCH_SIZE):
        batch    = segments[b_start : b_start + BATCH_SIZE]
        max_len  = max(len(w) for w in batch)

        padded, masks = [], []
        for w in batch:
            buf = np.zeros(max_len, dtype=np.float32)
            buf[:len(w)] = w
            padded.append(buf)
            m = np.zeros(max_len, dtype=np.int64)
            m[:len(w)] = 1
            masks.append(m)

        wav_t  = torch.tensor(np.array(padded), dtype=torch.float32).to(DEVICE)
        mask_t = torch.tensor(np.array(masks),  dtype=torch.long   ).to(DEVICE)

        emb = model(wav_t, mask_t)           # (B, 1536)
        all_embs.append(emb.cpu().numpy())

        del wav_t, mask_t
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    stacked = np.concatenate(all_embs, axis=0)  # (N_segs, 1536)
    return stacked.mean(axis=0)                  # (1536,) participant-level


# STAGE 3 — eGeMAPSv02 features
def extract_egemaps_per_segments(
    segments: List[np.ndarray], sr: int
) -> np.ndarray:
    """
    Per-segment eGeMAPSv02 extraction → element-wise mean → (88,)
    Returns zeros if opensmile is not installed.
    """

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    seg_feats: List[np.ndarray] = []
    for seg_wav in segments:
        try:
            df = smile.process_signal(seg_wav, sr)
            seg_feats.append(df.values.flatten().astype(np.float32))
        except Exception as e:
            seg_feats.append(np.zeros(EGEMAPS_DIM, dtype=np.float32))

    if not seg_feats:
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)

    return np.stack(seg_feats, axis=0).mean(axis=0)   # (88,)


# STAGE 1-3 combined — process one participant
def process_participant(
    info: Dict,
    phq_score: int,
    model: FrozenWav2VecExtractor,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Returns (fused_feature_vector [1624], phq_score) for one participant,
    or None if audio cannot be loaded.
    """
    pid = info["pid"]
    try:
        # ── Load audio ────────────────────────────────────────────────────
        wav, sr = librosa.load(info["audio"], sr=TARGET_SR, mono=True)
        wav = wav.astype(np.float32)

        # ── Resample if needed ────────────────────────────────────────────
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        # ── Participant speech only (strips Ellie's turns) ────────────────
        wav = get_participant_speech(wav, info["transcript"], TARGET_SR)

        # ── Normalize + VAD
        wav = normalize(wav)
        wav = apply_vad(wav, TARGET_SR)
        duration = len(wav) / TARGET_SR

        #Segment into 10s windows with 2s overlap 
        segments = make_segments(wav, TARGET_SR)
        print(f"    {pid}: {duration:.1f}s → {len(segments)} segments | PHQ={phq_score}")

        #Stage 2: Frozen wav2vec2-base
        wav2vec_vec = extract_wav2vec_participant(segments, model)   # (1536,)

        #Stage 3: eGeMAPSv02
        egemaps_vec = extract_egemaps_per_segments(segments, TARGET_SR)  # (88,)

        #Stage 4: Concatenate
        fused = np.concatenate([wav2vec_vec, egemaps_vec])              # (1624,)

        del wav, segments
        gc.collect()

        return fused, phq_score

    except Exception as e:
        print(f"    [ERROR] Participant {pid}: {e}")
        return None


# STAGE 4 — Feature extraction with disk caching
def load_or_extract_split(
    split_name: str,
    labels: Dict[int, int],
    model: FrozenWav2VecExtractor,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract or load cached (X, y) arrays for one split.
    X: (N, 1624), y: (N,) PHQ-8 integer labels 0-24
    """
    cache_X = CACHE_DIR / f"{split_name}_X.npy"
    cache_y = CACHE_DIR / f"{split_name}_y.npy"

    if not force_recompute and cache_X.exists() and cache_y.exists():
        X = np.load(str(cache_X))
        y = np.load(str(cache_y))
        print(f"  [Cache] {split_name}: loaded X={X.shape}, y={y.shape}")
        print(f"          PHQ range [{int(y.min())}–{int(y.max())}] | "
              f"{len(np.unique(y))} distinct classes")
        return X, y

    print(f"\n  Extracting features for '{split_name}' split "
          f"({len(labels)} participants in labels) ...")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    missing = []

    for pid, phq in sorted(labels.items()):
        info = discover_participant(pid, DATASET_DIR)
        if info is None:
            missing.append(pid)
            continue

        result = process_participant(info, phq, model)
        if result is not None:
            fused, score = result
            X_list.append(fused)
            y_list.append(score)

    if missing:
        print(f"[WARN] {len(missing)} participants in labels but no audio found locally: "
         f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    if not X_list:
        print(f"[ERROR] No features extracted for split '{split_name}'")
        return np.empty((0, FUSED_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(str(cache_X), X)
    np.save(str(cache_y), y)
    print(f"[Cache] Saved {split_name}: X={X.shape}, y={y.shape}")

    return X, y


# STAGE 6 — Evaluation
def evaluate_split(
    clf,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
) -> None:
    """Print classification report + Weighted F1 + UAR + MAE (§Stage 6)."""
    if len(X) == 0:
        print(f"  [SKIP] {split_name}: no samples to evaluate")
        return

    y_pred = clf.predict(X)
    # Ensure predictions are integers (XGBoost returns float)
    y_pred = np.array(y_pred, dtype=np.int64)

    print(f"{split_name.upper()} RESULTS  (PHQ-8 Score — classes 0-24)")
    print(classification_report(y_true, y_pred, zero_division=0))

    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uar   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mae   = float(np.mean(np.abs(y_pred.astype(float) - y_true.astype(float))))

    print(f"  Weighted F1         : {w_f1:.4f}")
    print(f"  UAR (Macro Recall)  : {uar:.4f}")
    print(f"  MAE (PHQ integer)   : {mae:.2f}")

    # Confusion matrix when class count is manageable
    unique = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique) <= 15:
        cm = confusion_matrix(y_true, y_pred, labels=unique)
        print(f"\n Confusion matrix (classes {unique.tolist()}):")
        print(cm)


# STAGE 4-5-6 — Preprocess + train classifiers + evaluate
def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev:   np.ndarray, y_dev:   np.ndarray,
) -> None:
    """
    Stage 4 (cont.): StandardScaler → PCA (95%)
    Stage 5 (primary):   SVM RBF, class_weight='balanced'
    Stage 5 (secondary): XGBoost multi:softmax, num_class=25
    Stage 6: Evaluate on train + dev
    """
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"STAGE 4 — Scaling & Dimensionality Reduction")
    print(f"Train : {len(X_train)} samples | feature dim : {X_train.shape[1]}")
    print(f"Dev   : {len(X_dev)} samples")
    print(f"Train PHQ classes seen : {np.unique(y_train).tolist()}")

    # StandardScaler (fit on train only) 
    print("\n  Fitting StandardScaler …")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)   if len(X_dev)   > 0 else X_dev
    joblib.dump(scaler, SCALERS_DIR / "scaler_hf.joblib")
    print("  Saved → scalers/scaler_hf.joblib")

    # PCA (95% variance)
    print("\n  Fitting PCA (95% variance) …")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    n_comp = X_train_p.shape[1]
    print(f"  Reduced {X_train_s.shape[1]} → {n_comp} components (95% variance retained)")

    X_dev_p = pca.transform(X_dev_s) if len(X_dev_s) > 0 else X_dev_s
    joblib.dump(pca, SCALERS_DIR / "pca_hf.joblib")
    print("  Saved → scalers/pca_hf.joblib")

    # STAGE 5 (Primary): SVM 

    print(f"STAGE 5 (Primary) — SVM · RBF Kernel")
    print("C=1.0 · gamma='scale' · class_weight='balanced' · probability=True")
    print("Training … (on small N this is fast; on 100+ participants ~1–3 min)")

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    svm.fit(X_train_p, y_train)
    joblib.dump(svm, SCALERS_DIR / "svm_phq_classifier.joblib")
    print("Saved → scalers/svm_phq_classifier.joblib")

    print("\nSVM Results")
    evaluate_split(svm, X_train_p, y_train, "train (SVM)")
    if len(X_dev_p) > 0:
        evaluate_split(svm, X_dev_p, y_dev, "dev (SVM)")

    # STAGE 5 (Secondary): XGBoost
    print(f"STAGE 5 (Secondary) — XGBoost")
    try:
        print(f"XGBoost {xgb.__version__} | objective=multi:softmax | num_class={NUM_CLASSES}")

        xgb_clf = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=NUM_CLASSES,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            verbosity=1,
            random_state=42,
            n_jobs=-1,
        )

        eval_sets = [(X_train_p, y_train)]
        if len(X_dev_p) > 0:
            eval_sets.append((X_dev_p, y_dev))

        xgb_clf.fit(X_train_p, y_train, eval_set=eval_sets, verbose=50)
        joblib.dump(xgb_clf, SCALERS_DIR / "xgb_phq_classifier.joblib")
        print("  Saved → scalers/xgb_phq_classifier.joblib")

        print("\n  === XGBoost Results ===")
        evaluate_split(xgb_clf, X_train_p, y_train, "train (XGBoost)")
        if len(X_dev_p) > 0:
            evaluate_split(xgb_clf, X_dev_p, y_dev, "dev (XGBoost)")

    except ImportError:
        print("  [SKIP] XGBoost not installed. Run: pip install xgboost")
    except Exception as e:
        print(f"  [ERROR] XGBoost failed: {e}")


# INFERENCE — predict PHQ-8 integer score from a wav file
def predict_phq_score(wav_path: str, classifier: str = "svm") -> int:
    """
    Predict PHQ-8 integer score (0-24) for a new audio file.

    Args:
        wav_path:    Path to a .wav file
        classifier:  'svm' (default) or 'xgb'

    Returns:
        int in [0, 24]

    Example:
        from train_hf_classifier import predict_phq_score
        score = predict_phq_score("audio_files/inference1.wav")
        print(f"Predicted PHQ-8 score: {score}")
    """
    scaler = joblib.load(SCALERS_DIR / "scaler_hf.joblib")
    pca    = joblib.load(SCALERS_DIR / "pca_hf.joblib")
    clf_name = "svm_phq_classifier" if classifier == "svm" else "xgb_phq_classifier"
    clf    = joblib.load(SCALERS_DIR / f"{clf_name}.joblib")

    model = FrozenWav2VecExtractor().to(DEVICE).eval()

    wav, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    wav = normalize(wav.astype(np.float32))
    wav = apply_vad(wav, TARGET_SR)
    segments = make_segments(wav, TARGET_SR)

    wav2vec_vec = extract_wav2vec_participant(segments, model)
    egemaps_vec = extract_egemaps_per_segments(segments, TARGET_SR)
    fused = np.concatenate([wav2vec_vec, egemaps_vec]).reshape(1, -1)

    fused_s    = scaler.transform(fused)
    fused_p    = pca.transform(fused_s)
    phq_pred   = int(clf.predict(fused_p)[0])

    del model
    gc.collect()

    return phq_pred


# MAIN
def main():
    print("Anxiety Detection — 25-Class PHQ-8 (Local DAIC-WOZ)")
    print("Architecture: Frozen wav2vec2-base + eGeMAPSv02 → SVM")
    print(f"Device     : {DEVICE}")
    print(f"Dataset    : {DATASET_DIR.resolve()}")
    print(f"Cache dir  : {CACHE_DIR.resolve()}")
    print(f"Models dir : {SCALERS_DIR.resolve()}")
    print()

    #Stage 1: Load labels from AVEC2017 CSVs
    print("Stage 1: Loading AVEC2017 labels")
    train_labels = load_labels(DATASET_DIR / TRAIN_CSV)
    dev_labels   = load_labels(DATASET_DIR / DEV_CSV)

    if not train_labels:
        print("\n[FATAL] No training labels found. "
              f"Expected: {DATASET_DIR / TRAIN_CSV}")
        return

    print(f"\n Train participants with labels : {len(train_labels)}")
    print(f"Dev participants with labels : {len(dev_labels)}")

    #Stage 2-3: Feature extraction
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if all caches exist before loading the heavy model
    all_cached = (
        (CACHE_DIR / "train_X.npy").exists() and
        (CACHE_DIR / "train_y.npy").exists() and
        (CACHE_DIR / "dev_X.npy").exists() and
        (CACHE_DIR / "dev_y.npy").exists()
    )

    if all_cached:
        print("\nStage 2-3: Loading cached features (skipping extraction)")
        X_train = np.load(str(CACHE_DIR / "train_X.npy"))
        y_train = np.load(str(CACHE_DIR / "train_y.npy"))
        X_dev   = np.load(str(CACHE_DIR / "dev_X.npy"))
        y_dev   = np.load(str(CACHE_DIR / "dev_y.npy"))
        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Dev  : X={X_dev.shape}, y={y_dev.shape}")
    else:
        print("\nStage 2-3: Extracting features (Frozen wav2vec2-base + eGeMAPS)")
        model = FrozenWav2VecExtractor().to(DEVICE).eval()

        X_train, y_train = load_or_extract_split("train", train_labels, model)
        X_dev,   y_dev   = load_or_extract_split("dev",   dev_labels,   model)

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("Extraction Summary")
    for name, X, y in [("train", X_train, y_train), ("dev", X_dev, y_dev)]:
        if len(X) > 0:
            unique, counts = np.unique(y, return_counts=True)
            print(f"{name:6s}: {len(X)} participants | " f"PHQ [{int(y.min())}-{int(y.max())}] | " f"{len(unique)} distinct classes")
        else:
            print(f"{name:6s}: 0 participants (no audio found)")

    if len(X_train) == 0:
        print("\n[FATAL] No training features. Check that audio files exist under " f"{DATASET_DIR.resolve()}")
        return

    # Stage 4-5-6: Scale → PCA → Train classifiers → Evaluate
    train_and_evaluate(X_train, y_train, X_dev, y_dev)

    print("PIPELINE COMPLETE")
    print("Saved artefacts:")
    for f in ["scaler_hf.joblib", "pca_hf.joblib", "svm_phq_classifier.joblib", "xgb_phq_classifier.joblib"]:
        p = SCALERS_DIR / f
        if p.exists():
            print(f"{p}")

if __name__ == "__main__":
    main()
