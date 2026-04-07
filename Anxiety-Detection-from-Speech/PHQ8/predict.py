"""
PHQ-8 Score Inference
=====================
Predicts an integer PHQ-8 score (0-24) from a speech audio file using the
trained artifacts produced by train_hf_classifier.py.

Pipeline (mirrors training exactly):
  Audio → Resample → Normalize → VAD → 10s segments (2s overlap)
        → Frozen wav2vec2-base  → mean+max pool → mean over segments → (1536,)
        → eGeMAPSv02 per segment → mean over segments               → (88,)
        → Concatenate                                                 → (1624,)
        → StandardScaler + PCA                                        → (~110,)
        → SVM / XGBoost                                               → PHQ int

Usage (command line):
    python predict_phq.py audio_files/inference1.wav
    python predict_phq.py audio_files/inference1.wav --classifier xgb
    python predict_phq.py audio_files/inference1.wav --classifier svm --verbose

Usage (Python API):
    from predict_phq import predict
    result = predict("path/to/audio.wav")
    print(result["phq_score"])    # integer 0-24
    print(result["severity"])     # "Minimal" / "Mild" / "Moderate" / "Severe"
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib

warnings.filterwarnings("ignore")

SCALERS_DIR   = Path("scalers")
WAV2VEC_MODEL = "facebook/wav2vec2-base"

# AUDIO CONSTANTS
TARGET_SR   = 16_000
SEGMENT_SEC = 10.0
OVERLAP_SEC = 2.0
HOP_SEC     = SEGMENT_SEC - OVERLAP_SEC
MIN_SEC     = 3.0
EGEMAPS_DIM = 88
BATCH_SIZE  = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PHQ-8 severity bands (clinical standard)
def phq_severity(score: int) -> str:
    if score <= 4:
        return "Minimal / None"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"


# AUDIO UTILITIES (identical to training)
def normalize(wav: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wav))
    return (wav / peak).astype(np.float32) if peak > 1e-6 else wav.astype(np.float32)


def apply_vad(wav: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    intervals = librosa.effects.split(wav, top_db=top_db)
    if intervals.size == 0:
        return wav
    return np.concatenate([wav[s:e] for s, e in intervals]).astype(np.float32)


def make_segments(wav: np.ndarray, sr: int) -> List[np.ndarray]:
    seg_samples = int(SEGMENT_SEC * sr)
    hop_samples = int(HOP_SEC * sr)
    min_samples = int(MIN_SEC * sr)

    if len(wav) <= seg_samples:
        pad = np.zeros(seg_samples, dtype=np.float32)
        pad[: len(wav)] = wav
        return [pad]

    segments: List[np.ndarray] = []
    start = 0
    while start < len(wav):
        chunk = wav[start : start + seg_samples]
        if len(chunk) >= min_samples:
            pad = np.zeros(seg_samples, dtype=np.float32)
            pad[: len(chunk)] = chunk
            segments.append(pad)
        start += hop_samples

    if not segments:
        pad = np.zeros(seg_samples, dtype=np.float32)
        pad[: min(len(wav), seg_samples)] = wav[:seg_samples]
        segments = [pad]

    return segments


# FROZEN WAV2VEC2-BASE
class FrozenWav2VecExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_values=wav,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        transformer_states = out.hidden_states[1:]           # 12 Transformer layers
        stacked    = torch.stack(transformer_states, dim=0)  # (12, B, T', 768)
        aggregated = stacked.mean(dim=0)                     # (B, T', 768)
        mean_pooled = aggregated.mean(dim=1)                 # (B, 768)
        max_pooled  = aggregated.max(dim=1).values           # (B, 768)
        return torch.cat([mean_pooled, max_pooled], dim=-1)  # (B, 1536)


def extract_wav2vec(segments: List[np.ndarray], model: FrozenWav2VecExtractor) -> np.ndarray:
    all_embs: List[np.ndarray] = []
    for b_start in range(0, len(segments), BATCH_SIZE):
        batch   = segments[b_start : b_start + BATCH_SIZE]
        max_len = max(len(w) for w in batch)
        padded, masks = [], []
        for w in batch:
            buf = np.zeros(max_len, dtype=np.float32); buf[: len(w)] = w
            m   = np.zeros(max_len, dtype=np.int64);   m[: len(w)]   = 1
            padded.append(buf); masks.append(m)
        wav_t  = torch.tensor(np.array(padded), dtype=torch.float32).to(DEVICE)
        mask_t = torch.tensor(np.array(masks),  dtype=torch.long   ).to(DEVICE)
        emb = model(wav_t, mask_t)
        all_embs.append(emb.cpu().numpy())
        del wav_t, mask_t
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0).mean(axis=0)   # (1536,)


def extract_egemaps(segments: List[np.ndarray], sr: int) -> np.ndarray:
    try:
        import opensmile
    except ImportError:
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feats: List[np.ndarray] = []
    for seg in segments:
        try:
            df = smile.process_signal(seg, sr)
            feats.append(df.values.flatten().astype(np.float32))
        except Exception:
            feats.append(np.zeros(EGEMAPS_DIM, dtype=np.float32))

    return np.stack(feats, axis=0).mean(axis=0) if feats else np.zeros(EGEMAPS_DIM, dtype=np.float32)


def _check_artefacts(classifier: str) -> None:
    clf_file = SCALERS_DIR / f"{'svm' if classifier == 'svm' else 'xgb'}_phq_classifier.joblib"
    missing = []
    for f in [SCALERS_DIR / "scaler_hf.joblib", SCALERS_DIR / "pca_hf.joblib", clf_file]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("ERROR: The following model files are missing:")
        for m in missing:
            print(f"  {m}")
        print("\nPlease run training first:")
        print("  python train_hf_classifier.py")
        sys.exit(1)

def predict(
    wav_path: str,
    classifier: str = "svm",
    verbose: bool = False,
) -> Dict:
    """
    Predict PHQ-8 score from a speech audio file.

    Args:
        wav_path:    Path to .wav (or any librosa-readable) audio file.
        classifier:  'svm' (default) or 'xgb'
        verbose:     Print step-by-step progress.

    Returns:
        dict with keys:
            phq_score   (int)  — predicted PHQ-8 integer score, 0–24
            severity    (str)  — clinical severity band
            confidence  (float | None) — SVM probability for predicted class
            duration_s  (float) — speech duration after VAD (seconds)
            n_segments  (int)  — number of 10s windows processed
            classifier  (str)  — which classifier was used
    """
    _check_artefacts(classifier)

    t0 = time.time()

    # ── Load saved artefacts ──────────────────────────────────────────────
    scaler   = joblib.load(SCALERS_DIR / "scaler_hf.joblib")
    pca      = joblib.load(SCALERS_DIR / "pca_hf.joblib")
    clf_name = "svm_phq_classifier" if classifier == "svm" else "xgb_phq_classifier"
    clf      = joblib.load(SCALERS_DIR / f"{clf_name}.joblib")

    if verbose:
        print(f"[1/5] Artefacts loaded  ({classifier.upper()} classifier)")

    # ── Stage 1: Audio preprocessing ─────────────────────────────────────
    wav, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    wav = normalize(wav.astype(np.float32))
    raw_dur = len(wav) / TARGET_SR

    wav = apply_vad(wav, TARGET_SR)
    speech_dur = len(wav) / TARGET_SR
    segments   = make_segments(wav, TARGET_SR)

    if verbose:
        print(f"[2/5] Audio loaded      raw={raw_dur:.1f}s | "
              f"after VAD={speech_dur:.1f}s | {len(segments)} segments")

    # ── Stage 2: Frozen wav2vec2-base ────────────────────────────────────
    if verbose:
        print(f"[3/5] Running frozen wav2vec2-base on {DEVICE} …")

    model = FrozenWav2VecExtractor().to(DEVICE).eval()
    wav2vec_vec = extract_wav2vec(segments, model)          # (1536,)
    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print(f"wav2vec feature: shape={wav2vec_vec.shape}")

    # ── Stage 3: eGeMAPSv02 ──────────────────────────────────────────────
    if verbose:
        print("[4/5] Extracting eGeMAPSv02 features …")

    egemaps_vec = extract_egemaps(segments, TARGET_SR)      # (88,)

    if verbose:
        print(f"      eGeMAPS feature: shape={egemaps_vec.shape}")

    # ── Stage 4: Fuse → Scale → PCA ──────────────────────────────────────
    fused   = np.concatenate([wav2vec_vec, egemaps_vec]).reshape(1, -1)  # (1, 1624)
    fused_s = scaler.transform(fused)
    fused_p = pca.transform(fused_s)

    if verbose:
        print(f"[5/5] PCA reduced: {fused.shape[1]} → {fused_p.shape[1]} dims")

    # ── Stage 5: Predict ─────────────────────────────────────────────────
    phq_score  = int(clf.predict(fused_p)[0])
    confidence = None
    if hasattr(clf, "predict_proba"):
        proba      = clf.predict_proba(fused_p)[0]
        classes    = clf.classes_
        idx        = np.where(classes == phq_score)[0]
        confidence = float(proba[idx[0]]) if len(idx) > 0 else None

    elapsed = time.time() - t0

    return {
        "phq_score":  phq_score,
        "severity":   phq_severity(phq_score),
        "confidence": confidence,
        "duration_s": round(speech_dur, 2),
        "n_segments": len(segments),
        "classifier": classifier,
        "elapsed_s":  round(elapsed, 2),
    }

def _print_result(result: Dict, wav_path: str) -> None:
    score = result["phq_score"]
    sev   = result["severity"]
    conf  = result["confidence"]

    # Severity bar (visual)
    bar_len   = 40
    filled    = int((score / 24) * bar_len)
    #bar       = "█" * filled + "░" * (bar_len - filled)
    #conf_str  = f"  (confidence: {conf:.1%})" if conf is not None else ""

    print("PHQ-8 Score Prediction Result")
    print(f"File       : {Path(wav_path).name:<35s}")
    print(f"PHQ-8 Score: {score:<35d}")
    print(f"Severity   : {sev:<35s}")
    print(f"Classifier : {result['classifier'].upper():<35s}")
    print(f"Duration   : {result['duration_s']:.1f}s speech | {result['n_segments']} segments{'':<12s}")
    print(f"Elapsed    : {result['elapsed_s']:.1f}s{'':<38s}")
    #print(f"[{bar}]  {score}/24 ")
    if conf is not None:
        print(f"Confidence : {conf:.1%}{'':<35s}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict PHQ-8 integer score (0-24) from a speech audio file."
    )
    parser.add_argument(
        "audio",
        help="Path to the input audio file (.wav recommended)",
    )
    parser.add_argument(
        "--classifier",
        choices=["svm", "xgb"],
        default="svm",
        help="Which trained classifier to use (default: svm)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print step-by-step processing details",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (for programmatic use)",
    )
    args = parser.parse_args()

    wav_path = args.audio
    if not Path(wav_path).exists():
        print(f"ERROR: Audio file not found: {wav_path}")
        sys.exit(1)

    print(f"Predicting PHQ-8 score for: {wav_path}")
    print(f"Using classifier: {args.classifier.upper()} | Device: {DEVICE}")

    result = predict(wav_path, classifier=args.classifier, verbose=args.verbose)

    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        _print_result(result, wav_path)


if __name__ == "__main__":
    main()
