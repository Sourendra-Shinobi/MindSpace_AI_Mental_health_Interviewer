"""
Audio preprocessing utilities for the Speech Anxiety Detection pipeline.

Handles: loading, resampling, normalization, and silence trimming.
All audio is converted to 16kHz mono float32 for Wav2Vec 2.0 compatibility.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional


def load_audio(
    path: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        path: Path to audio file (WAV, MP3, FLAC, OGG).
        sr: Target sample rate. Wav2Vec 2.0 requires 16000.
        mono: If True, convert stereo to mono by averaging channels.

    Returns:
        Tuple of (waveform as float32 numpy array, sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa handles format conversion and resampling internally
    waveform, sample_rate = librosa.load(str(path), sr=sr, mono=mono)

    # Ensure float32
    waveform = waveform.astype(np.float32)

    return waveform, sample_rate


def normalize_audio(
    waveform: np.ndarray,
    method: str = "peak",
    target_rms: float = 0.1,
) -> np.ndarray:
    """
    Normalize audio amplitude to prevent loudness variation from
    confusing the model.

    Args:
        waveform: Audio signal as float32 numpy array.
        method: "peak" for peak normalization (max=1.0),
                "rms" for RMS normalization to target level.
        target_rms: Target RMS level (only used if method="rms").

    Returns:
        Normalized waveform.
    """
    if len(waveform) == 0:
        return waveform

    if method == "peak":
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
    elif method == "rms":
        current_rms = np.sqrt(np.mean(waveform ** 2))
        if current_rms > 0:
            waveform = waveform * (target_rms / current_rms)
        # Clip to prevent clipping after RMS normalization
        waveform = np.clip(waveform, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'peak' or 'rms'.")

    return waveform.astype(np.float32)


def trim_silence(
    waveform: np.ndarray,
    top_db: float = 20.0,
    sr: int = 16000,
) -> np.ndarray:
    """
    Trim leading and trailing silence from the waveform.

    Args:
        waveform: Audio signal as float32 numpy array.
        top_db: Threshold (in dB) below reference to consider as silence.
        sr: Sample rate (used internally by librosa).

    Returns:
        Trimmed waveform.
    """
    if len(waveform) == 0:
        return waveform

    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def preprocess_audio(
    path: Union[str, Path],
    sr: int = 16000,
    normalize_method: str = "peak",
    trim_db: float = 20.0,
) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline: load → normalize → trim.

    Args:
        path: Path to audio file.
        sr: Target sample rate.
        normalize_method: Normalization method ("peak" or "rms").
        trim_db: Silence trimming threshold in dB.

    Returns:
        Tuple of (preprocessed waveform, sample_rate).
    """
    waveform, sr = load_audio(path, sr=sr)
    waveform = normalize_audio(waveform, method=normalize_method)
    waveform = trim_silence(waveform, top_db=trim_db, sr=sr)
    return waveform, sr


def save_audio(
    waveform: np.ndarray,
    path: Union[str, Path],
    sr: int = 16000,
) -> None:
    """Save waveform to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr)
