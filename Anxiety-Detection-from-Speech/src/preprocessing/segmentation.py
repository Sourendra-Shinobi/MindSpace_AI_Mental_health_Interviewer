"""
Audio segmentation for the Speech Anxiety Detection pipeline.

Handles windowed segmentation of long recordings (DAIC-WOZ interviews)
and zero-padding of short clips for Wav2Vec 2.0 input.
"""

import numpy as np
from typing import List, Tuple


def segment_audio(
    waveform: np.ndarray,
    sr: int = 16000,
    window_sec: float = 10.0,
    hop_sec: float = 5.0,
    min_segment_sec: float = 3.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split long audio into overlapping fixed-length windows.

    For DAIC-WOZ interviews (7–33 min), this creates 10-second windows
    with 50% overlap. All segments from the same speaker carry the
    same anxiety label.

    Args:
        waveform: Audio signal (mono, 16kHz, float32).
        sr: Sample rate.
        window_sec: Window length in seconds.
        hop_sec: Hop length in seconds (window_sec/2 for 50% overlap).
        min_segment_sec: Minimum segment length to keep.

    Returns:
        List of (segment_waveform, start_time_sec) tuples.
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    min_samples = int(min_segment_sec * sr)

    total_samples = len(waveform)

    # If audio is shorter than window, return as single segment
    if total_samples <= window_samples:
        return [(waveform, 0.0)]

    segments = []
    start = 0

    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment = waveform[start:end]

        # Only keep segments above minimum length
        if len(segment) >= min_samples:
            start_time = start / sr
            segments.append((segment, start_time))

        start += hop_samples

    return segments


def pad_audio(
    waveform: np.ndarray,
    sr: int = 16000,
    min_sec: float = 3.0,
) -> np.ndarray:
    """
    Zero-pad short audio clips to minimum length.

    Wav2Vec 2.0's CNN encoder has a receptive field that requires
    a minimum input length. Clips shorter than min_sec are padded.

    Args:
        waveform: Audio signal (mono, 16kHz, float32).
        sr: Sample rate.
        min_sec: Minimum audio duration in seconds.

    Returns:
        Padded waveform (or original if already long enough).
    """
    min_samples = int(min_sec * sr)
    current_samples = len(waveform)

    if current_samples >= min_samples:
        return waveform

    # Zero-pad at the end
    padding = np.zeros(min_samples - current_samples, dtype=np.float32)
    return np.concatenate([waveform, padding])


def segment_or_pad(
    waveform: np.ndarray,
    sr: int = 16000,
    window_sec: float = 10.0,
    hop_sec: float = 5.0,
    min_sec: float = 3.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Combined segmentation and padding logic.

    - Long audio (> window_sec): segment with overlap
    - Short audio (< min_sec): pad to min_sec
    - Medium audio: return as single segment

    Args:
        waveform: Audio signal (mono, 16kHz, float32).
        sr: Sample rate.
        window_sec: Window length in seconds for segmentation.
        hop_sec: Hop length for overlap.
        min_sec: Minimum duration (pad if shorter).

    Returns:
        List of (segment_waveform, start_time_sec) tuples.
    """
    duration = len(waveform) / sr

    if duration > window_sec:
        return segment_audio(waveform, sr, window_sec, hop_sec, min_sec)
    else:
        padded = pad_audio(waveform, sr, min_sec)
        return [(padded, 0.0)]
