"""
Voice Activity Detection (VAD) using Silero VAD.

Removes non-speech segments from audio. Critical for DAIC-WOZ
where interviewer (Ellie) speech must be excluded.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import warnings


class SileroVAD:
    """
    Silero VAD wrapper for detecting and extracting speech segments.

    The model is loaded once and reused for all files. Works on CPU
    and is highly optimized for single-threaded inference.
    """

    def __init__(self, sampling_rate: int = 16000):
        """
        Initialize Silero VAD model.

        Args:
            sampling_rate: Audio sample rate (8000 or 16000 Hz supported).
        """
        if sampling_rate not in (8000, 16000):
            raise ValueError(
                f"Silero VAD supports 8000 or 16000 Hz, got {sampling_rate}"
            )
        self.sampling_rate = sampling_rate

        # Load model — try pip package first, fall back to torch.hub
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
            self.model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._read_audio = read_audio
        except ImportError:
            try:
                self.model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._get_speech_timestamps = utils[0]
                self._read_audio = utils[2]
            except Exception as e:
                raise RuntimeError(
                    "Failed to load Silero VAD. Install via: "
                    "pip install silero-vad  OR  ensure torch.hub access.\n"
                    f"Error: {e}"
                )

        # Set optimal CPU threading
        torch.set_num_threads(1)

    def get_speech_segments(
        self,
        waveform: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Tuple[int, int]]:
        """
        Detect speech segments in the waveform.

        Args:
            waveform: Audio signal as float32 numpy array (mono, 16kHz).
            threshold: Speech detection threshold (0–1).
            min_speech_duration_ms: Minimum speech segment duration.
            min_silence_duration_ms: Minimum silence between speech segments.

        Returns:
            List of (start_sample, end_sample) tuples.
        """
        # Convert to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            wav_tensor = torch.from_numpy(waveform).float()
        else:
            wav_tensor = waveform.float()

        # Ensure 1D
        if wav_tensor.dim() > 1:
            wav_tensor = wav_tensor.squeeze()

        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        segments = [(ts["start"], ts["end"]) for ts in speech_timestamps]
        return segments

    def apply_vad(
        self,
        waveform: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
    ) -> np.ndarray:
        """
        Remove non-speech segments and concatenate speech-only audio.

        Args:
            waveform: Audio signal (mono, 16kHz, float32).
            threshold: VAD threshold.
            min_speech_duration_ms: Minimum speech segment length.

        Returns:
            Concatenated speech-only waveform.
        """
        segments = self.get_speech_segments(
            waveform,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
        )

        if not segments:
            warnings.warn("No speech detected in audio. Returning original waveform.")
            return waveform

        # Concatenate speech segments
        speech_chunks = [waveform[start:end] for start, end in segments]
        return np.concatenate(speech_chunks).astype(np.float32)

    def load_and_apply_vad(self, audio_path: str) -> np.ndarray:
        """Load audio file and apply VAD in one step."""
        wav = self._read_audio(audio_path, sampling_rate=self.sampling_rate)
        return self.apply_vad(wav.numpy())
