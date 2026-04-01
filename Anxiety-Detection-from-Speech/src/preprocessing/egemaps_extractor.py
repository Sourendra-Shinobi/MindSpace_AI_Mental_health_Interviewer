"""
eGeMAPS feature extraction using openSMILE.

Extracts the 88-feature extended Geneva Minimalistic Acoustic Parameter
Set (eGeMAPSv02) — clinically validated features for affective computing
including jitter, shimmer, HNR, formants, MFCCs, and spectral features.
"""

import numpy as np
import os
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Union, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import warnings


class EgemapsExtractor:
    """
    Wrapper around openSMILE for eGeMAPSv02 Functionals extraction.

    Extracts a single 88-dimensional feature vector per audio segment,
    aggregating frame-level features into statistical functionals
    (mean, variance, percentiles, etc.).
    """

    def __init__(self):
        """Initialize the openSMILE feature extractor."""
        try:
            import opensmile
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except ImportError:
            raise ImportError(
                "opensmile is required for eGeMAPS extraction. "
                "Install via: pip install opensmile"
            )

        self.scaler: Optional[StandardScaler] = None
        self.feature_names = None

    def extract(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract eGeMAPS features from an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended, 16kHz mono).

        Returns:
            Feature vector of shape [88] as float32 numpy array.
        """
        audio_path = str(audio_path)

        # openSMILE returns a pandas DataFrame
        df = self.smile.process_file(audio_path)

        # Store feature names on first extraction
        if self.feature_names is None:
            self.feature_names = df.columns.tolist()

        # Convert to numpy — shape [1, 88] → flatten to [88]
        features = df.values.astype(np.float32).flatten()

        return features

    def extract_from_waveform(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Extract eGeMAPS features from an in-memory waveform.

        openSMILE's Python binding can process signals directly, but
        the Functionals level sometimes requires file-based input.
        We write to a temp file as a reliable fallback.

        Args:
            waveform: Audio signal (mono, float32).
            sr: Sample rate.

        Returns:
            Feature vector of shape [88].
        """
        try:
            # Try direct signal processing first
            import opensmile
            df = self.smile.process_signal(waveform, sr)
            if self.feature_names is None:
                self.feature_names = df.columns.tolist()
            return df.values.astype(np.float32).flatten()
        except Exception:
            # Fallback: write to temp file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, waveform, sr)

            try:
                features = self.extract(tmp_path)
            finally:
                os.unlink(tmp_path)

            return features

    def fit_scaler(self, features: np.ndarray) -> None:
        """
        Fit a StandardScaler on training set features.

        Must be called ONLY on training data. The fitted scaler is
        then used to transform train/val/test features identically.

        Args:
            features: Array of shape [N, 88] from training set.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(features)

    def transform(
        self,
        features: np.ndarray,
        clip_sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Transform features using the fitted scaler, with outlier clipping.

        Args:
            features: Feature array of shape [N, 88] or [88].
            clip_sigma: Clip values beyond ±clip_sigma standard deviations.

        Returns:
            Scaled and clipped features.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_scaler() on training features first, "
                "or load a fitted scaler via load_scaler()."
            )

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        scaled = self.scaler.transform(features).astype(np.float32)

        # Clip outliers at ±clip_sigma (clinical recordings have artifacts)
        if clip_sigma > 0:
            scaled = np.clip(scaled, -clip_sigma, clip_sigma)

        return scaled.flatten() if single else scaled

    def save_scaler(self, path: Union[str, Path]) -> None:
        """Save fitted StandardScaler to disk with joblib."""
        if self.scaler is None:
            raise RuntimeError("No fitted scaler to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, str(path))

    def load_scaler(self, path: Union[str, Path]) -> None:
        """Load a previously fitted StandardScaler from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        self.scaler = joblib.load(str(path))
