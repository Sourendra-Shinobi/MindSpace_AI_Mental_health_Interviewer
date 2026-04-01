"""
Inference predictor for the Speech Anxiety Detection pipeline.

Handles the full prediction pipeline:
    Audio file → Preprocessing → Dual-Branch Forward → Post-processing → Result

Designed for loading checkpoints trained on Kaggle and running
inference locally on CPU.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional, List
import warnings

from src.preprocessing.audio_utils import load_audio, normalize_audio, trim_silence
from src.preprocessing.segmentation import segment_or_pad
from src.preprocessing.egemaps_extractor import EgemapsExtractor
from src.models.anxiety_classifier import AnxietyClassifier


class AnxietyPredictor:
    """
    End-to-end anxiety prediction from raw audio files.

    Usage:
        predictor = AnxietyPredictor.from_checkpoint("best_phase2.pt")
        result = predictor.predict("audio.wav")
        print(result["anxiety_score"])   # 0.73
        print(result["label"])           # "anxious"
    """

    def __init__(
        self,
        model: AnxietyClassifier,
        egemaps_extractor: EgemapsExtractor,
        device: str = "cpu",
        threshold: float = 0.5,
        sr: int = 16000,
        segment_sec: float = 10.0,
        hop_sec: float = 5.0,
    ):
        """
        Args:
            model: Loaded AnxietyClassifier.
            egemaps_extractor: Loaded EgemapsExtractor with fitted scaler.
            device: Inference device.
            threshold: Decision threshold for binary classification.
            sr: Sample rate.
            segment_sec: Segment window for long audio.
            hop_sec: Segment hop length.
        """
        self.model = model
        self.model.eval()
        self.egemaps = egemaps_extractor
        self.device = torch.device(device)
        self.model.to(self.device)
        self.threshold = threshold
        self.sr = sr
        self.segment_sec = segment_sec
        self.hop_sec = hop_sec

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        scaler_path: Optional[str] = None,
        model_name: str = "facebook/wav2vec2-base",
        device: str = "cpu",
        threshold: float = 0.5,
        **model_kwargs,
    ) -> "AnxietyPredictor":
        """
        Create predictor from a saved checkpoint.

        This is the primary way to use the predictor after training
        on Kaggle and downloading the checkpoint locally.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            scaler_path: Path to fitted StandardScaler for eGeMAPS.
            model_name: Wav2Vec model name (must match training).
            device: Inference device ("cpu" recommended for local).
            threshold: Decision threshold.
            **model_kwargs: Additional model config overrides.

        Returns:
            Loaded AnxietyPredictor ready for inference.
        """
        # Setup eGeMAPS extractor FIRST to avoid Windows threading deadlocks with PyTorch
        egemaps_extractor = EgemapsExtractor()
        if scaler_path and Path(scaler_path).exists():
            egemaps_extractor.load_scaler(scaler_path)

        # Build model
        model = AnxietyClassifier(
            wav2vec_model=model_name,
            **model_kwargs,
        )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(device)
        )

        # Load trainable weights
        model_state = model.state_dict()
        checkpoint_state = checkpoint.get("model_state_dict", checkpoint)
        for key, value in checkpoint_state.items():
            if key in model_state:
                model_state[key] = value
        model.load_state_dict(model_state)

        return cls(
            model=model,
            egemaps_extractor=egemaps_extractor,
            device=device,
            threshold=threshold,
        )

    def predict(
        self,
        audio_path: Union[str, Path],
        return_segments: bool = False,
    ) -> Dict:
        """
        Predict anxiety from an audio file.

        Handles the complete pipeline:
        1. Load and preprocess audio
        2. Segment if long (>10s)
        3. Extract eGeMAPS features
        4. Run dual-branch forward pass
        5. Aggregate segment predictions
        6. Return result with interpretable acoustic markers

        Args:
            audio_path: Path to audio file.
            return_segments: If True, include per-segment predictions.

        Returns:
            Dict with anxiety_score, label, confidence, acoustic_markers, etc.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Step 1: Load and preprocess
        waveform, sr = load_audio(str(audio_path), sr=self.sr)
        waveform = normalize_audio(waveform, method="peak")
        waveform = trim_silence(waveform, sr=sr)

        # Step 2: Segment
        segments = segment_or_pad(
            waveform, sr, self.segment_sec, self.hop_sec
        )

        # Step 3: Extract eGeMAPS features
        egemaps_features = self._extract_egemaps(waveform, sr)

        # Step 4: Run inference on each segment
        segment_results = []
        for seg_waveform, start_time in segments:
            result = self._predict_segment(seg_waveform, egemaps_features)
            result["start_time"] = start_time
            segment_results.append(result)

        # Step 5: Aggregate
        all_probs = [r["probability"] for r in segment_results]
        avg_prob = float(np.mean(all_probs))

        # Final prediction
        label = "anxious" if avg_prob >= self.threshold else "non-anxious"
        confidence = self._confidence_level(avg_prob)

        # Step 6: Get acoustic markers from eGeMAPS
        acoustic_markers = self._get_acoustic_markers(egemaps_features)

        result = {
            "anxiety_score": round(avg_prob, 4),
            "label": label,
            "confidence": confidence,
            "threshold": self.threshold,
            "num_segments": len(segments),
            "audio_duration_sec": round(len(waveform) / sr, 2),
            "top_acoustic_markers": acoustic_markers,
        }

        if return_segments:
            result["segments"] = segment_results

        return result

    def _predict_segment(
        self,
        waveform: np.ndarray,
        egemaps: np.ndarray,
    ) -> Dict:
        """Run inference on a single audio segment."""
        # Prepare tensors
        wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # [1, T]
        egemaps_tensor = torch.from_numpy(egemaps).float().unsqueeze(0)  # [1, 88]
        mask = torch.ones(1, wav_tensor.shape[1], dtype=torch.long)

        # Move to device
        wav_tensor = wav_tensor.to(self.device)
        egemaps_tensor = egemaps_tensor.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(wav_tensor, egemaps_tensor, mask)
            prob = torch.sigmoid(logits).item()

        return {
            "probability": round(prob, 4),
            "logit": round(logits.item(), 4),
        }

    def _extract_egemaps(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract and optionally scale eGeMAPS features."""
        try:
            features = self.egemaps.extract_from_waveform(waveform, sr)
            if self.egemaps.scaler is not None:
                features = self.egemaps.transform(features)
            return features
        except Exception as e:
            warnings.warn(f"eGeMAPS extraction failed: {e}. Using zero features.")
            return np.zeros(88, dtype=np.float32)

    def _confidence_level(self, probability: float) -> str:
        """Map probability to confidence level."""
        distance = abs(probability - 0.5)
        if distance > 0.3:
            return "high"
        elif distance > 0.15:
            return "medium"
        else:
            return "low"

    def _get_acoustic_markers(self, egemaps: np.ndarray) -> Dict[str, str]:
        """
        Extract interpretable acoustic markers from eGeMAPS features.

        This is a key advantage of the dual-branch approach — we can
        surface clinical acoustic evidence, not just a black-box score.
        """
        markers = {}

        if self.egemaps.feature_names and len(egemaps) == len(self.egemaps.feature_names):
            feature_dict = dict(zip(self.egemaps.feature_names, egemaps))

            # Pitch variability (F0 coefficient of variation)
            f0_cov = feature_dict.get("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", None)
            if f0_cov is not None:
                markers["pitch_variability"] = "elevated" if f0_cov > 0.5 else "normal"

            # Jitter
            jitter = feature_dict.get("jitterLocal_sma3nz_amean", None)
            if jitter is not None:
                markers["jitter"] = "elevated" if jitter > 0.5 else "normal"

            # Shimmer
            shimmer = feature_dict.get("shimmerLocaldB_sma3nz_amean", None)
            if shimmer is not None:
                markers["shimmer"] = "elevated" if shimmer > 0.5 else "normal"

            # HNR (Harmonics-to-Noise Ratio) — lower in anxious speech
            hnr = feature_dict.get("HNRdBACF_sma3nz_amean", None)
            if hnr is not None:
                markers["hnr"] = "reduced" if hnr < -0.5 else "normal"

            # Loudness variability
            loudness_cv = feature_dict.get("loudness_sma3_pctlrange0-2", None)
            if loudness_cv is not None:
                markers["loudness_variability"] = "elevated" if loudness_cv > 0.5 else "normal"
        else:
            markers["note"] = "eGeMAPS feature names unavailable for interpretation"

        return markers

    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
    ) -> List[Dict]:
        """Predict anxiety for multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
            except Exception as e:
                result = {"error": str(e), "file": str(path)}
            results.append(result)
        return results
