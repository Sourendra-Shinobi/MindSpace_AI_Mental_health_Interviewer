"""
FastAPI REST API for real-time anxiety prediction.

Endpoints:
    POST /predict — Upload audio file, get anxiety prediction
    GET  /health  — Health check

Returns JSON with anxiety_score, label, confidence, and
interpretable acoustic markers from eGeMAPS.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.inference.predictor import AnxietyPredictor


# ──── API Models ────

class PredictionResponse(BaseModel):
    anxiety_score: float
    label: str
    confidence: str
    threshold: float
    num_segments: int
    audio_duration_sec: float
    processing_time_ms: float
    top_acoustic_markers: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ──── App Setup ────

app = FastAPI(
    title="Speech Anxiety Detection API",
    description=(
        "Dual-Branch Fusion model: Wav2Vec 2.0 + LoRA + eGeMAPS. "
        "Upload audio to get anxiety probability with interpretable "
        "acoustic markers."
    ),
    version="1.0.0",
)

# Global predictor — initialized on startup
predictor: Optional[AnxietyPredictor] = None


@app.on_event("startup")
async def load_model():
    """Load model checkpoint on server startup."""
    global predictor

    checkpoint_path = os.environ.get(
        "CHECKPOINT_PATH", "checkpoints/best_phase2.pt"
    )
    scaler_path = os.environ.get(
        "SCALER_PATH", "scalers/egemaps_scaler.joblib"
    )
    model_name = os.environ.get(
        "MODEL_NAME", "facebook/wav2vec2-base"
    )
    device = os.environ.get("DEVICE", "cpu")
    threshold = float(os.environ.get("THRESHOLD", "0.5"))

    if Path(checkpoint_path).exists():
        try:
            predictor = AnxietyPredictor.from_checkpoint(
                checkpoint_path=checkpoint_path,
                scaler_path=scaler_path,
                model_name=model_name,
                device=device,
                threshold=threshold,
            )
            print(f"✓ Model loaded from {checkpoint_path} on {device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    else:
        print(
            f"⚠ Checkpoint not found at {checkpoint_path}. "
            f"API will return errors until a model is loaded."
        )


# ──── Endpoints ────

@app.post("/predict", response_model=PredictionResponse)
async def predict_anxiety(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC)"),
):
    """
    Predict anxiety from an uploaded audio file.

    Returns anxiety probability, binary label, confidence level,
    and interpretable acoustic markers from eGeMAPS features.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set CHECKPOINT_PATH environment variable.",
        )

    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Use: {allowed_extensions}",
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        suffix=ext, delete=False
    ) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        start_time = time.time()
        result = predictor.predict(tmp_path)
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            anxiety_score=result["anxiety_score"],
            label=result["label"],
            confidence=result["confidence"],
            threshold=result["threshold"],
            num_segments=result["num_segments"],
            audio_duration_sec=result["audio_duration_sec"],
            processing_time_ms=round(processing_time, 1),
            top_acoustic_markers=result.get("top_acoustic_markers", {}),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )
    finally:
        os.unlink(tmp_path)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "n/a",
    )
