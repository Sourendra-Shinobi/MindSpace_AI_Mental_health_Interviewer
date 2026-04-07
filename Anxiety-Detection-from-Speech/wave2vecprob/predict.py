"""
Inference entry point for Speech Anxiety Detection.

Usage:
    # Predict on a single audio file
    python predict.py --audio path/to/audio.wav --checkpoint checkpoints/best_phase2.pt

    # Predict on a directory of audio files
    python predict.py --audio_dir path/to/audios/ --checkpoint checkpoints/best_phase2.pt

    # Start the REST API server
    python predict.py --serve --checkpoint checkpoints/best_phase2.pt
"""

import argparse
import json
import os
from pathlib import Path

# Important: Import opensmile before torch/transformers to prevent Windows thread deadlocks
import opensmile
from src.inference.predictor import AnxietyPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speech Anxiety Detection — Inference"
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to a single audio file"
    )
    parser.add_argument(
        "--audio_dir", type=str, default=None,
        help="Directory of audio files to process"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--scaler", type=str, default="scalers/egemaps_scaler.joblib",
        help="Path to fitted eGeMAPS StandardScaler"
    )
    parser.add_argument(
        "--model_name", type=str, default="facebook/wav2vec2-base",
        help="Wav2Vec model name (must match training)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device (cpu/cuda)"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start FastAPI server instead of CLI prediction"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="API server port (with --serve)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.serve:
        # Start FastAPI server
        os.environ["CHECKPOINT_PATH"] = args.checkpoint
        os.environ["SCALER_PATH"] = args.scaler
        os.environ["MODEL_NAME"] = args.model_name
        os.environ["DEVICE"] = args.device
        os.environ["THRESHOLD"] = str(args.threshold)

        import uvicorn
        print(f"\n Starting Anxiety Detection API on port {args.port}")
        print(f"   Checkpoint: {args.checkpoint}")
        print(f"   Device: {args.device}")
        print(f"   Docs: http://localhost:{args.port}/docs\n")
        uvicorn.run(
            "src.inference.api:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
        )
        return

    # Load predictor
    print(f"\nLoading model from: {args.checkpoint}")
    predictor = AnxietyPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler if Path(args.scaler).exists() else None,
        model_name=args.model_name,
        device=args.device,
        threshold=args.threshold,
    )
    print("✓ Model loaded\n")

    results = []

    if args.audio:
        # Single file prediction
        print(f"Predicting: {args.audio}")
        result = predictor.predict(args.audio)
        results.append({"file": args.audio, **result})
        _print_result(result)

    elif args.audio_dir:
        # Directory prediction
        audio_dir = Path(args.audio_dir)
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            audio_files.extend(audio_dir.glob(ext))

        print(f"Found {len(audio_files)} audio files in {audio_dir}\n")

        for audio_path in sorted(audio_files):
            try:
                result = predictor.predict(str(audio_path))
                results.append({"file": str(audio_path), **result})
                print(
                    f"  {audio_path.name:30s} → "
                    f"Score: {result['anxiety_score']:.3f} | "
                    f"Label: {result['label']:12s} | "
                    f"Confidence: {result['confidence']}"
                )
            except Exception as e:
                print(f"  {audio_path.name:30s} → ERROR: {e}")
                results.append({"file": str(audio_path), "error": str(e)})

    else:
        print("Provide --audio (single file) or --audio_dir (batch)")
        return

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")


def _print_result(result: dict):
    """Pretty-print a prediction result."""
    print(f"\n{'='*50}")
    print(f"  Anxiety Detection Result")
    print(f"{'='*50}")
    print(f"  Score:       {result['anxiety_score']:.4f}")
    print(f"  Label:       {result['label']}")
    print(f"  Confidence:  {result['confidence']}")
    print(f"  Threshold:   {result['threshold']}")
    print(f"  Duration:    {result['audio_duration_sec']}s")
    print(f"  Segments:    {result['num_segments']}")

    markers = result.get("top_acoustic_markers", {})
    if markers:
        print(f"\n  Acoustic Markers:")
        for key, value in markers.items():
            print(f"    {key:25s}: {value}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
