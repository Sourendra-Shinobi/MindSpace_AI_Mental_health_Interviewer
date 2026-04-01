"""
Training entry point for Speech Anxiety Detection.

Usage:
    # Phase 2 training (anxiety fine-tuning on DAIC-WOZ)
    python train.py --phase phase2 --data_dir ../dataset --epochs 35

    # Phase 1 training (SER warm-up — requires IEMOCAP etc.)
    python train.py --phase phase1 --data_csv data/labels/train_ser.csv --epochs 15

    # Resume from checkpoint
    python train.py --phase phase2 --resume checkpoints/best_phase1.pt
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.models.anxiety_classifier import AnxietyClassifier
from src.data.dataset import AnxietyDataset, DAICWOZDataset
from src.data.collate import collate_fn
from src.data.augmentation import AudioAugmentor
from src.training.trainer import AnxietyTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Speech Anxiety Detection model"
    )
    parser.add_argument(
        "--phase", type=str, default="phase2",
        choices=["phase1", "phase2"],
        help="Training phase (phase1=SER warm-up, phase2=anxiety FT)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="DAIC-WOZ dataset directory (for Phase 2)"
    )
    parser.add_argument(
        "--train_csv", type=str, default=None,
        help="Training labels CSV (for generic dataset)"
    )
    parser.add_argument(
        "--val_csv", type=str, default=None,
        help="Validation labels CSV"
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/model_config.yaml",
        help="Model config YAML file"
    )
    parser.add_argument(
        "--train_config", type=str, default="configs/training_config.yaml",
        help="Training config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Checkpoint save directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load configs
    model_config = {}
    train_config = {}

    if Path(args.model_config).exists():
        with open(args.model_config) as f:
            model_config = yaml.safe_load(f)

    if Path(args.train_config).exists():
        with open(args.train_config) as f:
            train_config = yaml.safe_load(f)

    # Phase-specific config
    phase_config = train_config.get(args.phase, train_config.get("phase2", {}))

    # Override from CLI
    epochs = args.epochs or phase_config.get("epochs", 35)
    batch_size = args.batch_size or phase_config.get("batch_size", 16)

    print(f"\n{'='*60}")
    print(f"  Speech Anxiety Detection — Training ({args.phase})")
    print(f"  Epochs: {epochs} | Batch: {batch_size}")
    print(f"{'='*60}\n")

    # --- Build model ---
    wav2vec_config = model_config.get("wav2vec", {})
    lora_config = model_config.get("lora", {})
    egemaps_config = model_config.get("egemaps", {})
    fusion_config = model_config.get("fusion", {})
    classifier_config = model_config.get("classifier", {})

    model = AnxietyClassifier(
        wav2vec_model=wav2vec_config.get("model_name", "facebook/wav2vec2-base"),
        lora_r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0.05),
        lora_target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        num_transformer_layers=wav2vec_config.get("num_transformer_layers", 12),
        hidden_size=wav2vec_config.get("hidden_size", 768),
        egemaps_dim=egemaps_config.get("input_dim", 88),
        egemaps_proj_dim=egemaps_config.get("proj_dim", 128),
        egemaps_dropout=egemaps_config.get("dropout", 0.2),
        fusion_output_dim=fusion_config.get("output_dim", 256),
        fusion_dropout=fusion_config.get("dropout", 0.3),
        classifier_hidden=classifier_config.get("hidden_dim", 64),
        classifier_dropout=classifier_config.get("dropout", 0.3),
    )

    model.print_model_summary()

    # --- Build data loaders ---
    augmentor = AudioAugmentor()

    if args.data_dir:
        # DAIC-WOZ mode
        print(f"Loading DAIC-WOZ data from: {args.data_dir}")

        # For DAIC-WOZ, labels need to be provided
        # Default: all label=0 (inference mode) — override with a labels dict
        labels = {}  # TODO: Load from labels CSV when full dataset is available

        train_dataset = DAICWOZDataset(
            data_dir=args.data_dir,
            labels=labels,
            sr=16000,
            segment_sec=10.0,
            augmentor=augmentor,
        )
        val_dataset = DAICWOZDataset(
            data_dir=args.data_dir,
            labels=labels,
            sr=16000,
            segment_sec=10.0,
            augmentor=None,
        )
    elif args.train_csv:
        # Generic CSV mode
        train_dataset = AnxietyDataset(
            csv_path=args.train_csv,
            sr=16000,
            augmentor=augmentor,
        )
        val_csv = args.val_csv or args.train_csv  # Fallback to same CSV
        val_dataset = AnxietyDataset(
            csv_path=val_csv,
            sr=16000,
            augmentor=None,
        )
    else:
        print("ERROR: Provide --data_dir (DAIC-WOZ) or --train_csv (generic dataset)")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 4 on Kaggle/Linux
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # --- Training ---
    trainer = AnxietyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        info = trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: epoch {info['epoch']}")

    # Compute class weight
    if hasattr(train_dataset, "get_label_weights"):
        pos_weight = train_dataset.get_label_weights().item()
    else:
        pos_weight = 1.0

    # Train
    result = trainer.train(
        num_epochs=epochs,
        pos_weight=pos_weight,
        phase=args.phase,
        patience=train_config.get(args.phase, {}).get(
            "early_stopping", {}
        ).get("patience", 7),
    )

    print(f"\n✓ Training complete. Best metric: {result['best_metric']:.4f}")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
