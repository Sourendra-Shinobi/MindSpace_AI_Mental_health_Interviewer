"""
Training loop for the Speech Anxiety Detection pipeline.

Supports:
- Phase 1 (SER warm-up) and Phase 2 (anxiety fine-tuning)
- Differential learning rates for LoRA vs downstream head
- Gradient accumulation and clipping
- Early stopping on validation AUC-ROC
- Checkpoint saving (LoRA adapters + downstream heads only)
- TensorBoard logging
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import json

from src.training.metrics import compute_metrics, format_metrics
from src.training.losses import WeightedBCELoss, FocalLoss


class AnxietyTrainer:
    """
    Training manager for the dual-branch anxiety classifier.

    Handles both Phase 1 (SER warm-up) and Phase 2 (anxiety FT)
    with appropriate learning rate schedules and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[str] = None,
    ):
        """
        Args:
            model: AnxietyClassifier instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            config: Training config dict (from training_config.yaml).
            checkpoint_dir: Directory for saving checkpoints.
            log_dir: Directory for TensorBoard logs.
            device: Device string ("cuda", "cpu", or None for auto-detect).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.training_history = []

    def _setup_optimizer(self, phase: str = "phase2") -> None:
        """Configure optimizer with differential learning rates."""
        phase_config = self.config.get(phase, self.config.get("phase2", {}))
        lr_config = phase_config.get("lr", {})

        # Group parameters by component with different LRs
        param_groups = []

        # LoRA adapter parameters
        lora_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lr_config.get("lora_adapters", 5e-5),
                "name": "lora_adapters",
            })

        # eGeMAPS MLP parameters
        egemaps_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "egemaps" in n.lower()
        ]
        if egemaps_params:
            param_groups.append({
                "params": egemaps_params,
                "lr": lr_config.get("egemaps_mlp", 5e-5),
                "name": "egemaps_mlp",
            })

        # All other trainable parameters (fusion + head + layer agg + attention pool)
        other_names = set()
        for n, p in self.model.named_parameters():
            if p.requires_grad and "lora" not in n.lower() and "egemaps" not in n.lower():
                other_names.add(n)

        other_params = [
            p for n, p in self.model.named_parameters()
            if n in other_names
        ]
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": lr_config.get("fusion_head", lr_config.get("head", 1e-4)),
                "name": "fusion_head",
            })

        weight_decay = phase_config.get("weight_decay", 0.01)
        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        # Learning rate scheduler
        total_steps = len(self.train_loader) * phase_config.get("epochs", 35)
        warmup_steps = phase_config.get("warmup_steps", 300)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_loss(self, pos_weight: float = 1.0, use_focal: bool = False) -> None:
        """Configure loss function."""
        if use_focal:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = WeightedBCELoss(pos_weight=pos_weight)
        self.criterion.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        all_labels = []
        all_probs = []
        num_batches = 0

        grad_accum_steps = self.config.get("phase2", {}).get(
            "gradient_accumulation_steps", 4
        )
        grad_clip_norm = self.config.get("phase2", {}).get(
            "gradient_clip_norm", 1.0
        )

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            egemaps = batch["egemaps"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            logits = self.model(waveforms, egemaps, attention_mask)
            loss = self.criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0 or (
                batch_idx + 1
            ) == len(self.train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        # Compute epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_probs)
        )
        metrics["loss"] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_probs = []
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        for batch in pbar:
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            egemaps = batch["egemaps"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(waveforms, egemaps, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_probs)
        )
        metrics["loss"] = avg_loss

        return metrics

    def train(
        self,
        num_epochs: int = 35,
        pos_weight: float = 1.0,
        use_focal_loss: bool = False,
        phase: str = "phase2",
        patience: int = 7,
        metric: str = "auc_roc",
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            num_epochs: Maximum number of epochs.
            pos_weight: Class imbalance weight for BCE loss.
            use_focal_loss: If True, use focal loss instead of BCE.
            phase: Training phase ("phase1" or "phase2").
            patience: Early stopping patience.
            metric: Metric to monitor for early stopping.

        Returns:
            Training history dict.
        """
        self._setup_optimizer(phase=phase)
        self._setup_loss(pos_weight=pos_weight, use_focal=use_focal_loss)

        self.best_val_metric = 0.0
        self.patience_counter = 0

        print(f"\n{'='*60}")
        print(f"  Starting training — {phase}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs} | Patience: {patience}")
        print(f"  Monitor: {metric}")
        self.model.print_model_summary()
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            self._log_epoch(train_metrics, val_metrics)

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"Val UAR: {val_metrics.get('uar', 0):.4f}"
            )

            # Early stopping check
            current_metric = val_metrics.get(metric, 0.0)
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(
                    f"best_{phase}.pt", val_metrics
                )
                print(f"  ✓ New best {metric}: {current_metric:.4f} — saved checkpoint")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1} "
                          f"(no improvement for {patience} epochs)")
                    break

            # Save history
            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })

        # Save final checkpoint
        self._save_checkpoint(f"final_{phase}.pt", val_metrics)

        # Save training history
        history_path = self.checkpoint_dir / f"history_{phase}.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)

        return {"history": self.training_history, "best_metric": self.best_val_metric}

    def _save_checkpoint(self, filename: str, metrics: Dict) -> None:
        """
        Save checkpoint with LoRA adapters + downstream heads.

        Only saves trainable parameters — not the full Wav2Vec base.
        On Kaggle, these checkpoints are small (~1–3 MB) and can be
        downloaded and loaded locally.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if any(
                    key in k
                    for key in [
                        "lora", "layer_weights", "attention", "egemaps",
                        "fusion", "classifier",
                    ]
                )
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load a checkpoint (e.g., downloaded from Kaggle).

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint metadata (epoch, metrics).
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights (partial — only trainable parts)
        model_state = self.model.state_dict()
        for key, value in checkpoint["model_state_dict"].items():
            if key in model_state:
                model_state[key] = value
        self.model.load_state_dict(model_state)

        # Optionally load optimizer state
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass  # Optimizer structure may have changed

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        return {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
        }

    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, self.current_epoch)

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, self.current_epoch)
