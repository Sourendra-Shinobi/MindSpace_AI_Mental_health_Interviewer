"""
Train PHQ-8 Regressor — Local Script

Loads the 856-d speaker embeddings extracted by Notebook 3 and trains
Ridge (baseline) + XGBoost (primary) regressors to predict continuous
PHQ-8 scores (0–24).

=== USAGE ===
1. Download from Kaggle:
     embeddings/embeddings.npy   [N_speakers × 856]
     embeddings/phq_labels.npy   [N_speakers]
     embeddings/speaker_ids.npy  [N_speakers]

2. Place them in a folder (default: ./embeddings/)

3. Run:
     python train_regressor.py
     python train_regressor.py --embeddings_dir path/to/embeddings

=== OUTPUT ===
  scalers/embedding_scaler.joblib    (fitted StandardScaler)
  scalers/ridge_model.joblib         (Ridge baseline)
  scalers/xgboost_model.joblib       (XGBoost primary)
  scalers/results.json               (metrics summary)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def load_embeddings(embeddings_dir: str):
    """Load the 3 .npy files from the embeddings directory."""
    d = Path(embeddings_dir)

    embeddings = np.load(str(d / "embeddings.npy"))
    phq_labels = np.load(str(d / "phq_labels.npy"))
    speaker_ids = np.load(str(d / "speaker_ids.npy"))

    print(f"  embeddings:   shape={embeddings.shape}")
    print(f"  phq_labels:   shape={phq_labels.shape}  range=[{phq_labels.min():.0f}, {phq_labels.max():.0f}]")
    print(f"  speaker_ids:  shape={speaker_ids.shape}  unique={len(np.unique(speaker_ids))}")

    # Sanity checks
    assert embeddings.shape[0] == phq_labels.shape[0] == speaker_ids.shape[0], \
        "Mismatch in number of speakers across files!"
    assert embeddings.shape[1] == 856, \
        f"Expected 856-d feature vectors, got {embeddings.shape[1]}"
    assert not np.any(np.isnan(embeddings)), "NaN values found in embeddings!"
    assert not np.any(np.all(embeddings == 0, axis=1)), \
        "All-zero embeddings found — some participants may have failed extraction"

    return embeddings, phq_labels, speaker_ids


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Concordance Correlation Coefficient (CCC).
    Gold standard metric for regression in affective computing.
    Combines Pearson correlation with bias correction.
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return ccc


def train_ridge(X, y, groups, n_splits=5):
    """Train Ridge regression with GroupKFold cross-validation."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    print("\n" + "─" * 50)
    print("  STAGE A: Ridge Regression (Baseline)")
    print("─" * 50)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_metrics = []
    best_model = None
    best_scaler = None
    best_mae = float("inf")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit scaler on train fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Fit Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)

        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0, 24)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        corr = np.corrcoef(y_val, y_pred)[0, 1] if len(np.unique(y_val)) > 1 else 0.0
        ccc = concordance_correlation_coefficient(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.0

        fold_metrics.append({"mae": mae, "rmse": rmse, "pearson_r": corr, "ccc": ccc})

        n_train_speakers = len(np.unique(groups[train_idx]))
        n_val_speakers = len(np.unique(groups[val_idx]))
        print(f"  Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"r={corr:.3f}  CCC={ccc:.3f}  "
              f"(train={n_train_speakers} / val={n_val_speakers} speakers)")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_scaler = scaler

    # Average metrics
    avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}

    print(f"\n  Mean ± Std:")
    print(f"    MAE:       {avg['mae']:.2f} ± {std['mae']:.2f}")
    print(f"    RMSE:      {avg['rmse']:.2f} ± {std['rmse']:.2f}")
    print(f"    Pearson r: {avg['pearson_r']:.3f} ± {std['pearson_r']:.3f}")
    print(f"    CCC:       {avg['ccc']:.3f} ± {std['ccc']:.3f}")

    # Decision gate
    if avg["mae"] > 6.0:
        print("\n  ⚠️  WARNING: MAE > 6 — the signal may be too noisy.")
        print("     Check that the embeddings and labels match correctly.")
    else:
        print(f"\n  ✓ Ridge baseline passed (MAE={avg['mae']:.2f} < 6)")

    return best_model, best_scaler, {"avg": avg, "std": std, "per_fold": fold_metrics}


def train_xgboost(X, y, groups, n_splits=5):
    """Train XGBoost regression with GroupKFold cross-validation."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("\nxgboost not installed. Install with: pip install xgboost")
        print("Skipping XGBoost training.")
        return None, None, None

    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    print("\n" + "─" * 50)
    print("  STAGE B: XGBoost Regression (Primary)")
    print("─" * 50)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_metrics = []
    best_model = None
    best_scaler = None
    best_mae = float("inf")
    feature_importances = np.zeros(X.shape[1])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
        y_pred = model.predict(X_val_s)
        y_pred = np.clip(y_pred, 0, 24)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        corr = np.corrcoef(y_val, y_pred)[0, 1] if len(np.unique(y_val)) > 1 else 0.0
        ccc = concordance_correlation_coefficient(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.0

        fold_metrics.append({"mae": mae, "rmse": rmse, "pearson_r": corr, "ccc": ccc})
        feature_importances += model.feature_importances_

        n_train_speakers = len(np.unique(groups[train_idx]))
        n_val_speakers = len(np.unique(groups[val_idx]))
        print(f"  Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"r={corr:.3f}  CCC={ccc:.3f}  "
              f"(train={n_train_speakers} / val={n_val_speakers} speakers)")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_scaler = scaler

    # Average metrics
    avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}

    print(f"\n  Mean ± Std:")
    print(f"    MAE:       {avg['mae']:.2f} ± {std['mae']:.2f}")
    print(f"    RMSE:      {avg['rmse']:.2f} ± {std['rmse']:.2f}")
    print(f"    Pearson r: {avg['pearson_r']:.3f} ± {std['pearson_r']:.3f}")
    print(f"    CCC:       {avg['ccc']:.3f} ± {std['ccc']:.3f}")

    # Feature importance analysis
    feature_importances /= n_splits
    print(f"\nFeature Importance Analysis (856-d vector):")

    wav2vec_importance = feature_importances[:768].sum()
    egemaps_importance = feature_importances[768:].sum()
    total = wav2vec_importance + egemaps_importance + 1e-8

    print(f"Wav2Vec dims (0–767):  {wav2vec_importance/total*100:.1f}% of total importance")
    print(f"eGeMAPS dims (768–855): {egemaps_importance/total*100:.1f}% of total importance")

    # Top 10 most important features
    top_indices = np.argsort(feature_importances)[::-1][:10]
    print(f"\n  Top 10 most important features:")
    egemaps_names = _get_egemaps_feature_names()
    for rank, idx in enumerate(top_indices):
        if idx >= 768 and idx - 768 < len(egemaps_names):
            name = f"eGeMAPS: {egemaps_names[idx - 768]}"
        else:
            name = f"Wav2Vec dim {idx}"
        print(f"    {rank+1}. [{idx:3d}] {name}  (importance={feature_importances[idx]:.4f})")

    return best_model, best_scaler, {"avg": avg, "std": std, "per_fold": fold_metrics}


def _get_egemaps_feature_names():
    """Return approximate eGeMAPSv02 feature names."""
    # These are the standard 88 eGeMAPSv02 Functionals features in order
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        # Generate a dummy signal to get feature names
        dummy = np.random.randn(16000).astype(np.float32)
        df = smile.process_signal(dummy, 16000)
        return list(df.columns)
    except Exception:
        # Fallback: abbreviated names
        return [f"egemaps_{i}" for i in range(88)]


def main():
    parser = argparse.ArgumentParser(description="Train PHQ-8 regressors on speaker embeddings")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings",
                        help="Directory containing embeddings.npy, phq_labels.npy, speaker_ids.npy")
    parser.add_argument("--output_dir", type=str, default="scalers",
                        help="Directory to save trained models")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of GroupKFold splits")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHQ-8 Regressor Training (Local)")
    print("=" * 60)

    # ── 1. Load embeddings ────────────────────────────────────────────
    print("\n--- Step 1: Loading embeddings ---")
    try:
        X, y, speaker_ids = load_embeddings(args.embeddings_dir)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"  Make sure the embeddings directory exists at: {args.embeddings_dir}")
        print("  Download embeddings.npy, phq_labels.npy, speaker_ids.npy from Kaggle.")
        return

    n_speakers = len(np.unique(speaker_ids))
    print(f"\n  Total speakers: {n_speakers}")
    print(f"  PHQ-8 distribution: mean={y.mean():.1f}, std={y.std():.1f}")

    # Adjust n_splits if fewer speakers than folds
    n_splits = min(args.n_splits, n_speakers)
    if n_splits < args.n_splits:
        print(f"  Note: Reduced folds from {args.n_splits} to {n_splits} (not enough speakers)")

    if n_speakers < 3:
        print(f"\n  ERROR: Only {n_speakers} speaker(s) — need at least 3 for cross-validation.")
        print("  Upload more participant folders to Kaggle and re-run extraction.")
        return

    # ── 2. Train Ridge ────────────────────────────────────────────────
    ridge_model, ridge_scaler, ridge_results = train_ridge(X, y, speaker_ids, n_splits)

    # ── 3. Train XGBoost ──────────────────────────────────────────────
    xgb_model, xgb_scaler, xgb_results = train_xgboost(X, y, speaker_ids, n_splits)

    # ── 4. Save models ────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  SAVING MODELS")
    print("─" * 50)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib

        if ridge_model:
            joblib.dump(ridge_model, str(output_dir / "ridge_model.joblib"))
            joblib.dump(ridge_scaler, str(output_dir / "ridge_scaler.joblib"))
            print(f"  ✓ Ridge model saved to: {output_dir / 'ridge_model.joblib'}")

        if xgb_model:
            joblib.dump(xgb_model, str(output_dir / "xgboost_model.joblib"))
            joblib.dump(xgb_scaler, str(output_dir / "embedding_scaler.joblib"))
            print(f"  ✓ XGBoost model saved to: {output_dir / 'xgboost_model.joblib'}")
            print(f"  ✓ Scaler saved to: {output_dir / 'embedding_scaler.joblib'}")

    except ImportError:
        print("joblib not installed. Install with: pip install joblib")
        print("Models not saved to disk.")

    # ── 5. Save results summary ───────────────────────────────────────
    results = {
        "n_speakers": int(n_speakers),
        "n_splits": n_splits,
        "embedding_dim": int(X.shape[1]),
        "phq_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
    }

    if ridge_results:
        results["ridge"] = {k: float(v) for k, v in ridge_results["avg"].items()}
    if xgb_results:
        results["xgboost"] = {k: float(v) for k, v in xgb_results["avg"].items()}

    results_path = output_dir / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to: {results_path}")

    # ── 6. Final comparison ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)

    header = f"  {'Model':<12} {'MAE':>6} {'RMSE':>6} {'r':>6} {'CCC':>6}"
    print(header)
    print("  " + "─" * 40)

    if ridge_results:
        r = ridge_results["avg"]
        print(f"  {'Ridge':<12} {r['mae']:>6.2f} {r['rmse']:>6.2f} {r['pearson_r']:>6.3f} {r['ccc']:>6.3f}")
    if xgb_results:
        x = xgb_results["avg"]
        print(f"  {'XGBoost':<12} {x['mae']:>6.2f} {x['rmse']:>6.2f} {x['pearson_r']:>6.3f} {x['ccc']:>6.3f}")

    print()
    print("  Target benchmarks (from AVEC literature):")
    print("    Acceptable: MAE < 6,  r > 0.3")
    print("    Good:       MAE < 5,  r > 0.5")
    print("    Excellent:  MAE ≤ 4,  r ≥ 0.55")

    # Determine best model for inference
    if xgb_results and ridge_results:
        if xgb_results["avg"]["mae"] <= ridge_results["avg"]["mae"]:
            print("\n  → XGBoost selected as primary model for inference")
        else:
            print("\n  → Ridge performed better — consider increasing data or checking features")
    elif xgb_results:
        print("\n  → XGBoost selected as primary model for inference")
    elif ridge_results:
        print("\n  → Ridge selected (install xgboost for better results)")

    print("\n  Next step: Run predict.py with --mode stacking to get PHQ scores")
    print("=" * 60)


if __name__ == "__main__":
    main()
