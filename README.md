# Speech Anxiety & Depression Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)

A production-ready pipeline for detecting clinical anxiety and depression from raw speech using a **Dual-Branch Fusion Architecture**. 

This repository contains two parallel approaches to predicting anxiety/depression from speech, utilizing deep contextual embeddings from **Wav2Vec 2.0** alongside clinically validated acoustic markers from **eGeMAPS (openSMILE)**:

1.  **`PHQ8/`**: Predicts the exact integer **PHQ-8 Score (0-24)** utilizing classical ML (SVM/XGBoost) after feature extraction and PCA.
2.  **`wave2vecprob/`**: Predicts a **Binary Probability (0-1)** utilizing a PyTorch classification head and LoRA fine-tuning.

---

## Architecture 1: PHQ-8 Score Classifier (`PHQ8/`)

The primary pipeline leverages fully-frozen Wav2Vec 2.0 for robust representations without overfitting on small datasets like DAIC-WOZ.

```text
Raw Audio (16kHz) ──┬──→ [Frozen Wav2Vec 2.0 Base] → [Mean + Max Pooling] ──────────────────────────→ [1536-d]
                    │                                                                                    ↓
                    └──→ [eGeMAPS 88 features] → [Per-Segment Aggregation] ─────────────────────────→ [88-d]
                                                                                                         ↓
                                                                                               [Concatenation]  → (1624-d)
                                                                                                         ↓
                                                                                               [Standard Scaler + PCA] → (~110-d)
                                                                                                         ↓
                                                                                               [SVM / XGBoost]
                                                                                                         ↓
                                                                                               PHQ-8 Score (0-24)
```

### Key Decisions
1. **Frozen 95M Parameters**: Prevents catastrophic overfitting on tiny DAIC-WOZ datasets.
2. **Mean + Max Pooling**: Captures both sustained emotion/prosody (mean) and peak burst characteristics (max).
3. **PCA + SVM**: Reduces dimensions using PCA to 95% variance and balances 25 classes with strict SVM RBF weights (`class_weight='balanced'`).

---

## Architecture 2: Probability Score (`wave2vecprob/`)

This approach uses a late-fusion approach to combine the representational power of large self-supervised models with the explainability of traditional speech features.

```text
Raw Audio (16kHz) ──┬──→ [Wav2Vec 2.0 + LoRA] → [Weighted Layer Aggregation] → [Attention Pooling] → [768-d]
                    │                                                                                    ↓
                    └──→ [eGeMAPS 88 features] → [MLP Projection] ─────────────────────────────────→ [128-d]
                                                                                                         ↓
                                                                                               [Fusion Layer (LayerNorm)]
                                                                                                         ↓
                                                                                               [Classification Head]
                                                                                                         ↓
                                                                                                Anxiety Score (0-1)
```

### Key Innovations 
1. **LoRA Fine-Huning**: Wav2Vec 2.0 is trained using Low-Rank Adaptation (r=8), injecting only ~150K trainable parameters to prevent catastrophic forgetting.
2. **Weighted Layer Aggregation**: Instead of just using the final transformer layer, the model learns a 12-parameter scalar distribution to focus on middle layers.
3. **Attention Pooling**: Replaces simple mean pooling. A learned attention mechanism focuses on anxiety-salient sparse frames (e.g., specific pitch spikes).
4. **eGeMAPS Complement**: A lightweight 88-dimensional feature set injected during late fusion to ground deep representations with physiological markers.


---

## 📂 Project Structure

```text
speech_anxiety_detection/
├── PHQ8/
│   ├── train.py                   # Local feature extraction & SVM/XGBoost training (PHQ-8 0-24)
│   └── predict.py                 # Predict integer PHQ-8 score from new audios
├── wave2vecprob/
│   ├── train.py                   # PyTorch LoRA fine-tuning script (Anxiety 0-1 prob)
│   ├── predict.py                 # Predict binary probability score
│   ├── verify.py                  # Automated test script for tensor shapes
│   └── requirements.txt           # Pinned dependencies
├── configs/
│   └── dataset/                   # Local dataset placement (DAIC-WOZ)
├── kaggle/
│   ├── kaggle_dataprep.py         # Notebook for prep on Kaggle
│   └── kaggle_gpu_train.py        # Standalone Kaggle GPU training script
└── README.md
```

---

## Setup & Installation

```bash
git clone <repo-url>
cd Anxiety-Detection-from-Speech

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (utilize the ones provided in wave2vecprob)
pip install -r wave2vecprob/requirements.txt
```

---

# Usage: PHQ-8 Prediction Pipeline (`PHQ8/`)

### 1. Data Preparation
Our dataloader supports the **local DAIC-WOZ** format directly inside `configs/dataset/`.
Place the `train_split_Depression_AVEC2017.csv` and `dev_split_Depression_AVEC2017.csv` along with `<PID>_P` audio folders here. The dataloader automatically parses `*_TRANSCRIPT.csv` to strip out the virtual interviewer (Ellie) and only runs inference on participant speech.

### 2. Training
Run training to extract features, apply PCA, and train the SVM and XGBoost classifiers.

```bash
# Processes audio, caches hf_embeddings/, trains & saves PCA/SVM in scalers/
python PHQ8/train.py
```

### 3. Local Inference

Given a new audio interview, predict the precise PHQ-8 severity:

```bash
# Basic prediction (uses SVM by default)
python PHQ8/predict.py audio_files/inference1.wav

# Use XGBoost instead
python PHQ8/predict.py audio_files/inference1.wav --classifier xgb

# View step-by-step extraction details
python PHQ8/predict.py audio_files/inference1.wav --verbose

# Output pure JSON for programmatic integration
python PHQ8/predict.py audio_files/inference1.wav --json
```

**Full CLI Arguments:**
*   `audio` (positional): Path to the input `.wav` file
*   `--classifier`: Which trained model to use (`svm` or `xgb`), default: `svm`
*   `--verbose`, `-v`: Prints step-by-step timings and feature shapes
*   `--json`: Outputs only a JSON format without ASCII formatting

**Visual Output Format:**
```text
PHQ-8 Score Prediction Result Format        
File       :                   
PHQ-8 Score:                                   
Severity   :                                
Classifier :                                 
Duration   :           
Elapsed    :                              
Confidence :                            

PHQ-8 Clinical Scale Reference:
  0– 4  Minimal / None
  5– 9  Mild
 10–14  Moderate
 15–19  Moderately Severe
 20–24  Severe
```

**JSON Output Format (`--json`):**
```json
{
  "phq_score": 7,
  "severity": "Mild",
  "confidence": 0.127,
  "duration_s": 16.7,
  "n_segments": 2,
  "classifier": "svm",
  "elapsed_s": 8.5
}
```

---

## Usage: Probability Score Pipeline (`wave2vecprob/`)

### 1. Training

```bash
# Phase 1: SER warm-up (Optional but recommended)
python wave2vecprob/train.py --phase phase1 --train_csv data/ser_labels.csv --epochs 15 --device cuda

# Phase 2: Anxiety probability fine-tuning
python wave2vecprob/train.py --phase phase2 --data_dir configs/dataset --epochs 35 --device cuda
```

### 2. Local Inference / API

```bash
# Predict a single file
python wave2vecprob/predict.py --audio test_recording.wav --checkpoint checkpoints/best_phase2.pt

# Start the FastAPI REST server on port 8000
python wave2vecprob/predict.py --serve --checkpoint checkpoints/best_phase2.pt
```

---

## Performance Targets

Based on the implemented architectures, the expected baseline metrics on the DAIC-WOZ dataset are:

| Metric | Target | Literature SOTA (DAIC-WOZ Baseline) |
|--------|--------|-------------------------------------|
| **AUC-ROC** | ≥ 0.70 | 0.68–0.69 |
| **UAR (Macro Recall)** | ≥ 0.60 | 0.60 |
| **Weighted F1** | ≥ 0.65 | ~0.60 |

---
*Developed based on modern computational paralinguistics and parameter-efficient deep learning principles.*
