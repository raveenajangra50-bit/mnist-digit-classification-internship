# MNIST Digit Classification Using Neural Networks

## Overview

A reproducible PyTorch pipeline for recognizing handwritten digits using a compact Convolutional Neural Network (CNN). This project was completed as **Internship Project 1** at **Naviotech Solution**.

The pipeline includes dataset loading, validation splitting, data augmentation, training with checkpointing, comprehensive evaluation, and a demo notebook that runs inference on a sample handwritten digit image.

## Verified Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.11%** |
| Test Samples | 10,000 |
| Correct Predictions | 9,911 |
| Incorrect Predictions | 89 |
| Macro Precision | ~99.10% |
| Macro Recall | ~99.10% |
| Macro F1-Score | ~99.10% |
| Sample Prediction (digit 3) | Confidence: 0.999983 |

## Project Structure

```
mnist-digit-classification-internship/
├── src/
│   ├── data.py               # Data loading, augmentation, validation split
│   ├── model.py               # CNN architecture (SimpleCNN)
│   ├── train.py               # Training script with checkpointing
│   ├── evaluate.py            # Evaluation with metrics and visualizations
│   └── download_verify.py     # Dataset download and verification utility
├── tests/
│   └── test_data_shapes.py    # Unit tests for dataloader shapes
├── notebooks/
│   └── digit_classifier.ipynb # Demo notebook for inference on sample image
├── outputs/
│   ├── summary.json           # Evaluation metrics summary
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── misclassified.png
├── checkpoints/
│   └── model_best.pth         # Best trained model weights
├── assets/
│   └── 3-digit.PNG            # Original sample handwritten digit image
├── README.md
├── requirements.txt
├── .gitignore
└── plan.md
```

## Environment Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### 1. Download MNIST and Train

```bash
python src/train.py --epochs 2 --batch-size 64 --augment --download
```

### 2. Evaluate the Model

```bash
python -m src.evaluate --checkpoint checkpoints/model_best.pth --batch-size 256 --output-dir outputs
```

### 3. Run the Demo Notebook

```bash
jupyter notebook notebooks/digit_classifier.ipynb
```

The notebook loads `assets/3-digit.PNG` and runs inference using the trained model.

## CNN Architecture

The model (`SimpleCNN`) is a compact 3-layer CNN:

| Layer | Details |
|---|---|
| Conv1 | 1 → 32 channels, 3×3 kernel, padding=1 |
| Conv2 | 32 → 64 channels, 3×3 kernel, padding=1 |
| Conv3 | 64 → 128 channels, 3×3 kernel, padding=1 |
| Pooling | MaxPool2d(2) after Conv2 and Conv3 |
| Dropout | 0.25 after conv block and FC1 |
| FC1 | 128×7×7 → 256 |
| FC2 | 256 → 10 (output classes) |

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: RandomRotation(10°), RandomAffine (translate, shear)
- **Validation Split**: 10% of training data
- **Normalization**: MNIST mean=0.1307, std=0.3081
- **Reproducibility**: Seeded (seed=42)

## Key Features

- Complete training pipeline with validation monitoring
- Best model checkpointing based on validation loss
- Comprehensive evaluation with per-class metrics
- Confusion matrix visualization
- Sample predictions and misclassified examples visualization
- Demo notebook for single-image inference
- All paths are relative for portability

## Dependencies

- PyTorch ≥ 1.13.0
- torchvision ≥ 0.14.0
- NumPy ≥ 1.23
- matplotlib
- scikit-learn
- Jupyter
- tqdm

## Notes

- The project uses MNIST from torchvision with standard normalization.
- Validation data is created via a seeded split from the training set.
- No deployment server (Flask/FastAPI) was added per internship requirements.
- The original Naviotech project files were kept unchanged.

## Author

Internship project completed at **Naviotech Solution** — July/August 2026.
