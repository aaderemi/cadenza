# Cadenza Post-Competition Experiment

A deep learning framework for speech intelligibility prediction using Whisper encoder with cross-validation training and evaluation.

## Overview

This project implements a speech intelligibility prediction model that:
- Uses a pre-trained Whisper encoder for feature extraction
- Employs a transformer-based fine-tuning architecture
- Performs k-fold cross-validation training
- Evaluates on validation and test sets with ensemble predictions
- Applies SpecAugment data augmentation

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
safetensors>=0.3.0
accelerate>=0.20.0
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 12GB
- **Storage**: ~10GB for model checkpoints and data

## Installation

1. **Download data**
```txt
Download and unzip the dataset from https://zenodo.org/records/17950664
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**


Manually install PyTorch. Then install requirements.

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers librosa pandas numpy scikit-learn safetensors accelerate
```

## Data Structure

Unzipped data is organized as follows (script should be in the same folder as `cadenza_data`):

```
cadenza_data/
├── train/
│   └── signals/
│       ├── <signal_id1>.flac
│       ├── <signal_id2>.flac
│       └── ...
├── valid/
│   └── signals/
│       └── ...
├── eval/
│   └── signals/
│       └── ...
└── metadata/
    ├── train_metadata_with_phoneme.json
    ├── valid_metadata_with_phoneme.json
    └── eval_metadata_with_phoneme.json
```

### Metadata Format

JSON files should contain arrays of objects with the following structure:

```json
[
  {
    "signal": "signal_id",
    "prompt": "text prompt",
    "response": "transcribed response",
    "n_words": 10,
    "words_correct": 8,
    "correctness": 0.8,
    "hearing_loss": "Mild"
  }
]
```

## Usage

### Quick Start

**Train and evaluate** (recommended):
```bash
python3 cli_eval.py --mode both --epochs 2
```

### Training Only

```bash
python3 cli_eval.py --mode train --epochs 5 --use-weighted-mean
```

### Evaluation Only

```bash
python3 cli_eval.py --mode evaluate
```

## Command-Line Arguments

### Mode Selection
- `--mode {train,evaluate,both}` - Execution mode (default: `train`)

### Training Parameters
- `--epochs INT` - Number of training epochs (default: `10`)
- `--use-weighted-mean` - Use weighted mean of hidden states (default: enabled)
- `--no-norm` - Turns off hidden state normalization when using weighted mean.
- `--no-weighted-mean` - Disable weighted mean pooling. Defaults to training with final output hidden state.
- `--learning-rate FLOAT` - Learning rate (default: `1e-3`)
- `--batch-size INT` - Training batch size (default: `16`)
- `--warmup-steps INT` - Number of warmup steps (default: `500`)
- `--weight-decay FLOAT` - Weight decay for regularization (default: `0.001`)
- `--n-folds INT` - Number of cross-validation folds (default: `5`)

### Data Parameters
- `--data-dir PATH` - Path to data directory (default: `cadenza_data`)
- `--output-dir PATH` - Path to save outputs (default: `./results`)

### Other Parameters
- `--seed INT` - Random seed for reproducibility (default: `42`)

## Examples

### Basic Training with Default Settings
```bash
python3 cli_eval.py --mode both --epochs 2
```

### Custom Configuration
```bash
python3 cli_eval.py \
  --mode both \
  --epochs 10 \
  --learning-rate 5e-4 \
  --batch-size 32 \
  --warmup-steps 1000 \
  --weight-decay 0.01 \
  --n-folds 5 \
  --data-dir ./my_data \
  --output-dir ./my_results
```

### Training Without Weighted Mean
```bash
python3 cli_eval.py \
  --mode train \
  --epochs 5 \
  --no-weighted-mean \
  --learning-rate 1e-4
```

### Evaluate Pre-trained Models
```bash
python3 cli_eval.py \
  --mode evaluate \
  --output-dir ./results \
  --data-dir ./cadenza_data
```

## Output Files

After training and evaluation, the following files will be generated:

```
results/
├── model_fold_0/
│   └── checkpoint-XXX/
│       ├── model.safetensors
│       ├── config.json
│       └── ...
├── model_fold_1/
├── model_fold_2/
├── model_fold_3/
├── model_fold_4/
├── validation_predictions.csv
├── evaluation_predictions.csv
└── results_summary.json
```

### Output File Descriptions

- **`model_fold_X/`**: Trained model checkpoints for each fold
- **`validation_predictions.csv`**: Predictions on validation set
  - Columns: `signal`, `pred`
- **`evaluation_predictions.csv`**: Predictions on evaluation set
  - Columns: `signal`, `pred`
- **`results_summary.json`**: Performance metrics and configuration
  ```json
  {
    "validation_rmse": 0.291234,
    "evaluation_rmse": 0.305678,
    "n_models": 5,
    "use_weighted_mean": true,
    "epochs": 2
  }
  ```

## Model Architecture

### Components

1. **Encoder**: Pre-trained Whisper-medium.en encoder (frozen)
2. **Feature Aggregation**: Weighted mean or last layer pooling
3. **Transformer**: Whisper-small encoder for fine-tuning
4. **Prediction Head**: Linear layers with ReLU activation and sigmoid output

### Key Features

- **SpecAugment**: Time and frequency masking for augmentation
- **K-Fold Cross-Validation**: 5-fold by default
- **Ensemble Prediction**: Average predictions across all folds
- **Mixed Precision Training**: FP16 for faster training (if GPU available)

