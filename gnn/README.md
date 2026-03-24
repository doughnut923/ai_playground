# Simple GNN Fraud Detection (Python)

This project trains a small Graph Neural Network (GCN-style) for fraud detection.

It supports two data sources:
1. Kaggle `mlg-ulb/creditcardfraud` (if your Kaggle API is configured)
2. Synthetic mock transaction dataset (automatic fallback)

## 1) Setup

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## 2) Kaggle API config (optional)

To use Kaggle data, place `kaggle.json` in:
- `%USERPROFILE%/.kaggle/kaggle.json`

Then run:

```powershell
python train_gnn_fraud.py --use-kaggle
```

If Kaggle is not configured, the script falls back to synthetic data.

## 3) Run with synthetic data directly

```powershell
python train_gnn_fraud.py --synthetic-only
```

## 4) Useful options

```powershell
python train_gnn_fraud.py --help
```

Key args:
- `--epochs`: training epochs (default 40)
- `--k-neighbors`: k for graph edges (default 8)
- `--max-rows`: subsample row count for faster experiments (default 25000)

## Expected output

The script prints:
- dataset source
- class balance
- per-epoch loss and validation ROC-AUC
- final test ROC-AUC and PR-AUC
