import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_kaggle_creditcard(temp_dir: Path) -> pd.DataFrame:
    """Download and load Kaggle credit card fraud dataset if possible."""
    dataset = "mlg-ulb/creditcardfraud"
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(temp_dir),
        "--unzip",
        "-q",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    csv_path = temp_dir / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Expected creditcard.csv after Kaggle download")

    return pd.read_csv(csv_path)


def make_synthetic_fraud(n_rows: int = 20000, n_features: int = 28, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Create a synthetic transaction table with rare fraud labels."""
    rng = np.random.default_rng(42)
    y = (rng.random(n_rows) < fraud_rate).astype(np.int64)

    x_legit = rng.normal(0.0, 1.0, size=(n_rows, n_features))
    x_fraud = rng.normal(0.8, 1.2, size=(n_rows, n_features))
    x = np.where(y[:, None] == 1, x_fraud, x_legit)

    amount = np.exp(rng.normal(3.0 + 1.5 * y, 1.0, size=n_rows))
    time = np.sort(rng.integers(0, 172800, size=n_rows))

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y
    return df


def prepare_features(df: pd.DataFrame, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    if "Class" not in df.columns:
        raise ValueError("Dataset must include a 'Class' label column")

    y = np.array(df["Class"].to_numpy(dtype=np.int64), copy=True)
    x = np.array(df.drop(columns=["Class"]).to_numpy(dtype=np.float32), copy=True)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x.astype(np.float32), y


def build_knn_edges(x: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn_model.fit(x)
    indices = nn_model.kneighbors(return_distance=False)

    src, dst = [], []
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:  # skip self-neighbor
            src.append(i)
            dst.append(int(j))
            src.append(int(j))
            dst.append(i)

    edge_index = np.vstack([np.array(src), np.array(dst)])
    return edge_index


def normalized_adjacency(num_nodes: int, edge_index: np.ndarray) -> torch.Tensor:
    row, col = edge_index
    all_row = np.concatenate([row, np.arange(num_nodes)])
    all_col = np.concatenate([col, np.arange(num_nodes)])

    indices_t = torch.from_numpy(np.vstack([all_row, all_col]).astype(np.int64))
    values_t = torch.from_numpy(np.ones(len(all_row), dtype=np.float32))
    a = torch.sparse_coo_tensor(
        indices=indices_t,
        values=values_t,
        size=(num_nodes, num_nodes),
    ).coalesce()

    deg = torch.sparse.sum(a, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    i = a.indices()
    v = a.values() * deg_inv_sqrt[i[0]] * deg_inv_sqrt[i[1]]
    return torch.sparse_coo_tensor(i, v, a.size()).coalesce()


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(a_hat, x)
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(a_hat, h)
        logits = self.lin2(h).squeeze(1)
        return logits


def build_masks(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.30, random_state=42, stratify=y
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    source = "synthetic"
    df = None

    if not args.synthetic_only:
        try:
            with tempfile.TemporaryDirectory() as td:
                df = load_kaggle_creditcard(Path(td))
            source = "kaggle: mlg-ulb/creditcardfraud"
        except Exception as ex:  # fallback is intentional
            print(f"[info] Kaggle load failed ({ex}). Falling back to synthetic data.")

    if df is None:
        df = make_synthetic_fraud(n_rows=args.synthetic_rows, fraud_rate=args.synthetic_fraud_rate)

    x_np, y_np = prepare_features(df, max_rows=args.max_rows)
    edges = build_knn_edges(x_np, k_neighbors=args.k_neighbors)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np).float()
    a_hat = normalized_adjacency(num_nodes=len(y_np), edge_index=edges)

    train_mask, val_mask, test_mask = build_masks(y_np)
    train_mask_t = torch.from_numpy(train_mask)
    val_mask_t = torch.from_numpy(val_mask)
    test_mask_t = torch.from_numpy(test_mask)

    model = SimpleGCN(in_dim=x.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos = max(int(y_np.sum()), 1)
    neg = max(len(y_np) - pos, 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[info] Data source: {source}")
    print(f"[info] Rows: {len(y_np)} | Fraud rate: {y_np.mean():.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, a_hat)

        loss = criterion(logits[train_mask_t], y[train_mask_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(logits[val_mask_t]).cpu().numpy()
            val_y = y_np[val_mask]
            val_auc = safe_roc_auc(val_y, val_probs)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} val_auc={val_auc:.4f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(x, a_hat)
        test_probs = torch.sigmoid(final_logits[test_mask_t]).cpu().numpy()

    test_y = y_np[test_mask]
    test_auc = safe_roc_auc(test_y, test_probs)
    test_pr_auc = average_precision_score(test_y, test_probs)

    print("\n[results]")
    print(f"test_roc_auc={test_auc:.4f}")
    print(f"test_pr_auc={test_pr_auc:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple GNN fraud detector")
    p.add_argument("--use-kaggle", action="store_true", help="Try Kaggle first")
    p.add_argument("--synthetic-only", action="store_true", help="Skip Kaggle and use synthetic data")
    p.add_argument("--synthetic-rows", type=int, default=20000)
    p.add_argument("--synthetic-fraud-rate", type=float, default=0.02)
    p.add_argument("--max-rows", type=int, default=25000)
    p.add_argument("--k-neighbors", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
