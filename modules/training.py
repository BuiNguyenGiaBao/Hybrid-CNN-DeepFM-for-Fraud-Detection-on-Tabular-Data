import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from cnn_for_extract_feature import TabularCNNNetwork
from deepfm_for_relationship import DeepFM


def resolve_csv_path(csv_path: str) -> str:
    candidate = Path(csv_path)
    if candidate.exists():
        return str(candidate)

    script_dir = Path(__file__).resolve().parent
    search_roots = [Path.cwd(), script_dir, script_dir.parent]
    for root in search_roots:
        for rel in [
            candidate,
            Path("data/merge") / candidate.name,
            Path("merge") / candidate.name,
            Path("data") / candidate.name,
        ]:
            full = (root / rel).resolve()
            if full.exists():
                return str(full)

    raise FileNotFoundError(f"Could not find CSV file: {csv_path}")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1 - self.alpha),
        )
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def bounded_best_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1",
    min_precision: float = 0.30,
    min_threshold: float = 0.70,
    max_threshold: float = 0.95,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    thresholds = thresholds.astype(float)

    valid = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    if not valid.any():
        valid = np.ones_like(thresholds, dtype=bool)

    if metric == "recall":
        mask = valid & (precision >= min_precision)
        if mask.any():
            return float(thresholds[mask][np.argmax(recall[mask])])
        return float(thresholds[valid][np.argmax(recall[valid])])

    if metric == "f1":
        score = 2 * precision * recall / (precision + recall + 1e-8)
    else:  # f2 default for fraud recall emphasis
        beta2 = 4.0
        score = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

    mask = valid & (precision >= min_precision)
    if mask.any():
        return float(thresholds[mask][np.argmax(score[mask])])
    return float(thresholds[valid][np.argmax(score[valid])])


class HybridCNNDeepFM(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        embed_dim: int = 40,
        conv_channels: int = 56,
        kernel_size: int = 3,
        bilinear_rank: int = 24,
        bilinear_out_dim: int = 96,
        seq_length: int = 8,
        cnn_dropout: float = 0.55,
        num_classes: int = 2,
        deepfm_embed_dim: int = 10,
        deepfm_hidden=None,
        deepfm_dropout: float = 0.45,
        dense_num_fields: int = 4,
        freeze_cnn: bool = False,
    ):
        super().__init__()
        self.freeze_cnn = freeze_cnn
        deepfm_hidden = deepfm_hidden or [96, 48]

        self.cnn = TabularCNNNetwork(
            tabular_dim=tabular_dim,
            embed_dim=embed_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            bilinear_rank=bilinear_rank,
            bilinear_out_dim=bilinear_out_dim,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout=cnn_dropout,
        )

        self.deepfm = DeepFM(
            num_classes=num_classes,
            categorical_cardinalities=None,
            num_numerical=0,
            embed_dim=deepfm_embed_dim,
            deep_hidden=deepfm_hidden,
            dropout=deepfm_dropout,
            dense_in_dim=bilinear_out_dim,
            dense_num_fields=dense_num_fields,
            use_bias=True,
        )

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn.get_embedding(x, detach=self.freeze_cnn)
        return self.deepfm(dense_x=features)


class IEEEFraudDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str = "isFraud", drop_id_cols: bool = True):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan)

        if drop_id_cols:
            drop_cols = [c for c in ["TransactionID"] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        self.has_target = target_col in df.columns
        if self.has_target:
            self.y = torch.FloatTensor(df[target_col].values)
            X_df = df.drop(columns=[target_col])
        else:
            self.y = None
            X_df = df

        X_df = X_df.fillna(0.0)
        self.feature_names = X_df.columns.tolist()
        self.X = torch.FloatTensor(X_df.values)

        print(f"✓ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        if self.has_target:
            fraud_ratio = float(self.y.mean().item())
            n_fraud = int(self.y.sum().item())
            n_normal = len(self.y) - n_fraud
            print(f"✓ Class distribution: Normal={n_normal}, Fraud={n_fraud}")
            print(f"✓ Fraud ratio: {fraud_ratio * 100:.4f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_target:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class FraudDetectionTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 5e-5,
        pos_weight: Optional[float] = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        threshold_metric: str = "f1",
        threshold_min_precision: float = 0.20,
        threshold_warmup_epochs: int = 6,
        fixed_threshold: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.best_threshold = float(fixed_threshold) if fixed_threshold is not None else 0.5
        self.fixed_threshold = fixed_threshold
        self.threshold_metric = threshold_metric
        self.threshold_min_precision = threshold_min_precision
        self.threshold_warmup_epochs = threshold_warmup_epochs
        self.current_epoch = 0

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"✓ Using FocalLoss(alpha={focal_alpha:.2f}, gamma={focal_gamma:.2f})")
        else:
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
                print(f"✓ Using BCEWithLogitsLoss(pos_weight={pos_weight:.4f})")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                print("✓ Using BCEWithLogitsLoss()")

    def _compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> Dict[str, float]:
        y_pred = (y_probs >= threshold).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "pr_auc": average_precision_score(y_true, y_probs),
            "threshold": float(threshold),
        }
        beta2 = 4.0
        metrics["f2"] = (1 + beta2) * metrics["precision"] * metrics["recall"] / (
            beta2 * metrics["precision"] + metrics["recall"] + 1e-8
        )
        try:
            metrics["auc"] = roc_auc_score(y_true, y_probs)
        except Exception:
            metrics["auc"] = 0.0
        return metrics

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        pbar = tqdm(loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()

            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            loss = self.criterion(logits, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(batch_y.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_probs = np.asarray(all_probs)
        all_labels = np.asarray(all_labels)
        metrics = self._compute_metrics(all_labels, all_probs, threshold=self.best_threshold)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tune_threshold: bool = False) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()
            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item()
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(batch_y.detach().cpu().numpy())

        all_probs = np.asarray(all_probs)
        all_labels = np.asarray(all_labels)

        threshold = float(self.best_threshold)

        # Warmup: avoid overly conservative thresholds in early epochs
        if self.current_epoch <= 3:
            threshold = 0.50
            self.best_threshold = threshold
        elif self.current_epoch <= self.threshold_warmup_epochs:
            threshold = 0.60 if self.fixed_threshold is None else min(float(self.fixed_threshold), 0.60)
            self.best_threshold = threshold
        elif self.fixed_threshold is not None:
            threshold = float(self.fixed_threshold)
            self.best_threshold = threshold
        elif tune_threshold:
            threshold = bounded_best_threshold(
                all_labels,
                all_probs,
                metric=self.threshold_metric,
                min_precision=self.threshold_min_precision,
                min_threshold=0.60,
                max_threshold=0.85,
            )
            self.best_threshold = threshold

        metrics = self._compute_metrics(all_labels, all_probs, threshold=threshold)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, early_stopping_patience: int, save_path: str):
        history = {k: [] for k in [
            "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "train_f2", "train_auc", "train_pr_auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_f2", "val_auc", "val_pr_auc", "val_threshold",
        ]}

        best_score = -1e9
        patience = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            print("\n" + "=" * 70)
            print(f"Epoch {epoch + 1}/{epochs}")
            print("=" * 70)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, tune_threshold=True)

            precision_gate = 1.0 if val_metrics["precision"] >= self.threshold_min_precision else 0.0
            recall_gate = 1.0 if val_metrics["recall"] >= 0.10 else 0.0
            score = (
                0.35 * val_metrics["precision"]
                + 0.30 * val_metrics["f1"]
                + 0.20 * val_metrics["pr_auc"]
                + 0.10 * val_metrics["auc"]
                + 0.05 * precision_gate
            )
            if recall_gate == 0.0:
                score -= 0.25
            self.scheduler.step(score)

            for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
                for m in ["loss", "accuracy", "precision", "recall", "f1", "f2", "auc", "pr_auc"]:
                    history[f"{prefix}_{m}"].append(metrics[m])
            history["val_threshold"].append(val_metrics["threshold"])

            print("\nTrain Metrics:")
            print(
                f"  Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f}"
            )
            print(
                f"  F1: {train_metrics['f1']:.4f} | F2: {train_metrics['f2']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}"
            )

            print("\nValidation Metrics:")
            print(
                f"  Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}"
            )
            print(
                f"  F1: {val_metrics['f1']:.4f} | F2: {val_metrics['f2']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | Thr: {val_metrics['threshold']:.4f}"
            )
            print(f"  Composite score: {score:.4f}")

            if score > best_score:
                best_score = score
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_threshold": self.best_threshold,
                        "best_score": best_score,
                        "val_metrics": val_metrics,
                    },
                    save_path,
                )
                print(f"\n✓ Best model saved! (Composite: {best_score:.4f})")
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"\n⛔ Early stopping triggered after {epoch + 1} epochs")
                    break

        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds, all_probs = [], []
        for batch in tqdm(loader, desc="Predicting"):
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= self.best_threshold).long()
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_probs)

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"✓ Loaded checkpoint from epoch {ckpt['epoch'] + 1}")
        print(f"✓ Best threshold: {self.best_threshold:.4f}")
        if "best_score" in ckpt:
            print(f"✓ Best composite score: {ckpt['best_score']:.4f}")

    @torch.no_grad()
    def save_best_metrics(self, val_loader: DataLoader, save_dir: str = "./results"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.eval()
        all_probs, all_labels = [], []

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())

        all_probs = np.asarray(all_probs)
        all_labels = np.asarray(all_labels)
        all_preds = (all_probs >= self.best_threshold).astype(int)

        report = classification_report(all_labels, all_preds, target_names=["Normal", "Fraud"])
        print("\nCLASSIFICATION REPORT")
        print(report)
        with open(Path(save_dir) / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(f"Best threshold: {self.best_threshold:.6f}\n\n")
            f.write(report)

        cm = confusion_matrix(all_labels, all_preds)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        axes[1].plot([0, 1], [0, 1], lw=2, linestyle="--")
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "evaluation_metrics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Metrics saved to {save_dir}")


def plot_training_history(history: Dict[str, list], save_dir: str = "./results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    metrics = ["loss", "accuracy", "precision", "recall", "f1", "f2", "auc", "pr_auc"]
    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        ax.plot(history[f"train_{metric}"], label=f"Train {metric}", marker="o")
        ax.plot(history[f"val_{metric}"], label=f"Val {metric}", marker="s")
        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[2, 2].plot(history["val_threshold"], label="Val threshold", marker="d")
    axes[2, 2].set_title("THRESHOLD")
    axes[2, 2].set_xlabel("Epoch")
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Training history saved to {Path(save_dir) / 'training_history.png'}")


def build_sampler(labels: np.ndarray, minority_weight: float = 1.30) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    # softer than pure inverse frequency to reduce overcorrection
    class_weights = np.array([1.0, minority_weight], dtype=np.float64)
    # moderate minority upweighting to avoid flooding false positives
    if counts[1] == 0:
        class_weights[1] = 1.0
    sample_weights = class_weights[labels.astype(int)]
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="High-precision Fraud Detection with CNN-DeepFM")
    parser.add_argument("--train_csv", type=str, default="data/merge/train_processed.csv")
    parser.add_argument("--val_csv", type=str, default="data/merge/val_processed.csv")
    parser.add_argument("--test_csv", type=str, default="data/merge/test_processed.csv")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--mode", type=str, default="train_and_predict", choices=["train", "predict", "train_and_predict"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default="best_fraud_model_final.pth")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_focal_loss", action="store_true", default=False)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "f2", "recall"])
    parser.add_argument("--threshold_min_precision", type=float, default=0.20)
    parser.add_argument("--minority_weight", type=float, default=1.30,
                       help="Minority-class weight for WeightedRandomSampler.")
    parser.add_argument("--fixed_threshold", type=float, default=-1.0,
                       help="Fixed decision threshold. Set to a negative value to enable automatic threshold tuning.")
    parser.add_argument("--dense_num_fields", type=int, default=4,
                       help="Number of latent dense fields created from CNN embedding for the FM branch.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.train_csv = resolve_csv_path(args.train_csv)
    args.val_csv = resolve_csv_path(args.val_csv)
    args.test_csv = resolve_csv_path(args.test_csv)

    print("\n" + "=" * 70)
    print("HIGH-PRECISION IEEE-CIS FRAUD DETECTION: CNN-DeepFM")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")
    print("=" * 70 + "\n")

    train_dataset = val_dataset = test_dataset = None
    train_loader = val_loader = test_loader = None
    n_features = None
    pos_weight = None

    if args.mode in ["train", "train_and_predict"]:
        train_dataset = IEEEFraudDataset(args.train_csv)
        val_dataset = IEEEFraudDataset(args.val_csv)
        n_features = train_dataset.X.shape[1]
        if val_dataset.X.shape[1] != n_features:
            raise ValueError(f"Feature mismatch: train={n_features}, val={val_dataset.X.shape[1]}")
        if train_dataset.feature_names != val_dataset.feature_names:
            raise ValueError("Feature name/order mismatch between train and val.")

        train_labels = train_dataset.y.numpy().astype(int)
        sampler = build_sampler(train_labels, minority_weight=args.minority_weight)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

        n_pos = float(train_dataset.y.sum().item())
        n_neg = float(len(train_dataset.y) - n_pos)
        pos_weight = min(2.0, (n_neg / n_pos) ** 0.40) if n_pos > 0 else 1.0
        print(f"✓ High-precision pos_weight: {pos_weight:.4f}")
    else:
        test_dataset = IEEEFraudDataset(args.test_csv)
        n_features = test_dataset.X.shape[1]

    model = HybridCNNDeepFM(tabular_dim=n_features, dense_num_fields=args.dense_num_fields)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total trainable parameters: {total_params:,}")

    trainer = FraudDetectionTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight if args.mode in ["train", "train_and_predict"] else None,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        threshold_metric=args.threshold_metric,
        threshold_min_precision=args.threshold_min_precision,
        threshold_warmup_epochs=8,
        fixed_threshold=(None if args.fixed_threshold < 0 else args.fixed_threshold),
    )

    if args.mode in ["train", "train_and_predict"]:
        history = trainer.fit(train_loader, val_loader, epochs=args.epochs, early_stopping_patience=6, save_path=args.model_save_path)
        plot_training_history(history, save_dir=args.results_dir)
        trainer.load_checkpoint(args.model_save_path)
        trainer.save_best_metrics(val_loader, save_dir=args.results_dir)
        print("\n✅ TRAINING COMPLETED")

    if args.mode in ["predict", "train_and_predict"]:
        if args.mode == "predict":
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for predict mode")
            trainer.load_checkpoint(args.checkpoint)
            test_dataset = IEEEFraudDataset(args.test_csv)
        else:
            test_dataset = IEEEFraudDataset(args.test_csv)

        if test_dataset.X.shape[1] != n_features:
            raise ValueError(f"Feature mismatch: train/model={n_features}, test={test_dataset.X.shape[1]}")
        if args.mode in ["train", "train_and_predict"] and train_dataset is not None:
            if train_dataset.feature_names != test_dataset.feature_names:
                raise ValueError("Feature name/order mismatch between train and test.")

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        predictions, probabilities = trainer.predict(test_loader)
        test_df = pd.read_csv(args.test_csv)
        if "TransactionID" in test_df.columns:
            submission_df = pd.DataFrame({"TransactionID": test_df["TransactionID"], "isFraud": probabilities})
        else:
            submission_df = pd.DataFrame({"TransactionID": range(len(probabilities)), "isFraud": probabilities})
        submission_df.to_csv(args.output, index=False)
        print(f"✓ Submission saved to {args.output}")
        print(f"✓ Mean fraud probability: {probabilities.mean():.4f}")
        print(f"✓ Predictions >= threshold ({trainer.best_threshold:.4f}): {(probabilities >= trainer.best_threshold).sum()}")
