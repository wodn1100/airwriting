"""
train.py — 모델 학습 (PyTorch 기반)

Usage:
    python python/ml/train.py --data-dir data --epochs 100
    python python/ml/train.py --data-dir data --epochs 50 --lr 0.0005

학습 후 ONNX 모델로 자동 export됩니다.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch는 선택적 의존성
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. Training disabled. Install: pip install torch")


# ──────────────────────────────────────────────
# PyTorch 모델 정의
# model.py의 NumPy 구조를 PyTorch로 재구현
# ──────────────────────────────────────────────

if HAS_TORCH:
    class AirWritingNet(nn.Module):
        """CNN + Bi-LSTM + Attention 모델 (PyTorch)."""

        def __init__(self, num_classes: int, channels: int = 28,
                     seq_len: int = 150):
            super().__init__()
            self.num_classes = num_classes

            # CNN Feature Extractor
            self.cnn = nn.Sequential(
                nn.Conv1d(channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),          # seq_len → 75

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),          # 75 → 37
            )

            # Bi-LSTM
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )

            # Attention
            self.attn_W = nn.Linear(256, 256, bias=False)
            self.attn_v = nn.Linear(256, 1, bias=False)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def attention(self, lstm_out: torch.Tensor) -> tuple:
            """
            Attention mechanism.

            Args:
                lstm_out: (B, T, 256)
            Returns:
                context: (B, 256)
                weights: (B, T)
            """
            energy = torch.tanh(self.attn_W(lstm_out))  # (B, T, 256)
            scores = self.attn_v(energy).squeeze(-1)     # (B, T)
            weights = torch.softmax(scores, dim=-1)      # (B, T)
            context = torch.sum(
                lstm_out * weights.unsqueeze(-1), dim=1
            )  # (B, 256)
            return context, weights

        def forward(self, x: torch.Tensor) -> tuple:
            """
            Args:
                x: (B, seq_len, channels)
            Returns:
                logits: (B, num_classes)
                attn_weights: (B, T)
            """
            # CNN expects (B, C, T)
            h = x.permute(0, 2, 1)
            h = self.cnn(h)
            h = h.permute(0, 2, 1)  # back to (B, T', 128)

            # Bi-LSTM
            h, _ = self.lstm(h)  # (B, T', 256)

            # Attention
            context, attn_w = self.attention(h)  # (B, 256)

            # Classifier
            logits = self.classifier(context)  # (B, num_classes)

            return logits, attn_w

    class AugmentedDataset(Dataset):
        """데이터 증강(Data Augmentation)을 적용한 데이터셋"""
        def __init__(self, X: torch.Tensor, y: torch.Tensor):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = self.X[idx].clone()
            y = self.y[idx]
            
            # 1. Jittering (가우시안 잡음)
            noise = torch.randn_like(x) * 0.03
            x = x + noise
            
            # 2. Scaling (진폭 변동 85% ~ 115%)
            scale = torch.rand(x.shape[-1]) * 0.3 + 0.85
            x = x * scale
            
            return x, y

def train_model(data_dir: str, epochs: int = 100,
                batch_size: int = 32, lr: float = 0.001,
                model_save_dir: str = "models",
                epoch_callback: callable = None):
    """모델 학습 및 ONNX export."""
    if not HAS_TORCH:
        logger.error("PyTorch required for training. pip install torch")
        return

    ROOT = Path(__file__).parent.parent.parent.resolve()
    sys.path.insert(0, str(ROOT))

    from python.ml.dataset import AirWritingDataset

    # 데이터 로드
    dataset = AirWritingDataset(data_dir=data_dir)
    if not dataset.load():
        logger.error(f"No dataset found in {data_dir}/samples.npz")
        logger.error("Collect data first using the recording system.")
        return

    logger.info(f"\n{dataset.summary()}")

    if dataset.num_samples < 10:
        logger.error(f"Not enough samples: {dataset.num_samples} (need ≥10)")
        return

    if dataset.num_classes < 2:
        logger.error(f"Not enough classes: {dataset.num_classes} (need ≥2)")
        return

    # 정규화
    dataset.compute_normalization()
    X, y = dataset.get_tensors()
    X_norm = dataset.normalize(X)

    # Train/Val 분할 (80/20)
    n = len(X_norm)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.FloatTensor(X_norm[train_idx])
    y_train = torch.LongTensor(y[train_idx])
    X_val = torch.FloatTensor(X_norm[val_idx])
    y_val = torch.LongTensor(y[val_idx])

    # 학습 데이터 증강 적용
    train_loader = DataLoader(
        AugmentedDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )

    logger.info(
        f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, "
        f"Classes: {dataset.num_classes}"
    )

    # 모델 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirWritingNet(
        num_classes=dataset.num_classes,
        channels=dataset.channels,
        seq_len=dataset.seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    logger.info(f"Training on: {device}")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 학습 루프
    best_val_acc = 0.0
    best_epoch = 0
    save_dir = Path(model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        
        # 클래스별 통계
        class_corrects = {i: 0 for i in range(dataset.num_classes)}
        class_totals = {i: 0 for i in range(dataset.num_classes)}

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                preds = logits.argmax(1)
                
                # Global
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
                
                # Per class
                for i in range(len(y_batch)):
                    label = y_batch[i].item()
                    pred = preds[i].item()
                    class_totals[label] += 1
                    if label == pred:
                        class_corrects[label] += 1

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        
        # 콜백용 per-class 메트릭 포맷팅
        class_metrics = {}
        for i in range(dataset.num_classes):
            cls_name = dataset.classes[i]
            total = class_totals[i]
            correct = class_corrects[i]
            c_acc = correct / total if total > 0 else 0.0
            class_metrics[cls_name] = {"acc": c_acc, "samples": total}

        scheduler.step(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%}"
            )
            
        if epoch_callback:
            epoch_callback(epoch, epochs, avg_loss, train_acc, val_acc, class_metrics)

        # 최고 성능 저장 (첫 에포크는 무조건 저장하여 극소량 데이터셋에서 파일 누락 방지)
        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = max(val_acc, best_val_acc)
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pth")

    logger.info(
        f"\n✅ Training complete! "
        f"Best Val Acc: {best_val_acc:.2%} (epoch {best_epoch})"
    )

    # ── ONNX Export ──
    model.load_state_dict(torch.load(save_dir / "best_model.pth", weights_only=True))
    model.eval()
    model.to("cpu")

    dummy = torch.randn(1, dataset.seq_len, dataset.channels)
    onnx_path = save_dir / "airwriting_attn.onnx"

    try:
        torch.onnx.export(
            model, dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits", "attention"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
                "attention": {0: "batch"},
            },
            opset_version=17,
        )
        logger.info(f"✅ ONNX model exported: {onnx_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")

    # 메타데이터 저장
    import json
    meta = {
        "classes": dataset.classes,
        "channels": dataset.channels,
        "seq_len": dataset.seq_len,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "normalization": {
            "mean": dataset._mean.tolist() if dataset._mean is not None else None,
            "std": dataset._std.tolist() if dataset._std is not None else None,
        },
    }
    with open(save_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Model metadata saved: {save_dir / 'model_meta.json'}")
    
    return best_val_acc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="AirWriting Model Training")
    ap.add_argument("--data-dir", default="data", help="Dataset directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
