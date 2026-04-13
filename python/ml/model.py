"""
model.py — 에어라이팅 인식 모델 (계획서 §4)

아키텍처: CNN Feature Extractor → Bi-LSTM → Attention → FC

1. CNN: 센서별 국부 특징 추출 (시간 축 1D 컨볼루션)
2. Bi-LSTM: 양방향 시퀀스 학습 (시작↔끝 곡선 흐름)
3. Attention: 획 복잡도에 따른 가중치 (ㄹ, ㅎ 등 복잡 문자)
4. FC: 최종 분류

입력: (batch, seq_len, channels)  예: (32, 150, 27)
출력: (batch, num_classes)

순수 NumPy 기반 정의 → PyTorch/ONNX 변환 시 이 구조를 따릅니다.
(학습은 PyTorch, 추론은 ONNX Runtime)
"""

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# NumPy 기반 레이어 (추론 전용 가중치 레이어)
# 학습은 PyTorch에서 하고, 가중치를 ONNX로 export
# 여기서는 모델 구조 정의 + numpy forward pass
# ──────────────────────────────────────────────

class Conv1DBlock:
    """
    1D 컨볼루션 블록 (CNN Feature Extractor).

    kernel_size 보통 3 또는 5, stride=1, padding='same'
    Conv1D → BatchNorm → ReLU → MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Xavier 초기화
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size
        ).astype(np.float32) * scale
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # BatchNorm 파라미터
        self.bn_gamma = np.ones(out_channels, dtype=np.float32)
        self.bn_beta = np.zeros(out_channels, dtype=np.float32)
        self.bn_mean = np.zeros(out_channels, dtype=np.float32)
        self.bn_var = np.ones(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, in_channels)
        Returns:
            (batch, seq_len//2, out_channels)
        """
        batch, seq_len, _ = x.shape
        pad = self.kernel_size // 2

        # Pad
        x_padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode='constant')

        # Conv1D (naive but correct)
        out = np.zeros((batch, seq_len, self.out_channels), dtype=np.float32)
        for b in range(batch):
            for oc in range(self.out_channels):
                for t in range(seq_len):
                    window = x_padded[b, t:t+self.kernel_size, :]  # (K, IC)
                    out[b, t, oc] = np.sum(
                        window * self.weight[oc].T
                    ) + self.bias[oc]

        # BatchNorm
        out = (out - self.bn_mean) / np.sqrt(self.bn_var + 1e-5)
        out = out * self.bn_gamma + self.bn_beta

        # ReLU
        out = np.maximum(out, 0)

        # MaxPool (stride=2)
        new_len = seq_len // 2
        pooled = np.zeros((batch, new_len, self.out_channels), dtype=np.float32)
        for i in range(new_len):
            pooled[:, i, :] = np.maximum(
                out[:, 2*i, :], out[:, 2*i+1, :]
            )

        return pooled


class AttentionLayer:
    """
    Self-Attention 레이어 (계획서 §4).

    시퀀스의 각 타임스텝에 대한 중요도 가중치를 학습합니다.
    복잡한 획(ㄹ, ㅎ)에서 전환점에 높은 가중치를 부여합니다.
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        scale = math.sqrt(1.0 / hidden_size)
        self.W = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.v = np.random.randn(hidden_size).astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> tuple:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — 가중 합
            weights: (batch, seq_len) — 어텐션 가중치
        """
        # Score = v^T * tanh(W * x)
        energy = np.tanh(x @ self.W.T)  # (B, T, H)
        scores = energy @ self.v  # (B, T)

        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-10)

        # Weighted sum
        context = np.sum(x * weights[:, :, np.newaxis], axis=1)  # (B, H)

        return context, weights


class AirWritingModel:
    """
    에어라이팅 인식 모델 — CNN + Bi-LSTM + Attention.

    이 클래스는 모델 구조와 NumPy 기반 추론을 정의합니다.
    실제 학습은 train.py에서 PyTorch로 하고,
    추론은 ONNX Runtime으로 합니다.

    구조:
        Input (B, 150, 27)
        → Conv1D(27→64, k=5) + BN + ReLU + MaxPool → (B, 75, 64)
        → Conv1D(64→128, k=3) + BN + ReLU + MaxPool → (B, 37, 128)
        → Bi-LSTM(128→128) → (B, 37, 256)
        → Attention → (B, 256)
        → FC(256→128) + ReLU + Dropout
        → FC(128→num_classes)
        → Softmax
    """

    def __init__(self, num_classes: int, seq_len: int = 150,
                 channels: int = 28):
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.channels = channels

        # CNN blocks
        self.conv1 = Conv1DBlock(channels, 64, kernel_size=5)
        self.conv2 = Conv1DBlock(64, 128, kernel_size=3)

        # Attention
        self.attention = AttentionLayer(256)  # Bi-LSTM output size

        # FC layers
        scale_fc1 = math.sqrt(2.0 / 256)
        self.fc1_weight = np.random.randn(256, 128).astype(np.float32) * scale_fc1
        self.fc1_bias = np.zeros(128, dtype=np.float32)

        scale_fc2 = math.sqrt(2.0 / 128)
        self.fc2_weight = np.random.randn(128, num_classes).astype(np.float32) * scale_fc2
        self.fc2_bias = np.zeros(num_classes, dtype=np.float32)

        logger.info(
            f"Model created: CNN→BiLSTM→Attn→FC "
            f"(classes={num_classes}, seq={seq_len}, ch={channels})"
        )

    def forward(self, x: np.ndarray) -> tuple:
        """
        Forward pass (NumPy 추론용).

        Args:
            x: (batch, seq_len, channels)
        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, T)
        """
        # CNN feature extraction
        h = self.conv1.forward(x)      # (B, 75, 64)
        h = self.conv2.forward(h)      # (B, 37, 128)

        # Bi-LSTM는 NumPy로 구현하기 복잡 → ONNX Runtime 사용
        # 여기서는 단순히 Linear으로 대체 (구조 시연용)
        # 실제 추론은 inference.py에서 ONNX Runtime으로
        h_bidir = np.concatenate([h, h[:, ::-1, :]], axis=-1)  # (B, 37, 256)

        # Attention
        context, attn_weights = self.attention.forward(h_bidir)  # (B, 256)

        # FC
        h = context @ self.fc1_weight + self.fc1_bias  # (B, 128)
        h = np.maximum(h, 0)  # ReLU

        logits = h @ self.fc2_weight + self.fc2_bias  # (B, num_classes)

        return logits, attn_weights

    def predict(self, x: np.ndarray) -> tuple:
        """
        예측 (softmax 적용).

        Returns:
            predicted_class: int
            confidence: float
            probabilities: ndarray
        """
        logits, attn_w = self.forward(x)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        predicted = np.argmax(probs, axis=-1)
        confidence = np.max(probs, axis=-1)

        return predicted, confidence, probs

    def get_config(self) -> dict:
        """모델 설정 반환 (ONNX export 시 참조)."""
        return {
            "architecture": "CNN_BiLSTM_Attention",
            "num_classes": self.num_classes,
            "seq_len": self.seq_len,
            "channels": self.channels,
            "cnn_filters": [64, 128],
            "cnn_kernels": [5, 3],
            "lstm_hidden": 128,
            "lstm_bidirectional": True,
            "attention_size": 256,
            "fc_hidden": 128,
        }
