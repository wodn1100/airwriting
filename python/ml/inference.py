"""
inference.py — ONNX Runtime 기반 실시간 추론

학습된 ONNX 모델을 로드하여 실시간으로 문자를 인식합니다.
Web Worker 연동용 WebSocket 서빙도 포함합니다.

Usage:
    # 단독 테스트
    engine = InferenceEngine("models/airwriting_attn.onnx")
    result = engine.predict(stroke_data)

    # 파이프라인에서 사용
    pipeline.on_stroke = engine.predict
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    logger.warning("onnxruntime not installed. pip install onnxruntime")


class InferenceEngine:
    """
    ONNX Runtime 기반 추론 엔진.

    특징:
    - ONNX RT로 초저지연 추론 (<50ms 목표)
    - 자동 정규화 (학습 시 저장된 mean/std 사용)
    - 신뢰도 임계값 기반 필터링
    - Attention 가중치 출력 (시각화용)
    """

    def __init__(self, model_path: str = "models/airwriting_attn.onnx",
                 meta_path: str = None,
                 threshold: float = 0.5):
        self.threshold = threshold
        self.classes: list = []
        self.channels = 28
        self.seq_len = 150
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._session: Optional[object] = None

        model_path = Path(model_path)
        if meta_path is None:
            meta_path = model_path.parent / "model_meta.json"
        else:
            meta_path = Path(meta_path)

        # 메타데이터 로드
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.classes = meta.get("classes", [])
            self.channels = meta.get("channels", 27)
            self.seq_len = meta.get("seq_len", 150)
            norm = meta.get("normalization", {})
            if norm.get("mean"):
                self._mean = np.array(norm["mean"], dtype=np.float32)
            if norm.get("std"):
                self._std = np.array(norm["std"], dtype=np.float32)
            logger.info(
                f"Model meta loaded: {len(self.classes)} classes, "
                f"seq_len={self.seq_len}, channels={self.channels}"
            )
        else:
            logger.warning(f"Model meta not found: {meta_path}")

        # ONNX 세션 로드
        if not HAS_ORT:
            logger.error("onnxruntime required for inference")
            return

        if not model_path.exists():
            logger.warning(f"ONNX model not found: {model_path}")
            logger.warning("Train a model first: python python/ml/train.py")
            return

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(model_path), providers=providers
            )
            actual_providers = self._session.get_providers()
            logger.info(
                f"ONNX model loaded: {model_path} "
                f"(providers: {actual_providers})"
            )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")

    def reload_model(self):
        """저장된 최신 모델과 메타데이터를 다시 메모리로 로드합니다 (재시작 방지)."""
        logger.info("♻️ Hot-reloading ONNX inference engine...")
        # 기존 세션 삭제
        self._session = None
        # 속성 초기화 후 다시 로드
        self._mean = None
        self._std = None
        # 초기화 함수 다시 호출
        self.__init__(threshold=self.threshold)
        logger.info("✅ Hot-reload complete!")

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score 정규화."""
        if self._mean is not None and self._std is not None:
            return (data - self._mean) / self._std
        return data

    def _pad_or_truncate(self, stroke: np.ndarray) -> np.ndarray:
        """시퀀스 길이 맞추기."""
        T, C = stroke.shape
        if T > self.seq_len:
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            return stroke[indices]
        elif T < self.seq_len:
            pad = np.zeros((self.seq_len - T, C), dtype=stroke.dtype)
            return np.vstack([stroke, pad])
        return stroke

    def predict(self, stroke: np.ndarray) -> dict:
        """
        스트로크 데이터로 문자 인식.

        Args:
            stroke: (T, channels) 시계열 데이터

        Returns:
            {
                "class": "ㄱ",
                "class_index": 10,
                "confidence": 0.95,
                "above_threshold": True,
                "latency_ms": 12.5,
                "attention_weights": [...],
                "top3": [("ㄱ", 0.95), ("ㄴ", 0.03), ("ㅋ", 0.01)],
            }
        """
        if not self.is_ready:
            return {"class": None, "confidence": 0.0, "error": "Model not loaded"}

        start = time.perf_counter()

        # 전처리
        stroke = self._pad_or_truncate(stroke)
        stroke = self._normalize(stroke)
        x = stroke[np.newaxis, :, :].astype(np.float32)  # (1, T, C)

        # 추론
        try:
            outputs = self._session.run(None, {"input": x})
            logits = outputs[0]  # (1, num_classes)
            attn_w = outputs[1] if len(outputs) > 1 else None
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"class": None, "confidence": 0.0, "error": str(e)}

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        probs = probs.flatten()

        # 결과
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Top-3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            (self.classes[i] if i < len(self.classes) else f"class_{i}",
             float(probs[i]))
            for i in top3_idx
        ]

        latency = (time.perf_counter() - start) * 1000  # ms

        result = {
            "class": self.classes[pred_idx] if pred_idx < len(self.classes) else None,
            "class_index": pred_idx,
            "confidence": confidence,
            "above_threshold": confidence >= self.threshold,
            "latency_ms": round(latency, 2),
            "top3": top3,
        }

        if attn_w is not None:
            result["attention_weights"] = attn_w.flatten().tolist()

        logger.info(
            f"🔤 Predicted: '{result['class']}' "
            f"({confidence:.1%}) [{latency:.1f}ms]"
        )

        return result

    def predict_batch(self, strokes: list[np.ndarray]) -> list[dict]:
        """배치 추론."""
        return [self.predict(s) for s in strokes]
