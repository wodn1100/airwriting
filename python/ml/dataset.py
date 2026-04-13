"""
dataset.py — 에어라이팅 시계열 데이터셋 관리

IMU 궤적 데이터를 학습용 데이터셋으로 변환합니다.
- 3센서 × (accel 3 + gyro 3) = 18축 기본 + S3 mag 3축 = 21축
- 쿼터니언 3개 × 4 = 12축 추가 가능 → 최대 33축
- 시계열 윈도잉, 정규화, 증강

파일 형식: NumPy .npz (samples.npz)
  - X: (N, seq_len, channels) — 시계열 데이터
  - y: (N,) — 레이블 인덱스
  - classes: 클래스 이름 리스트
"""

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 기본 채널 구성 (27축)
# S1(forearm): ax,ay,az,gx,gy,gz = 6
# S2(hand):    ax,ay,az,gx,gy,gz = 6
# S3(finger):  ax,ay,az,gx,gy,gz = 6
# S3 mag:      mx,my,mz = 3
# S3 quat:     qw,qx,qy,qz = 4 (Madgwick 출력)
# fingertip:   x,y,z = 3 (궤적)  [사용시 추가]
CHANNEL_NAMES = [
    "s1_ax", "s1_ay", "s1_az", "s1_gx", "s1_gy", "s1_gz",
    "s2_ax", "s2_ay", "s2_az", "s2_gx", "s2_gy", "s2_gz",
    "s3_ax", "s3_ay", "s3_az", "s3_gx", "s3_gy", "s3_gz",
    "s3_mx", "s3_my", "s3_mz",
    "q_w", "q_x", "q_y", "q_z",
    "tip_x", "tip_y", "tip_z",
]

DEFAULT_CHANNELS = 28  # S1(6) + S2(6) + S3(6) + mag(3) + quat(4) + tip(3)
DEFAULT_SEQ_LEN = 150  # 100Hz × 1.5초 = 하나의 글자 최대 길이


class StrokeRecorder:
    """
    실시간 스트로크(한 글자) 데이터 기록기.

    버튼 누르는 동안의 센서 데이터를 프레임별로 기록합니다.

    Usage:
        recorder = StrokeRecorder()

        # 매 프레임:
        recorder.add_frame(frame_data)

        # 버튼 뗐을 때:
        stroke = recorder.finish_stroke()
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS):
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._recording = False

    def start_stroke(self):
        """스트로크 기록 시작."""
        self._frames.clear()
        self._recording = True

    def add_frame(self, frame_data: dict):
        """프레임 데이터 추가."""
        if not self._recording:
            return

        # frame_data에서 채널 벡터 추출
        vec = self._extract_vector(frame_data)
        self._frames.append(vec)

    def finish_stroke(self) -> Optional[np.ndarray]:
        """스트로크 기록 종료. (seq_len, channels) 반환."""
        self._recording = False
        if len(self._frames) < 5:  # 너무 짧은 스트로크 무시
            logger.debug(f"Stroke too short: {len(self._frames)} frames")
            return None

        stroke = np.array(self._frames)  # (T, C)
        logger.info(f"Stroke recorded: {stroke.shape[0]} frames × {stroke.shape[1]} channels")
        return stroke

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @staticmethod
    def _extract_vector(frame: dict) -> np.ndarray:
        """프레임 딕셔너리에서 학습용 벡터 추출."""
        vec = []

        # S1/S2/S3 raw 센서 데이터 (18축)
        for sensor_key in ["s1", "s2", "s3"]:
            sensor = frame.get(sensor_key, {})
            vec.extend([
                sensor.get("ax", 0), sensor.get("ay", 0), sensor.get("az", 0),
                sensor.get("gx", 0), sensor.get("gy", 0), sensor.get("gz", 0),
            ])

        # S3 지자기 (3축)
        s3 = frame.get("s3", {})
        vec.extend([s3.get("mx", 0), s3.get("my", 0), s3.get("mz", 0)])

        # S3 쿼터니언 (4축)
        orientations = frame.get("orientations", {})
        finger_q = orientations.get("finger", [1, 0, 0, 0])
        vec.extend(finger_q)

        # fingertip 위치 (2축 — X, Y만, Z는 깊이라 글자 인식에 불필요할 수 있지만 포함)
        tip = frame.get("fingertip", [0, 0, 0])
        vec.extend(tip[:3])

        return np.array(vec, dtype=np.float32)


class AirWritingDataset:
    """
    에어라이팅 학습 데이터셋.

    .npz 파일로 저장/로드하며, 시계열 윈도잉과 정규화를 지원합니다.
    """

    def __init__(self, data_dir: str = "data",
                 seq_len: int = DEFAULT_SEQ_LEN,
                 channels: int = DEFAULT_CHANNELS):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.channels = channels

        self.X: list[np.ndarray] = []  # 각 항목: (T, C) 가변 길이
        self.y: list[int] = []
        self.classes: list[str] = []
        self._class_to_idx: dict[str, int] = {}

        # 정규화 파라미터
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def add_sample(self, stroke: np.ndarray, label: str):
        """
        스트로크 데이터와 레이블 추가.

        Args:
            stroke: (T, C) 시계열 데이터
            label: 클래스 이름 (예: "ㄱ", "0")
        """
        if label not in self._class_to_idx:
            idx = len(self.classes)
            self.classes.append(label)
            self._class_to_idx[label] = idx

        self.X.append(stroke)
        self.y.append(self._class_to_idx[label])
        logger.info(
            f"Sample added: '{label}' (class {self._class_to_idx[label]}), "
            f"shape={stroke.shape}, total={len(self.X)}"
        )

    def pad_or_truncate(self, stroke: np.ndarray) -> np.ndarray:
        """시퀀스 길이를 seq_len에 맞추기 (패딩 또는 자르기)."""
        T, C = stroke.shape
        if T > self.seq_len:
            # 균등 간격 서브샘플링 (잘라내기보다 정보 손실 적음)
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            return stroke[indices]
        elif T < self.seq_len:
            # 제로 패딩
            pad = np.zeros((self.seq_len - T, C), dtype=stroke.dtype)
            return np.vstack([stroke, pad])
        return stroke

    def get_tensors(self) -> tuple:
        """
        학습용 텐서 반환.

        Returns:
            X: (N, seq_len, channels) ndarray
            y: (N,) ndarray
        """
        if not self.X:
            return np.array([]), np.array([])

        X_padded = np.array([self.pad_or_truncate(s) for s in self.X])
        y_arr = np.array(self.y, dtype=np.int64)

        return X_padded, y_arr

    def compute_normalization(self):
        """전체 데이터셋의 채널별 평균/표준편차 계산."""
        all_data = np.concatenate(self.X, axis=0)  # (total_frames, C)
        self._mean = np.mean(all_data, axis=0)
        self._std = np.std(all_data, axis=0)
        self._std[self._std < 1e-6] = 1.0  # 제로 방지
        logger.info(f"Normalization computed over {all_data.shape[0]} frames")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score 정규화."""
        if self._mean is None:
            self.compute_normalization()
        return (data - self._mean) / self._std

    def save(self, filename: str = "samples.npz"):
        """데이터셋을 .npz로 저장."""
        X_padded, y_arr = self.get_tensors()
        save_path = self.data_dir / filename

        np.savez(
            save_path,
            X=X_padded,
            y=y_arr,
            classes=np.array(self.classes),
            mean=self._mean if self._mean is not None else np.zeros(self.channels),
            std=self._std if self._std is not None else np.ones(self.channels),
        )
        logger.info(
            f"Dataset saved: {save_path} "
            f"({len(self.X)} samples, {len(self.classes)} classes)"
        )

    def load(self, filename: str = "samples.npz"):
        """데이터셋을 .npz에서 로드."""
        load_path = self.data_dir / filename
        if not load_path.exists():
            logger.warning(f"Dataset not found: {load_path}")
            return False

        data = np.load(load_path, allow_pickle=True)
        X_padded = data["X"]
        y_arr = data["y"]
        self.classes = list(data["classes"])
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self._mean = data["mean"]
        self._std = data["std"]

        # 개별 샘플로 복원 (패딩된 상태)
        self.X = [X_padded[i] for i in range(len(X_padded))]
        self.y = list(y_arr)

        logger.info(
            f"Dataset loaded: {load_path} "
            f"({len(self.X)} samples, {len(self.classes)} classes)"
        )
        return True

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_samples(self) -> int:
        return len(self.X)

    def summary(self) -> str:
        """데이터셋 요약."""
        counts = {}
        for label_idx in self.y:
            name = self.classes[label_idx]
            counts[name] = counts.get(name, 0) + 1
        lines = [f"Dataset: {len(self.X)} samples, {len(self.classes)} classes"]
        for cls, cnt in sorted(counts.items()):
            lines.append(f"  {cls}: {cnt} samples")
        return "\n".join(lines)
