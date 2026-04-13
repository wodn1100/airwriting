"""
trajectory.py — 궤적 추정 (이중적분 + ZUPT 드리프트 보정)

가속도 데이터를 이중 적분하여 3D 위치(궤적)를 추정합니다.
ZUPT(Zero Velocity Update)로 정지 구간에서 드리프트를 리셋합니다.
"""

import logging
from typing import Optional

import numpy as np

from python.fusion import quaternion as quat

logger = logging.getLogger(__name__)


class ZUPTDetector:
    """
    Zero Velocity Update 감지기.

    자이로스코프와 가속도의 크기가 임계값 이하일 때 정지 상태로 판단.
    """

    def __init__(self, gyro_threshold: float = 0.05,
                 accel_threshold: float = 0.3,
                 window_size: int = 5):
        self.gyro_threshold = gyro_threshold
        self.accel_threshold = accel_threshold
        self.window_size = window_size
        self._gyro_buffer: list = []
        self._accel_buffer: list = []

    def update(self, gyro_norm: float, accel_deviation: float) -> bool:
        """
        현재 프레임이 정지 상태인지 판단.

        Args:
            gyro_norm: 자이로스코프 크기 (rad/s)
            accel_deviation: 가속도에서 중력을 뺀 크기 (m/s²)

        Returns:
            True if stationary (ZUPT 적용 가능)
        """
        self._gyro_buffer.append(gyro_norm)
        self._accel_buffer.append(accel_deviation)

        if len(self._gyro_buffer) > self.window_size:
            self._gyro_buffer.pop(0)
            self._accel_buffer.pop(0)

        if len(self._gyro_buffer) < self.window_size:
            return False

        avg_gyro = sum(self._gyro_buffer) / len(self._gyro_buffer)
        avg_accel = sum(self._accel_buffer) / len(self._accel_buffer)

        return avg_gyro < self.gyro_threshold and avg_accel < self.accel_threshold

    def reset(self):
        self._gyro_buffer.clear()
        self._accel_buffer.clear()


class TrajectoryEstimator:
    """
    3D 궤적 추정기.

    가속도 → 속도(적분) → 위치(적분)
    ZUPT로 정지 시 속도 리셋하여 드리프트 억제.
    """

    def __init__(self, sample_rate: float = 100.0,
                 zupt_enabled: bool = True,
                 gyro_threshold: float = 0.05,
                 accel_threshold: float = 0.3):
        self.dt = 1.0 / sample_rate
        self.zupt_enabled = zupt_enabled

        # 상태
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.gravity = np.array([0.0, 0.0, 9.81])

        # ZUPT 감지기
        self._zupt = ZUPTDetector(gyro_threshold, accel_threshold)

        # 궤적 기록
        self._trajectory: list = []
        self._is_writing = False
        self._stroke_start_idx = 0

        # 모드 전환 상태 (더블 클릭)
        import time
        self.input_mode = "WRITE"
        self._prev_button = False
        self._last_click_time = 0.0

    def update(self, accel_world: np.ndarray, gyro: np.ndarray,
               button_pressed: bool) -> np.ndarray:
        """
        새 센서 데이터로 위치 업데이트.

        Args:
            accel_world: 월드 좌표계 가속도 (중력 제거 필요)
            gyro: 자이로스코프 데이터
            button_pressed: 펜 버튼 상태

        Returns:
            현재 3D 위치
        """
        # 중력 제거 (선형 가속도)
        linear_accel = accel_world - self.gravity

        # ZUPT 감지
        gyro_norm = np.linalg.norm(gyro)
        accel_deviation = np.linalg.norm(linear_accel)
        is_stationary = self._zupt.update(gyro_norm, accel_deviation)

        if self.zupt_enabled and is_stationary:
            # 정지 상태: 속도 리셋
            self.velocity = np.zeros(3)
        else:
            # 사다리꼴 적분 (Trapezoidal integration)
            self.velocity += linear_accel * self.dt

        # 위치 적분
        self.position += self.velocity * self.dt

        # 더블 클릭 감지 (버튼이 방금 눌린 순간)
        import time
        if button_pressed and not self._prev_button:
            now = time.time()
            if now - self._last_click_time < 0.3: # 0.3초 이내 연타 시 전환
                self.input_mode = "WRITE" if self.input_mode == "MOUSE" else "MOUSE"
                logger.info(f"🔄 [더블클릭 감지] 모드가 전환되었습니다: {self.input_mode}")
                self._last_click_time = 0.0 # 초기화하여 3연타 방지
            else:
                self._last_click_time = now
        self._prev_button = button_pressed

        # 궤적 버퍼 기록 모드 관리
        if button_pressed:
            if not self._is_writing:
                self._is_writing = True
                self._trajectory.clear()  # << 메모리 누수 방지: 과거 궤적 삭제
                self._stroke_start_idx = 0
                logger.debug("✏️ Writing started")
            self._trajectory.append(self.position.copy())
        elif self._is_writing:
            self._is_writing = False
            logger.debug(
                f"✏️ Writing ended. Stroke points: "
                f"{len(self._trajectory) - self._stroke_start_idx}"
            )

        return self.position.copy()

    def get_current_stroke(self) -> np.ndarray:
        """현재 진행 중인 스트로크 궤적."""
        if self._is_writing and self._stroke_start_idx < len(self._trajectory):
            return np.array(self._trajectory[self._stroke_start_idx:])
        return np.array([]).reshape(0, 3)

    def get_full_trajectory(self) -> np.ndarray:
        """전체 기록된 궤적."""
        if not self._trajectory:
            return np.array([]).reshape(0, 3)
        return np.array(self._trajectory)

    def clear_trajectory(self):
        """궤적 초기화."""
        self._trajectory.clear()
        self._stroke_start_idx = 0

    def reset(self):
        """전체 상태 리셋."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self._trajectory.clear()
        self._stroke_start_idx = 0
        self._is_writing = False
        self._zupt.reset()

    @property
    def is_writing(self) -> bool:
        return self._is_writing

    @property
    def trajectory_length(self) -> int:
        return len(self._trajectory)
