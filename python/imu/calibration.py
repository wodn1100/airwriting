"""
calibration.py — IMU 센서 캘리브레이션

1. 정지 바이어스 측정: 센서를 수평으로 놓고 정지 상태에서 bias offset 측정
2. 8자 캘리브레이션: 사용자 신체 특성(팔 길이, 손가락 크기) 개인화 보정 계수 계산
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from python.imu.parser import IMUPacket

logger = logging.getLogger(__name__)


@dataclass
class SensorBias:
    """단일 센서의 바이어스 오프셋."""
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag: Optional[np.ndarray] = None  # S3만 해당

    def to_dict(self) -> dict:
        d = {
            "accel": self.accel.tolist(),
            "gyro": self.gyro.tolist(),
        }
        if self.mag is not None:
            d["mag"] = self.mag.tolist()
        return d


@dataclass
class CalibrationResult:
    """캘리브레이션 결과."""
    s1_bias: SensorBias
    s2_bias: SensorBias
    s3_bias: SensorBias
    gravity_vector: np.ndarray  # 측정된 중력 방향
    calibrated: bool = False
    sample_count: int = 0


class BiasCalibrator:
    """
    정지 상태 바이어스 캘리브레이션.

    센서를 수평으로 놓고 일정 시간(기본 3초) 동안 데이터를 수집하여
    가속도계와 자이로의 bias offset을 계산합니다.

    Usage:
        calibrator = BiasCalibrator(duration_sec=3.0)
        # 매 패킷마다:
        calibrator.feed(packet)
        if calibrator.is_complete:
            result = calibrator.result
    """

    def __init__(self, duration_sec: float = 3.0, sample_rate: int = 100):
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.expected_samples = int(duration_sec * sample_rate)

        # 누적 버퍼
        self._s1_accel: list = []
        self._s1_gyro: list = []
        self._s2_accel: list = []
        self._s2_gyro: list = []
        self._s3_accel: list = []
        self._s3_gyro: list = []
        self._s3_mag: list = []

        self._start_time: Optional[float] = None
        self._complete = False
        self._result: Optional[CalibrationResult] = None

    def reset(self):
        """캘리브레이션 상태 초기화."""
        self._s1_accel.clear()
        self._s1_gyro.clear()
        self._s2_accel.clear()
        self._s2_gyro.clear()
        self._s3_accel.clear()
        self._s3_gyro.clear()
        self._s3_mag.clear()
        self._start_time = None
        self._complete = False
        self._result = None

    def feed(self, packet: IMUPacket):
        """패킷 데이터를 캘리브레이션 버퍼에 추가."""
        if self._complete:
            return

        if self._start_time is None:
            self._start_time = time.monotonic()
            logger.info(
                f"📐 Calibration started. "
                f"Hold sensors still for {self.duration_sec}s..."
            )

        # 정지 상태 검증: 자이로 값이 너무 크면 움직이고 있는 것
        gyro_norm = np.linalg.norm(packet.s1.gyro)
        if gyro_norm > 0.5:  # rad/s — 움직이는 중
            if len(self._s1_accel) > 10:
                logger.warning("⚠️ Movement detected during calibration! Hold still.")
            return

        self._s1_accel.append(packet.s1.accel)
        self._s1_gyro.append(packet.s1.gyro)
        self._s2_accel.append(packet.s2.accel)
        self._s2_gyro.append(packet.s2.gyro)
        self._s3_accel.append(packet.s3.accel)
        self._s3_gyro.append(packet.s3.gyro)
        self._s3_mag.append(packet.s3.mag)

        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.duration_sec and len(self._s1_accel) >= self.expected_samples * 0.8:
            self._compute()

    def _compute(self):
        """수집된 데이터에서 바이어스 계산."""
        s1_accel_mean = np.mean(self._s1_accel, axis=0)
        s1_gyro_mean = np.mean(self._s1_gyro, axis=0)
        s2_accel_mean = np.mean(self._s2_accel, axis=0)
        s2_gyro_mean = np.mean(self._s2_gyro, axis=0)
        s3_accel_mean = np.mean(self._s3_accel, axis=0)
        s3_gyro_mean = np.mean(self._s3_gyro, axis=0)
        s3_mag_mean = np.mean(self._s3_mag, axis=0)

        # 중력 벡터 추정 (정지 상태에서 가속도 평균 ≈ 중력)
        gravity = s1_accel_mean.copy()
        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm > 0:
            gravity = gravity / gravity_norm * 9.81

        # 가속도 바이어스: 정지 상태 평균 - 중력
        # (센서가 수평이면 Z축 ≈ 9.81, X/Y ≈ 0)
        gravity_expected = np.array([0.0, 0.0, 9.81])

        self._result = CalibrationResult(
            s1_bias=SensorBias(
                accel=s1_accel_mean - gravity_expected,
                gyro=s1_gyro_mean,
            ),
            s2_bias=SensorBias(
                accel=s2_accel_mean - gravity_expected,
                gyro=s2_gyro_mean,
            ),
            s3_bias=SensorBias(
                accel=s3_accel_mean - gravity_expected,
                gyro=s3_gyro_mean,
                mag=s3_mag_mean,
            ),
            gravity_vector=gravity,
            calibrated=True,
            sample_count=len(self._s1_accel),
        )

        self._complete = True
        logger.info(
            f"✅ Calibration complete! "
            f"Samples: {self._result.sample_count}, "
            f"Gravity: [{gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f}]"
        )
        logger.info(f"   S1 gyro bias: {s1_gyro_mean}")
        logger.info(f"   S2 gyro bias: {s2_gyro_mean}")
        logger.info(f"   S3 gyro bias: {s3_gyro_mean}")

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    @property
    def progress(self) -> float:
        """캘리브레이션 진행률 (0.0 ~ 1.0)."""
        if self._complete:
            return 1.0
        if self._start_time is None:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return min(elapsed / self.duration_sec, 0.99)


def apply_bias(packet: IMUPacket, cal: CalibrationResult) -> IMUPacket:
    """패킷에 바이어스 보정을 적용합니다."""
    if not cal.calibrated:
        return packet

    # S1 보정
    packet.s1.ax -= cal.s1_bias.accel[0]
    packet.s1.ay -= cal.s1_bias.accel[1]
    packet.s1.az -= cal.s1_bias.accel[2]
    packet.s1.gx -= cal.s1_bias.gyro[0]
    packet.s1.gy -= cal.s1_bias.gyro[1]
    packet.s1.gz -= cal.s1_bias.gyro[2]

    # S2 보정
    packet.s2.ax -= cal.s2_bias.accel[0]
    packet.s2.ay -= cal.s2_bias.accel[1]
    packet.s2.az -= cal.s2_bias.accel[2]
    packet.s2.gx -= cal.s2_bias.gyro[0]
    packet.s2.gy -= cal.s2_bias.gyro[1]
    packet.s2.gz -= cal.s2_bias.gyro[2]

    # S3 보정
    packet.s3.ax -= cal.s3_bias.accel[0]
    packet.s3.ay -= cal.s3_bias.accel[1]
    packet.s3.az -= cal.s3_bias.accel[2]
    packet.s3.gx -= cal.s3_bias.gyro[0]
    packet.s3.gy -= cal.s3_bias.gyro[1]
    packet.s3.gz -= cal.s3_bias.gyro[2]

    return packet
