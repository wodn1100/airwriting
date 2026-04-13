"""
filters.py — 센서 융합 필터 모음

1. LowPassFilter: 고주파 노이즈 제거 (계획서 §3)
2. MadgwickFilter: 9축 AHRS 쿼터니언 추정 (계획서 §5)
3. ComplementaryFilter: 경량 가속도+자이로 융합
4. KalmanFilter1D: 1축 칼만 필터 (계획서 §2 전략 1)
"""

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LowPassFilter:
    """
    1차 IIR 저역 통과 필터.

    y[n] = α * x[n] + (1 - α) * y[n-1]
    α = dt / (RC + dt), RC = 1 / (2π * cutoff_hz)
    """

    def __init__(self, cutoff_hz: float = 20.0, sample_rate: float = 100.0):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.alpha = dt / (rc + dt)
        self._prev: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value.copy()
            return value
        self._prev = self.alpha * value + (1.0 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self):
        self._prev = None


class MadgwickFilter:
    """
    Madgwick AHRS 필터 — 9축(가속도+자이로+지자기) 쿼터니언 추정.

    드리프트 보정의 핵심 알고리즘 (계획서 §5).
    beta가 클수록 가속도/지자기 보정이 강함 (수렴 빠르지만 잡음에 민감).
    """

    def __init__(self, beta: float = 0.1, sample_rate: float = 100.0):
        self.beta = beta
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        # 쿼터니언 [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update_imu(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """6축 업데이트 (가속도 + 자이로만, 지자기 없음)."""
        q = self.q.copy()

        # 가속도 정규화
        a_norm = np.linalg.norm(accel)
        if a_norm < 1e-10:
            return q
        a = accel / a_norm

        # 자이로 쿼터니언 미분
        qw, qx, qy, qz = q
        gx, gy, gz = gyro

        q_dot = 0.5 * np.array([
            -qx * gx - qy * gy - qz * gz,
             qw * gx + qy * gz - qz * gy,
             qw * gy - qx * gz + qz * gx,
             qw * gz + qx * gy - qy * gx,
        ])

        # Gradient descent step
        f = np.array([
            2.0 * (qx * qz - qw * qy) - a[0],
            2.0 * (qw * qx + qy * qz) - a[1],
            2.0 * (0.5 - qx * qx - qy * qy) - a[2],
        ])

        j = np.array([
            [-2.0 * qy,  2.0 * qz, -2.0 * qw, 2.0 * qx],
            [ 2.0 * qx,  2.0 * qw,  2.0 * qz, 2.0 * qy],
            [ 0.0,       -4.0 * qx, -4.0 * qy, 0.0     ],
        ])

        step = j.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-10:
            step /= step_norm

        # 쿼터니언 적분
        q += (q_dot - self.beta * step) * self.dt

        # 정규화
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q /= q_norm

        self.q = q
        return q.copy()

    def update_marg(self, accel: np.ndarray, gyro: np.ndarray,
                    mag: np.ndarray) -> np.ndarray:
        """9축 업데이트 (가속도 + 자이로 + 지자기)."""
        q = self.q.copy()

        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)
        if a_norm < 1e-10 or m_norm < 1e-10:
            return self.update_imu(accel, gyro)

        a = accel / a_norm
        m = mag / m_norm

        qw, qx, qy, qz = q
        gx, gy, gz = gyro

        # 지자기를 지구 좌표계로 변환
        hx = (2.0 * m[0] * (0.5 - qy*qy - qz*qz) +
              2.0 * m[1] * (qx*qy - qw*qz) +
              2.0 * m[2] * (qx*qz + qw*qy))
        hy = (2.0 * m[0] * (qx*qy + qw*qz) +
              2.0 * m[1] * (0.5 - qx*qx - qz*qz) +
              2.0 * m[2] * (qy*qz - qw*qx))

        bx = math.sqrt(hx*hx + hy*hy)
        bz = (2.0 * m[0] * (qx*qz - qw*qy) +
              2.0 * m[1] * (qy*qz + qw*qx) +
              2.0 * m[2] * (0.5 - qx*qx - qy*qy))

        # Objective functions (6개)
        f = np.array([
            2.0 * (qx*qz - qw*qy) - a[0],
            2.0 * (qw*qx + qy*qz) - a[1],
            2.0 * (0.5 - qx*qx - qy*qy) - a[2],
            2.0*bx*(0.5 - qy*qy - qz*qz) + 2.0*bz*(qx*qz - qw*qy) - m[0],
            2.0*bx*(qx*qy - qw*qz) + 2.0*bz*(qw*qx + qy*qz) - m[1],
            2.0*bx*(qw*qy + qx*qz) + 2.0*bz*(0.5 - qx*qx - qy*qy) - m[2],
        ])

        j = np.array([
            [-2*qy,      2*qz,     -2*qw,      2*qx],
            [ 2*qx,      2*qw,      2*qz,      2*qy],
            [ 0,        -4*qx,     -4*qy,      0   ],
            [-2*bz*qy,   2*bz*qz,  -4*bx*qy-2*bz*qw, -4*bx*qz+2*bz*qx],
            [-2*bx*qz+2*bz*qx, 2*bx*qy+2*bz*qw, 2*bx*qx+2*bz*qz, -2*bx*qw+2*bz*qy],
            [ 2*bx*qy,   2*bx*qz-4*bz*qx, 2*bx*qw-4*bz*qy, 2*bx*qx],
        ])

        step = j.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-10:
            step /= step_norm

        # 자이로 쿼터니언 미분
        q_dot = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx,
        ])

        q += (q_dot - self.beta * step) * self.dt
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q /= q_norm

        self.q = q
        return q.copy()

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


class ComplementaryFilter:
    """
    보완 필터 — 경량 가속도+자이로 융합.

    angle = alpha * (angle + gyro * dt) + (1 - alpha) * accel_angle
    """

    def __init__(self, alpha: float = 0.98, sample_rate: float = 100.0):
        self.alpha = alpha
        self.dt = 1.0 / sample_rate
        self.roll = 0.0
        self.pitch = 0.0

    def update(self, accel: np.ndarray, gyro: np.ndarray) -> tuple:
        """Roll/Pitch 추정 반환."""
        # 가속도에서 각도 계산
        accel_roll = math.atan2(accel[1], accel[2])
        accel_pitch = math.atan2(
            -accel[0], math.sqrt(accel[1]**2 + accel[2]**2)
        )

        # 보완 필터 적용
        self.roll = (self.alpha * (self.roll + gyro[0] * self.dt) +
                     (1 - self.alpha) * accel_roll)
        self.pitch = (self.alpha * (self.pitch + gyro[1] * self.dt) +
                      (1 - self.alpha) * accel_pitch)

        return self.roll, self.pitch

    def reset(self):
        self.roll = 0.0
        self.pitch = 0.0


class KalmanFilter1D:
    """1차원 칼만 필터."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.5):
        self.q = process_noise   # process noise
        self.r = measurement_noise  # measurement noise
        self.x = 0.0  # state estimate
        self.p = 1.0  # error covariance

    def update(self, measurement: float) -> float:
        # Prediction
        self.p += self.q

        # Update
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)

        return self.x

    def reset(self):
        self.x = 0.0
        self.p = 1.0

class OneEuroFilter:
    """
    1-Euro Filter (One Euro Filter)
    HCI 논문(Casiez et al., 2012)에서 증명된 노이즈 및 지연 방지 적응형 필터.
    속도가 느릴 때는 Jitter를 최소화하고, 빠를 때는 Lag를 최소화합니다.
    """
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.1, d_cutoff: float = 1.0, sample_rate: float = 100.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dt = 1.0 / sample_rate
        self.x_prev = None
        self.dx_prev = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        # 속도(미분값) 저대역 통과 필터링
        a_d = self._smoothing_factor(self.d_cutoff)
        dx = (x - self.x_prev) / self.dt
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # 동적 컷오프 주파수 계산 (이동 속도 비례)
        # 속도가 빠를수록 cutoff 상승 -> a가 커짐 (기존값 비중 감소, lag 최소화)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self._smoothing_factor(cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()

        return x_hat

    def _smoothing_factor(self, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * self.dt
        return r / (r + 1.0)

    def _exponential_smoothing(self, a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        return a * x + (1.0 - a) * x_prev

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
