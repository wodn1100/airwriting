"""
quaternion.py — 쿼터니언 유틸리티

Three.js에 전달할 회전 데이터 생성 및 쿼터니언 연산.
"""

import math
import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    """쿼터니언 정규화 [w, x, y, z]."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """쿼터니언 곱셈 q1 * q2. [w, x, y, z] 형식."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def conjugate(q: np.ndarray) -> np.ndarray:
    """쿼터니언 켤레 (역회전)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def inverse(q: np.ndarray) -> np.ndarray:
    """쿼터니언 역원."""
    c = conjugate(q)
    n_sq = np.dot(q, q)
    if n_sq < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return c / n_sq


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """벡터 v를 쿼터니언 q로 회전: q * v * q^(-1)."""
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = multiply(multiply(q, v_quat), conjugate(q))
    return rotated[1:4]


def to_euler(q: np.ndarray) -> np.ndarray:
    """쿼터니언 → 오일러 각도 [roll, pitch, yaw] (라디안)."""
    w, x, y, z = q

    # Roll (x축 회전)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    # Pitch (y축 회전)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 짐벌 락
    else:
        pitch = math.asin(sinp)

    # Yaw (z축 회전)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return np.array([roll, pitch, yaw])


def from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """오일러 각도 → 쿼터니언 [w, x, y, z]."""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    구면 선형 보간 (SLERP).

    q1과 q2 사이를 t(0~1)로 보간합니다.
    시간 동기화에서 센서 데이터 보간에 사용 (계획서 §3).
    """
    dot = np.dot(q1, q2)

    # 최단 경로 보장
    if dot < 0:
        q2 = -q2
        dot = -dot

    # 거의 같은 방향이면 선형 보간
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize(result)

    theta = math.acos(min(dot, 1.0))
    sin_theta = math.sin(theta)

    s1 = math.sin((1 - t) * theta) / sin_theta
    s2 = math.sin(t * theta) / sin_theta

    return normalize(s1 * q1 + s2 * q2)


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """쿼터니언 → 3×3 회전 행렬."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """두 쿼터니언 사이의 각도 (라디안)."""
    q_diff = multiply(inverse(q1), q2)
    return 2.0 * math.acos(min(abs(q_diff[0]), 1.0))
