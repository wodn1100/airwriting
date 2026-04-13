"""
kinematic_chain.py — 직렬 운동학적 사슬 제약 (계획서 §1, §3)

S1(forearm) → S2(hand) → S3(finger)로 이어지는 직렬 관절 체인을 모델링합니다.
드리프트로 인한 비현실적인 관절 꺾임이나 골격 연장을 방지합니다.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from python.fusion import quaternion as quat

logger = logging.getLogger(__name__)


@dataclass
class Joint:
    """운동학적 사슬의 관절 노드."""
    name: str
    bone_length: float     # 다음 관절까지의 뼈 길이 (미터)
    position: np.ndarray   # 3D 위치
    orientation: np.ndarray  # 쿼터니언 [w, x, y, z]
    parent: Optional[str] = None
    bone_direction: Optional[np.ndarray] = None  # 다음 관절을 향하는 단위 방향 벡터
    offset_mm: Optional[np.ndarray] = None  # S3 기준 센서 위치 (mm)

    # 관절 각도 제한 (라디안) - 에어라이팅 자유도 최대화
    max_flexion: float = math.radians(170)   # 거의 제한 없음
    max_extension: float = math.radians(170)
    max_abduction: float = math.radians(170)


class KinematicChain:
    """
    3단 직렬 운동학적 사슬.

    forearm → hand → finger 순서로 연결된 관절 체인을 관리하며,
    물리적 제약 조건을 적용하여 궤적의 신뢰도를 높입니다.

    Usage:
        chain = KinematicChain(skeleton_config)
        chain.update_joint("forearm", orientation_q, accel)
        chain.update_joint("hand", orientation_q, accel)
        chain.update_joint("finger", orientation_q, accel)

        fingertip = chain.get_fingertip_position()
    """

    def __init__(self, skeleton_config: list, sample_rate: float = 100.0):
        """
        Args:
            skeleton_config: settings.yaml의 skeleton_chain 리스트
            sample_rate: 샘플링 주파수 (Hz)
        """
        self.joints: dict[str, Joint] = {}
        self._chain_order: list[str] = []
        self._dt = 1.0 / sample_rate

        # 루트 관절 자유도 추적 상태
        self._root_velocity = np.zeros(3)
        self._root_gravity = np.array([0.0, 0.0, 9.81])
        self._root_damping = 0.95  # 속도 감쇠 (드리프트 억제)
        self._root_displacement_smoothed = np.zeros(3)  # 갑작스런 튐 방지용 스무더

        for joint_cfg in skeleton_config:
            name = joint_cfg["joint"]
            offset_raw = joint_cfg.get("offset_mm", [0, 0, 0])
            joint = Joint(
                name=name,
                bone_length=joint_cfg.get("bone_length_m", 0.1),
                position=np.zeros(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                parent=joint_cfg.get("parent"),
                offset_mm=np.array(offset_raw, dtype=float),
            )
            self.joints[name] = joint
            self._chain_order.append(name)

        # 실측 오프셋으로 뼈 방향 벡터 계산
        # 좌표계: 펜 끝 방향=-Y, 왼쪽=+X, 위=+Z
        for i in range(len(self._chain_order) - 1):
            parent_name = self._chain_order[i]
            child_name = self._chain_order[i + 1]
            parent_joint = self.joints[parent_name]
            child_joint = self.joints[child_name]

            # 부모 → 자식 벡터 (mm → m)
            diff_mm = child_joint.offset_mm - parent_joint.offset_mm
            diff_m = diff_mm / 1000.0
            length = np.linalg.norm(diff_m)
            if length > 0:
                parent_joint.bone_direction = diff_m / length
            else:
                parent_joint.bone_direction = np.array([0.0, -1.0, 0.0])

            logger.info(
                f"  Bone {parent_name}→{child_name}: "
                f"dir={parent_joint.bone_direction}, len={parent_joint.bone_length*1000:.0f}mm"
            )

        # 마지막 관절(finger)의 뼈 방향: 펜 끝을 향해 -Y
        last_name = self._chain_order[-1]
        self.joints[last_name].bone_direction = np.array([0.0, -1.0, 0.0])

        logger.info(
            f"Kinematic chain initialized: "
            f"{' → '.join(self._chain_order)}"
        )

    def update_joint(self, name: str, orientation: np.ndarray,
                     accel: Optional[np.ndarray] = None):
        """
        특정 관절의 방향을 업데이트하고 제약 조건 적용.

        Args:
            name: 관절 이름 (forearm / hand / finger)
            orientation: 쿼터니언 [w, x, y, z]
            accel: 가속도 (선택, 루트 관절 위치 추적에 사용)
        """
        if name not in self.joints:
            return

        joint = self.joints[name]

        # 관절 각도 제한 적용
        constrained_q = self._apply_angle_limits(joint, orientation)
        joint.orientation = constrained_q

        # 루트 관절(forearm): 가속도 기반 위치 추적 (자유도 해제)
        if joint.parent is None and accel is not None:
            self._update_root_position(joint, accel, orientation)

        # Forward Kinematics: 부모로부터 위치 계산
        self._forward_kinematics(name)

    def _apply_angle_limits(self, joint: Joint,
                            orientation: np.ndarray) -> np.ndarray:
        """관절 각도 제한 적용."""
        if joint.parent is None:
            return quat.normalize(orientation)

        parent = self.joints.get(joint.parent)
        if parent is None:
            return quat.normalize(orientation)

        # 부모 대비 상대 회전 계산
        relative_q = quat.multiply(quat.inverse(parent.orientation), orientation)
        euler = quat.to_euler(relative_q)

        # 각도 클리핑
        roll = np.clip(euler[0], -joint.max_abduction, joint.max_abduction)
        pitch = np.clip(euler[1], -joint.max_extension, joint.max_flexion)
        yaw = np.clip(euler[2], -joint.max_abduction, joint.max_abduction)

        # 클리핑이 적용되었는지 확인
        if not np.allclose(euler, [roll, pitch, yaw], atol=0.01):
            logger.debug(
                f"Joint {joint.name}: angle limited "
                f"[{math.degrees(euler[0]):.1f}, {math.degrees(euler[1]):.1f}, "
                f"{math.degrees(euler[2]):.1f}] → "
                f"[{math.degrees(roll):.1f}, {math.degrees(pitch):.1f}, "
                f"{math.degrees(yaw):.1f}]"
            )

        # 보정된 오일러 → 절대 쿼터니언
        constrained_relative = quat.from_euler(roll, pitch, yaw)
        return quat.normalize(
            quat.multiply(parent.orientation, constrained_relative)
        )

    def _update_root_position(self, joint: Joint, accel: np.ndarray,
                              orientation: np.ndarray):
        """
        루트 관절(forearm) 위치: 적분 없는 직접 변위 방식.

        적분 = 드리프트 축적이므로 사용하지 않습니다.
        대신 두 가지를 조합합니다:

        1. 방향 오프셋: 쿼터니언에서 기울기(tilt)를 추출하여
           팔이 기울어진 방향으로 위치를 약간 이동 → 자연스러움
        2. 순간 가속도 변위: 가속도의 *현재값*을 직접 소량 반영
           (적분이 아님 → 가속 멈추면 즉시 0으로 복귀)

        결과: 적분이 없으므로 드리프트가 원리적으로 불가능.
        """
        # ── 1. 방향(tilt) 기반 오프셋 ──
        # 쿼터니언에서 기울기를 추출: 중력 벡터가 어디를 가리키는지
        gravity_local = quat.rotate_vector(
            self._root_gravity, quat.inverse(orientation)
        )
        # 기울어진 정도를 XY 오프셋으로 변환 (Z는 수직이므로 제외)
        tilt_scale = 0.10  # 최대 ~10cm 오프셋 (자유도 확대)
        tilt_x = -gravity_local[0] / 9.81 * tilt_scale
        tilt_y = -gravity_local[1] / 9.81 * tilt_scale

        # ── 2. 순간 가속도 변위 (부드러운 데드밴드 및 스무딩) ──
        accel_world = quat.rotate_vector(accel, orientation)
        linear_accel = accel_world - self._root_gravity
        accel_norm = np.linalg.norm(linear_accel)

        # 데드밴드 축소: 미세 움직임도 반영
        effective_accel = max(0.0, accel_norm - 0.15)
        accel_displacement = np.zeros(3)
        
        if effective_accel > 0:
            # 가속도 크기에 비례 (최대 15cm까지 허용)
            scale = min(effective_accel * 0.010, 0.15)
            target_accel_displacement = (linear_accel / accel_norm) * scale
        else:
            target_accel_displacement = np.zeros(3)

        # 스무딩 (alpha 높을수록 반응 빠름)
        alpha = 0.30
        self._root_displacement_smoothed = (
            alpha * target_accel_displacement + 
            (1.0 - alpha) * self._root_displacement_smoothed
        )

        # ── 최종 위치: 원점 + tilt + smoothed accel (적분 없음) ──
        joint.position = np.array([
            tilt_x + self._root_displacement_smoothed[0],
            tilt_y + self._root_displacement_smoothed[1],
            self._root_displacement_smoothed[2],
        ])

    def _forward_kinematics(self, name: str):
        """부모 관절로부터 자식 관절 위치 계산 (Forward Kinematics)."""
        joint = self.joints[name]

        if joint.parent is None:
            # 루트 관절: _update_root_position에서 이미 처리
            return

        parent = self.joints.get(joint.parent)
        if parent is None:
            return

        # 부모 위치 + 부모 방향으로 실측 뼈 벡터만큼 이동
        bone_dir = parent.bone_direction
        if bone_dir is None:
            bone_dir = np.array([0.0, -1.0, 0.0])
        bone_vector = bone_dir * parent.bone_length
        rotated_bone = quat.rotate_vector(bone_vector, parent.orientation)
        joint.position = parent.position + rotated_bone

    def get_fingertip_position(self) -> np.ndarray:
        """손가락 끝(fingertip) 위치 반환."""
        if "finger" in self.joints:
            finger = self.joints["finger"]
            # finger 관절에서 끝점까지: 실측 방향으로 이동
            tip_dir = finger.bone_direction
            if tip_dir is None:
                tip_dir = np.array([0.0, -1.0, 0.0])
            tip_offset = tip_dir * finger.bone_length
            rotated = quat.rotate_vector(tip_offset, finger.orientation)
            return finger.position + rotated
        return np.zeros(3)

    def get_all_positions(self) -> dict[str, np.ndarray]:
        """모든 관절 위치 반환 (시각화용)."""
        positions = {}
        for name, joint in self.joints.items():
            positions[name] = joint.position.copy()
        positions["fingertip"] = self.get_fingertip_position()
        return positions

    def get_all_orientations(self) -> dict[str, np.ndarray]:
        """모든 관절 쿼터니언 반환 (Three.js 전달용)."""
        return {
            name: joint.orientation.copy()
            for name, joint in self.joints.items()
        }

    def get_chain_length(self) -> float:
        """현재 체인의 총 길이 (드리프트 검증용)."""
        total = 0.0
        for i in range(1, len(self._chain_order)):
            name = self._chain_order[i]
            prev_name = self._chain_order[i - 1]
            dist = np.linalg.norm(
                self.joints[name].position - self.joints[prev_name].position
            )
            total += dist
        return total

    def get_max_chain_length(self) -> float:
        """체인의 최대 이론 길이 (완전히 펴진 상태)."""
        return sum(
            self.joints[name].bone_length
            for name in self._chain_order
        )

    def is_chain_valid(self, tolerance: float = 0.05) -> bool:
        """
        체인 길이가 물리적으로 타당한지 검증.
        
        실제 길이가 최대 길이를 넘으면 드리프트로 판단.
        """
        current = self.get_chain_length()
        maximum = self.get_max_chain_length()
        return current <= maximum * (1.0 + tolerance)

    def reset(self):
        """모든 관절 초기화."""
        for joint in self.joints.values():
            joint.position = np.zeros(3)
            joint.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._root_velocity = np.zeros(3)

