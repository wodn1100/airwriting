"""
parser.py — AirWriting V3 패킷 파싱

ESP32에서 전송되는 바이너리 패킷을 구조체로 변환합니다.

패킷 구조 (V3):
  Header(0xAA) | Timestamp(4B) | S1(24B) | S2(24B) | S3(36B) | Button(1B) | Checksum(1B) | Footer(0x55)

S1/S2 (SensorData6): ax,ay,az,gx,gy,gz  (float32 × 6 = 24 bytes)
S3    (SensorData9): ax,ay,az,gx,gy,gz,mx,my,mz (float32 × 9 = 36 bytes)
"""

import struct
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 패킷 상수
HEADER = 0xAA
FOOTER = 0x55
PACKET_SIZE = 1 + 4 + 24 + 24 + 36 + 1 + 1 + 1  # = 92 bytes

# struct 포맷 (little-endian)
# B = header, I = timestamp, 6f = S1, 6f = S2, 9f = S3, B = button, B = checksum, B = footer
PACKET_FMT = "<BI6f6f9fBBB"


@dataclass
class SensorData6:
    """6축 센서 데이터 (MPU6050: S1, S2)."""
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gx, self.gy, self.gz])


@dataclass
class SensorData9:
    """9축 센서 데이터 (ICM20948: S3)."""
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gx, self.gy, self.gz])

    @property
    def mag(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz])


@dataclass
class IMUPacket:
    """파싱된 IMU 패킷."""
    timestamp_ms: int
    s1: SensorData6  # forearm
    s2: SensorData6  # hand
    s3: SensorData9  # finger
    button: bool
    valid: bool = True


class PacketParser:
    """
    V3 패킷 바이너리 → IMUPacket 변환.

    Usage:
        parser = PacketParser()
        packet = parser.parse(raw_bytes)
        if packet and packet.valid:
            print(packet.s1.accel)
    """

    def __init__(self):
        self._parse_count = 0
        self._error_count = 0

    def parse(self, data: bytes) -> Optional[IMUPacket]:
        """바이너리 데이터를 IMUPacket으로 파싱."""
        if len(data) < PACKET_SIZE:
            self._error_count += 1
            return None

        # 헤더/푸터 검증
        if data[0] != HEADER or data[-1] != FOOTER:
            self._error_count += 1
            return None

        # 체크섬 검증
        computed_checksum = 0
        for i in range(1, PACKET_SIZE - 2):  # header 제외, checksum/footer 제외
            computed_checksum ^= data[i]

        if computed_checksum != data[-2]:
            self._error_count += 1
            logger.debug(
                f"Checksum mismatch: computed=0x{computed_checksum:02X}, "
                f"received=0x{data[-2]:02X}"
            )
            return None

        # 구조체 언팩
        try:
            values = struct.unpack(PACKET_FMT, data[:PACKET_SIZE])
        except struct.error as e:
            self._error_count += 1
            logger.warning(f"Unpack error: {e}")
            return None

        idx = 0
        header = values[idx]; idx += 1
        timestamp = values[idx]; idx += 1

        # S1 (forearm) — 6축
        s1 = SensorData6(*values[idx:idx+6]); idx += 6

        # S2 (hand) — 6축
        s2 = SensorData6(*values[idx:idx+6]); idx += 6

        # S3 (finger) — 9축
        s3 = SensorData9(*values[idx:idx+9]); idx += 9

        button = bool(values[idx]); idx += 1
        # checksum, footer는 이미 검증됨

        self._parse_count += 1
        return IMUPacket(
            timestamp_ms=timestamp,
            s1=s1,
            s2=s2,
            s3=s3,
            button=button,
            valid=True,
        )

    @property
    def stats(self) -> dict:
        return {
            "parsed": self._parse_count,
            "errors": self._error_count,
        }
