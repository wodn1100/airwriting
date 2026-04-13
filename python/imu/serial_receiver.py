"""
serial_receiver.py — ESP32 USB Serial 데이터 수신기

WiFi/UDP 대신 USB Serial로 바이너리 패킷을 수신합니다.
패킷 구조는 기존 V3과 동일 (92 bytes, 0xAA~0x55).

Usage:
    receiver = SerialIMUReceiver(port="COM3", baudrate=921600)
    receiver.on_packet = my_callback
    await receiver.run()
"""

import asyncio
import logging
import struct
import threading
from typing import Callable, Optional

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)

HEADER = 0xAA
FOOTER = 0x55
PACKET_SIZE = 92  # 1 + 4 + 24 + 24 + 36 + 1 + 1 + 1


def list_serial_ports() -> list[str]:
    """사용 가능한 시리얼 포트 목록을 반환합니다."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def find_esp32_port() -> Optional[str]:
    """ESP32가 연결된 포트를 자동 탐색합니다."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        # 일반적인 ESP32 USB 칩셋 식별
        if any(kw in desc for kw in ["cp210", "ch340", "ch9102", "usb-serial", "silicon labs"]):
            logger.info(f"Auto-detected ESP32 on {p.device} ({p.description})")
            return p.device
        if any(kw in mfg for kw in ["silicon", "wch", "espressif"]):
            logger.info(f"Auto-detected ESP32 on {p.device} (mfg: {p.manufacturer})")
            return p.device
    return None


class SerialIMUReceiver:
    """
    ESP32에서 USB Serial로 전송되는 IMU 패킷을 수신하는 비동기 수신기.

    기존 IMUReceiver(UDP)와 동일한 인터페이스를 제공합니다.
    on_packet 콜백은 (bytes, tuple) 형태로 호출되며,
    tuple은 ("SERIAL", port_name) 형태입니다.

    Usage:
        receiver = SerialIMUReceiver(port="COM3")
        receiver.on_packet = my_callback
        await receiver.run()
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 921600,
        buffer_size: int = 4096,
    ):
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        self.on_packet: Optional[Callable[[bytes, tuple], None]] = None
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._packet_count = 0
        self._error_count = 0
        self._sync_loss_count = 0

    def _open_serial(self) -> bool:
        """시리얼 포트를 열고 연결합니다."""
        if self.port is None:
            self.port = find_esp32_port()
            if self.port is None:
                available = list_serial_ports()
                logger.error(
                    f"ESP32 포트를 자동 탐색할 수 없습니다. "
                    f"사용 가능한 포트: {available}"
                )
                return False

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            # DTR/RTS 초기화 (ESP32 리셋 방지)
            self._ser.dtr = False
            self._ser.rts = False

            logger.info(f"Serial port opened: {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            return False

    def _read_until_header(self) -> bool:
        """스트림에서 다음 0xAA 헤더를 찾습니다."""
        while self._running:
            byte = self._ser.read(1)
            if len(byte) == 0:
                continue  # timeout
            if byte[0] == HEADER:
                return True
        return False

    def _read_packet(self) -> Optional[bytes]:
        """
        바이너리 스트림에서 한 패킷(92바이트)을 추출합니다.
        헤더(0xAA)와 푸터(0x55)를 검증합니다.
        """
        # 헤더를 이미 찾았으므로, 나머지 91바이트 읽기
        remaining = self._ser.read(PACKET_SIZE - 1)
        if len(remaining) < PACKET_SIZE - 1:
            self._error_count += 1
            return None

        packet_data = bytes([HEADER]) + remaining

        # 푸터 검증
        if packet_data[-1] != FOOTER:
            self._sync_loss_count += 1
            if self._sync_loss_count % 100 == 1:
                logger.warning(
                    f"Sync loss #{self._sync_loss_count}: "
                    f"footer=0x{packet_data[-1]:02X}, expected 0x55"
                )
            return None

        return packet_data

    def _blocking_reader(self):
        """블로킹 Serial 읽기 루프 (별도 스레드에서 실행)."""
        logger.info("Serial reader thread started")

        while self._running:
            try:
                if not self._read_until_header():
                    break

                packet_data = self._read_packet()
                if packet_data is None:
                    continue

                self._packet_count += 1
                if self.on_packet:
                    self.on_packet(packet_data, ("SERIAL", self.port))

            except serial.SerialException as e:
                if self._running:
                    logger.error(f"Serial read error: {e}")
                    self._error_count += 1
                break
            except Exception as e:
                if self._running:
                    self._error_count += 1
                    if self._error_count % 100 == 1:
                        logger.warning(f"Reader error #{self._error_count}: {e}")

        logger.info("Serial reader thread stopped")

    async def run(self):
        """비동기 수신 루프 시작."""
        if not self._open_serial():
            raise RuntimeError(f"Cannot open serial port: {self.port}")

        self._running = True

        # 부팅 메시지 스킵 (ESP32가 리셋 후 텍스트 메시지를 보냄)
        await asyncio.sleep(0.5)
        if self._ser and self._ser.in_waiting:
            boot_msg = self._ser.read(self._ser.in_waiting)
            try:
                logger.debug(f"ESP32 boot message: {boot_msg.decode('utf-8', errors='replace')}")
            except Exception:
                pass

        logger.info(f"Serial IMU Receiver listening on {self.port}")

        # 블로킹 읽기를 별도 스레드에서 실행
        reader_thread = threading.Thread(
            target=self._blocking_reader, daemon=True
        )
        reader_thread.start()

        # 메인 루프는 asyncio에서 대기
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def send_to_device(self, message: str):
        """ESP32로 텍스트 메시지를 전송합니다 (OLED 업데이트 등)."""
        if self._ser and self._ser.is_open:
            try:
                self._ser.write((message + "\n").encode("utf-8"))
            except serial.SerialException as e:
                logger.warning(f"Failed to send to device: {e}")

    def stop(self):
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        logger.info(
            f"Serial IMU Receiver stopped. "
            f"Packets: {self._packet_count}, "
            f"Errors: {self._error_count}, "
            f"Sync losses: {self._sync_loss_count}"
        )

    @property
    def stats(self) -> dict:
        return {
            "packets_received": self._packet_count,
            "errors": self._error_count,
            "sync_losses": self._sync_loss_count,
            "running": self._running,
        }
