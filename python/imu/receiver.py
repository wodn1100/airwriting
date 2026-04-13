"""
receiver.py — ESP32 UDP 데이터 수신 및 Discovery 응답 서버

ESP32에서 12345 포트로 전송되는 IMU 데이터를 수신하고,
12344 포트 Discovery 요청에 자동 응답하여 서버 IP를 알려줍니다.
"""

import asyncio
import logging
import socket
import struct
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DISCOVERY_REQUEST = "AIRWRITING_DISCOVER_V1"
DISCOVERY_RESPONSE_FMT = "AIRWRITING_SERVER_V1 {} {}"


class DiscoveryResponder:
    """ESP32의 브로드캐스트 탐색 요청에 자동 응답."""

    def __init__(self, discovery_port: int, data_port: int):
        self.discovery_port = discovery_port
        self.data_port = data_port
        self._sock: Optional[socket.socket] = None
        self._running = False

    def start(self):
        """Discovery 응답 서버 시작 (별도 스레드에서 호출)."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.bind(("", self.discovery_port))
        self._sock.settimeout(1.0)
        self._running = True

        local_ip = self._get_local_ip()
        logger.info(f"Discovery responder started on :{self.discovery_port} (IP: {local_ip})")

        while self._running:
            try:
                data, addr = self._sock.recvfrom(256)
                msg = data.decode("utf-8", errors="ignore").strip()
                if msg == DISCOVERY_REQUEST:
                    response = DISCOVERY_RESPONSE_FMT.format(local_ip, self.data_port)
                    self._sock.sendto(response.encode(), addr)
                    logger.info(f"Discovery response sent to {addr}: {response}")
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.warning("Discovery socket error", exc_info=True)
                break

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
            self._sock = None

    @staticmethod
    def _get_local_ip() -> str:
        """로컬 IP 주소를 얻습니다."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


class IMUReceiver:
    """
    ESP32에서 UDP로 전송되는 IMU 패킷을 수신하는 비동기 수신기.

    Usage:
        receiver = IMUReceiver(port=12345)
        receiver.on_packet = my_callback
        await receiver.run()
    """

    def __init__(self, port: int = 12345, buffer_size: int = 256):
        self.port = port
        self.buffer_size = buffer_size
        self.on_packet: Optional[Callable[[bytes, tuple], None]] = None
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._packet_count = 0
        self._error_count = 0

    async def run(self):
        """비동기 수신 루프 시작."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", self.port))
        self._sock.setblocking(False)
        self._running = True

        logger.info(f"IMU Receiver listening on UDP :{self.port}")

        loop = asyncio.get_event_loop()
        while self._running:
            try:
                data, addr = await loop.run_in_executor(
                    None, self._blocking_recv
                )
                if data and self.on_packet:
                    self._packet_count += 1
                    self.on_packet(data, addr)
            except Exception as e:
                if self._running:
                    self._error_count += 1
                    if self._error_count % 100 == 1:
                        logger.warning(f"Receive error #{self._error_count}: {e}")

    def _blocking_recv(self) -> tuple:
        """블로킹 수신 (executor에서 실행)."""
        self._sock.setblocking(True)
        self._sock.settimeout(0.5)
        try:
            return self._sock.recvfrom(self.buffer_size)
        except socket.timeout:
            return None, None

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info(
            f"IMU Receiver stopped. "
            f"Packets: {self._packet_count}, Errors: {self._error_count}"
        )

    @property
    def stats(self) -> dict:
        return {
            "packets_received": self._packet_count,
            "errors": self._error_count,
            "running": self._running,
        }
