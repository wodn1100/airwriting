"""
main.py — AirWriting 에어라이팅 메인 엔트리 포인트

데이터 흐름:
  ESP32 → [UDP:12345] → receiver → parser → calibration
       → LPF → Madgwick → kinematic_chain → trajectory
       → [WebSocket:8765] → 웹 브라우저 (Phase 3)

Usage:
  python python/main.py                    # 일반 실행
  python python/main.py --no-calibration   # 캘리브레이션 건너뛰기
  python python/main.py --log-level DEBUG  # 디버그
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import yaml
import numpy as np

# 프로젝트 루트 추가
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from python.imu.receiver import IMUReceiver, DiscoveryResponder
from python.imu.parser import PacketParser, IMUPacket
from python.imu.calibration import BiasCalibrator, CalibrationResult, SensorBias, apply_bias
from python.fusion.filters import LowPassFilter, MadgwickFilter
from python.fusion.kinematic_chain import KinematicChain
from python.fusion.trajectory import TrajectoryEstimator
from python.fusion import quaternion as quat
from python.ws_server import WebSocketServer

logger = logging.getLogger("AirWriting")


class AirWritingPipeline:
    """
    전체 에어라이팅 파이프라인.

    센서 수신 → 파싱 → 캘리브레이션 → 필터링 → 융합 → 궤적 추정
    """

    def __init__(self, config: dict):
        self.config = config
        self._running = False

        # ── 설정 추출 ──
        sys_cfg = config.get("system", {})
        fusion_cfg = config.get("fusion", {})
        zupt_cfg = fusion_cfg.get("zupt", {})
        cal_cfg = config.get("calibration", {})

        sample_rate = sys_cfg.get("sampling_rate", 100)
        udp_port = sys_cfg.get("udp_port", 12345)
        discovery_port = sys_cfg.get("discovery_port", 12344)

        # ── IMU 수신 / 파싱 ──
        self.receiver = IMUReceiver(port=udp_port)
        self.parser = PacketParser()
        self.discovery = DiscoveryResponder(discovery_port, udp_port)

        # ── 캘리브레이션 ──
        self.calibrator = BiasCalibrator(
            duration_sec=cal_cfg.get("static_duration_sec", 3.0),
            sample_rate=sample_rate,
        )
        self._calibrated = False
        self._calibration_result = None

        # ── 필터 (각 센서별) ──
        lpf_cutoff = fusion_cfg.get("lpf_cutoff_hz", 20)
        self.lpf_s1_accel = LowPassFilter(lpf_cutoff, sample_rate)
        self.lpf_s1_gyro = LowPassFilter(lpf_cutoff, sample_rate)
        self.lpf_s2_accel = LowPassFilter(lpf_cutoff, sample_rate)
        self.lpf_s2_gyro = LowPassFilter(lpf_cutoff, sample_rate)
        self.lpf_s3_accel = LowPassFilter(lpf_cutoff, sample_rate)
        self.lpf_s3_gyro = LowPassFilter(lpf_cutoff, sample_rate)

        # ── Madgwick AHRS (각 센서별) ──
        beta = fusion_cfg.get("madgwick_beta", 0.1)
        self.madgwick_s1 = MadgwickFilter(beta, sample_rate)
        self.madgwick_s2 = MadgwickFilter(beta, sample_rate)
        self.madgwick_s3 = MadgwickFilter(beta, sample_rate)

        # ── 운동학적 사슬 ──
        skeleton_cfg = config.get("skeleton_chain", [])
        self.kinematic_chain = KinematicChain(skeleton_cfg, sample_rate)

        # ── 궤적 추정 ──
        self.trajectory = TrajectoryEstimator(
            sample_rate=sample_rate,
            zupt_enabled=zupt_cfg.get("enabled", True),
            gyro_threshold=zupt_cfg.get("gyro_threshold", 0.05),
            accel_threshold=zupt_cfg.get("accel_threshold", 0.3),
        )

        # ── 통계 ──
        self._frame_count = 0
        self._last_log_time = time.monotonic()

        # ── 추론 엔진 및 스트리밍 (Sliding Window) ──
        try:
            from python.ml.inference import InferenceEngine
            from python.ml.streaming import StreamingInference
            self.inference_engine = InferenceEngine(
                config.get("inference", {}).get("model_path", "models/airwriting_attn.onnx")
            )
            self.streamer = StreamingInference(
                engine=self.inference_engine,
                buffer_size=150,
                stride=50,
                debounce_time=0.8,
                space_timeout=1.0
            )
        except Exception as e:
            logger.warning(f"Inference engine could not be loaded: {e}")
            logger.info("학습 데이터 부재로 인해 테스트용 Dummy(가짜) 엔진을 로드합니다.")
            
            class DummyEngine:
                def predict(self, window):
                    import random
                    return {"above_threshold": True, "class": random.choice(["A", "B", "C"])}
            
            from python.ml.streaming import StreamingInference
            self.inference_engine = DummyEngine()
            self.streamer = StreamingInference(
                engine=self.inference_engine,
                buffer_size=30, # 더 빠른 더미 출력을 위해 버퍼 감소
                stride=10, 
                debounce_time=0.5,
                space_timeout=1.0
            )
        # ── WebSocket 서버 ──
        ws_port = sys_cfg.get("websocket_port", 8765)
        self.ws_server = WebSocketServer(port=ws_port, data_dir="data")
        self.ws_server.on_command = self._handle_ws_command
        self.yaw_offset = 0.0
        self._ws_loop: asyncio.AbstractEventLoop = None

        # ── 콜백 ──
        self.on_frame: callable = None  # 프레임 처리 완료 콜백

        # 스트리머 이벤트 콜백 바인딩
        if self.streamer:
            self.streamer.on_text_updated = self._on_streamer_text

    def _handle_ws_command(self, cmd: str):
        """웹 클라이언트로부터의 명령 처리."""
        if cmd == "reset_yaw":
            logger.info("🎯 Yaw reference reset by Web UI (Z key)!")
        elif cmd == "reload_model":
            if hasattr(self, 'inference_engine') and hasattr(self.inference_engine, 'reload_model'):
                self.inference_engine.reload_model()
            else:
                logger.warning("reload_model called, but inference_engine doesn't support it.")
        elif cmd == "erase_last_char":
            if self.streamer and len(self.streamer._current_sentence) > 0:
                erased = self.streamer._current_sentence[-1]
                self.streamer._current_sentence = self.streamer._current_sentence[:-1]
                logger.info(f"⌫ Erased '{erased}' → '{self.streamer._current_sentence}'")
        elif cmd == "clear_all_text":
            if self.streamer:
                self.streamer._current_sentence = ""
                self.streamer._last_emitted_char = None
                logger.info("🗑️ All text cleared")

    def _on_streamer_text(self, sentence: str, char: str):
        """스트리밍 추론 콜백: 새 글자가 인식되면 WS로 전송"""
        if self._ws_loop and self.ws_server.client_count > 0:
            msg = {
                "type": "streaming_text",
                "sentence": sentence,
                "latest_char": char
            }
            asyncio.run_coroutine_threadsafe(
                self.ws_server.broadcast(msg),
                self._ws_loop,
            )

    def _on_packet(self, data: bytes, addr: tuple):
        """UDP 패킷 수신 콜백."""
        packet = self.parser.parse(data)
        if packet is None or not packet.valid:
            return

        # ── 1. 캘리브레이션 단계 ──
        if not self._calibrated:
            self.calibrator.feed(packet)
            if self.calibrator.is_complete:
                self._calibration_result = self.calibrator.result
                self._calibrated = True
                logger.info("✅ Calibration complete! Starting pipeline.")
            else:
                progress = self.calibrator.progress
                if self._frame_count % 50 == 0:
                    logger.info(f"📐 Calibrating... {progress*100:.0f}%")
                self._frame_count += 1
            return

        # ── 2. 바이어스 보정 ──
        packet = apply_bias(packet, self._calibration_result)

        # ── 3. 저역 통과 필터링 ──
        s1_accel = self.lpf_s1_accel.update(packet.s1.accel)
        s1_gyro = self.lpf_s1_gyro.update(packet.s1.gyro)
        s2_accel = self.lpf_s2_accel.update(packet.s2.accel)
        s2_gyro = self.lpf_s2_gyro.update(packet.s2.gyro)
        s3_accel = self.lpf_s3_accel.update(packet.s3.accel)
        s3_gyro = self.lpf_s3_gyro.update(packet.s3.gyro)

        # ── 4. AHRS 쿼터니언 추정 ──
        q_s1 = self.madgwick_s1.update_imu(s1_accel, s1_gyro)
        q_s2 = self.madgwick_s2.update_imu(s2_accel, s2_gyro)
        # S3는 지자기 포함 9축
        q_s3 = self.madgwick_s3.update_marg(s3_accel, s3_gyro, packet.s3.mag)

        # ── 5. 운동학적 사슬 업데이트 ──
        self.kinematic_chain.update_joint("forearm", q_s1, s1_accel)
        self.kinematic_chain.update_joint("hand", q_s2, s2_accel)
        self.kinematic_chain.update_joint("finger", q_s3, s3_accel)

        # 체인 유효성 검증
        if not self.kinematic_chain.is_chain_valid():
            logger.warning("⚠️ Kinematic chain out of bounds — possible drift")

        # ── 6. 궤적 추정 ──
        # S3(finger)의 가속도를 월드 좌표계로 변환
        accel_world = quat.rotate_vector(s3_accel, q_s3)
        position = self.trajectory.update(accel_world, s3_gyro, packet.button)

        # ── 7. 결과 패키징 ──
        self._frame_count += 1
        frame_data = {
            "timestamp": packet.timestamp_ms,
            "frame": self._frame_count,
            "raw_sensors": {
                "s1": {
                    "ax": float(packet.s1.ax), "ay": float(packet.s1.ay), "az": float(packet.s1.az),
                    "gx": float(packet.s1.gx), "gy": float(packet.s1.gy), "gz": float(packet.s1.gz),
                },
                "s2": {
                    "ax": float(packet.s2.ax), "ay": float(packet.s2.ay), "az": float(packet.s2.az),
                    "gx": float(packet.s2.gx), "gy": float(packet.s2.gy), "gz": float(packet.s2.gz),
                },
                "s3": {
                    "ax": float(packet.s3.ax), "ay": float(packet.s3.ay), "az": float(packet.s3.az),
                    "gx": float(packet.s3.gx), "gy": float(packet.s3.gy), "gz": float(packet.s3.gz),
                    "mx": float(packet.s3.mx), "my": float(packet.s3.my), "mz": float(packet.s3.mz),
                },
            },
            "orientations": {
                "forearm": q_s1.tolist(),
                "hand": q_s2.tolist(),
                "finger": q_s3.tolist(),
            },
            "positions": {
                k: v.tolist()
                for k, v in self.kinematic_chain.get_all_positions().items()
            },
            "fingertip": self.kinematic_chain.get_fingertip_position().tolist(),
            "trajectory_position": position.tolist(),
            "button": packet.button,
            "is_writing": self.trajectory.is_writing,
        }

        # 콜백 호출
        if self.on_frame:
            self.on_frame(frame_data)

        # 스트리밍 버퍼에 프레임 전달
        if self.streamer:
            self.streamer.process_frame(frame_data)

        # WebSocket 브로드캐스트
        if self._ws_loop and self.ws_server.client_count > 0:
            asyncio.run_coroutine_threadsafe(
                self.ws_server.broadcast(frame_data),
                self._ws_loop,
            )

        # 주기적 로그
        now = time.monotonic()
        if now - self._last_log_time >= 5.0:
            self._log_stats()
            self._last_log_time = now

    def _log_stats(self):
        """주기적 통계 출력."""
        recv_stats = self.receiver.stats
        parse_stats = self.parser.stats
        euler = quat.to_euler(self.madgwick_s3.q)
        logger.info(
            f"📊 Frame #{self._frame_count} | "
            f"RX: {recv_stats['packets_received']} | "
            f"Parse OK: {parse_stats['parsed']} ERR: {parse_stats['errors']} | "
            f"S3 RPY: [{np.degrees(euler[0]):.1f}°, "
            f"{np.degrees(euler[1]):.1f}°, {np.degrees(euler[2]):.1f}°] | "
            f"Writing: {self.trajectory.is_writing}"
        )

    async def run(self, skip_calibration: bool = False):
        """파이프라인 시작."""
        self._running = True

        if skip_calibration:
            self._calibrated = True
            self._calibration_result = CalibrationResult(
                s1_bias=SensorBias(),
                s2_bias=SensorBias(),
                s3_bias=SensorBias(),
                gravity_vector=np.array([0.0, 0.0, 9.81]),
                calibrated=True,
            )
            logger.info("⏭️ Calibration skipped (zero-bias)")

        # Discovery 응답 서버 (별도 스레드)
        discovery_thread = threading.Thread(
            target=self.discovery.start, daemon=True
        )
        discovery_thread.start()

        # 패킷 수신 콜백 등록
        self.receiver.on_packet = self._on_packet

        logger.info("=" * 55)
        logger.info("  AirWriting Pipeline v4.0")
        logger.info("=" * 55)
        if not skip_calibration:
            logger.info("📐 Waiting for sensor data to start calibration...")
            logger.info("   Hold sensors still on a flat surface.")

        # WebSocket 서버 시작
        self._ws_loop = asyncio.get_event_loop()
        await self.ws_server.start()
        logger.info(f"🌐 Open web/index.html to see 3D visualization")

        try:
            await self.receiver.run()
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        """파이프라인 정지."""
        self._running = False
        self.receiver.stop()
        self.discovery.stop()
        logger.info("🛑 Pipeline stopped.")
        self._log_stats()


def load_config(config_dir: str = None) -> dict:
    """설정 파일 로드."""
    if config_dir:
        config_path = Path(config_dir) / "settings.yaml"
    else:
        config_path = ROOT / "config" / "settings.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="AirWriting Pipeline v4.0")
    ap.add_argument("--config-dir", default=None, help="Config directory")
    ap.add_argument("--no-calibration", action="store_true",
                    help="Skip calibration")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("airwriting.log", encoding="utf-8"),
        ],
    )

    config = load_config(args.config_dir)
    pipeline = AirWritingPipeline(config)

    # 프레임 로그 콜백 (디버그용)
    def print_frame(data):
        if data["frame"] % 100 == 0:
            tip = data["fingertip"]
            logger.debug(
                f"Fingertip: [{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]"
            )

    pipeline.on_frame = print_frame

    # 시그널 핸들링
    loop = asyncio.new_event_loop()

    def shutdown():
        pipeline.stop()
        loop.stop()

    signal.signal(signal.SIGINT, lambda *_: shutdown())
    signal.signal(signal.SIGTERM, lambda *_: shutdown())

    try:
        loop.run_until_complete(
            pipeline.run(skip_calibration=args.no_calibration)
        )
    except KeyboardInterrupt:
        shutdown()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
