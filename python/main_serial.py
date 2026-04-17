"""
main_serial.py — AirWriting Serial 모드 엔트리 포인트

USB Serial을 통해 ESP32와 통신하는 테스트용 파이프라인.
기존 main.py와 동일한 파이프라인을 사용하되, UDP 대신 Serial 수신기를 연결합니다.

Usage:
  python python/main_serial.py                        # 자동 포트 탐지
  python python/main_serial.py --port COM3             # 특정 포트 지정
  python python/main_serial.py --port COM3 --no-calibration
  python python/main_serial.py --list-ports            # 사용 가능 포트 목록
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
import time
import math
from pathlib import Path

import yaml
import numpy as np

# 프로젝트 루트 추가
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from python.imu.serial_receiver import SerialIMUReceiver, list_serial_ports
from python.imu.parser import PacketParser, IMUPacket
from python.imu.calibration import BiasCalibrator, CalibrationResult, SensorBias, apply_bias
from python.fusion.filters import LowPassFilter, MadgwickFilter, OneEuroFilter
from python.fusion.kinematic_chain import KinematicChain
from python.fusion.trajectory import TrajectoryEstimator
from python.fusion import quaternion as quat
from python.ws_server import WebSocketServer

logger = logging.getLogger("AirWriting")


class AirWritingSerialPipeline:
    """
    Serial 모드 에어라이팅 파이프라인.

    기존 AirWritingPipeline과 동일한 처리 로직을 사용하되,
    IMUReceiver 대신 SerialIMUReceiver를 사용합니다.
    """

    def __init__(self, config: dict, serial_port: str = None, baudrate: int = 921600):
        self.config = config
        self._running = False

        # ── 설정 추출 ──
        sys_cfg = config.get("system", {})
        fusion_cfg = config.get("fusion", {})
        zupt_cfg = fusion_cfg.get("zupt", {})
        cal_cfg = config.get("calibration", {})

        sample_rate = sys_cfg.get("sampling_rate", 100)

        # ── Serial 수신기 ──
        self.receiver = SerialIMUReceiver(port=serial_port, baudrate=baudrate)
        self.parser = PacketParser()

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
        # Ray casting 전용 6축 필터 (자기장 제외 → yaw 점프 방지)
        self.madgwick_s3_ray = MadgwickFilter(beta * 0.5, sample_rate)

        # 1-Euro 적응형 필터 (Jitter 제거 및 극저지연)
        self.one_euro_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0, sample_rate=sample_rate)

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

        # ── 추론 엔진 및 스트리밍 ──
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
                debounce_time=0.3
            )
        except Exception as e:
            logger.warning(f"Inference engine could not be loaded: {e}")
            logger.info("테스트용 Dummy 엔진을 로드합니다.")

            class DummyEngine:
                def predict(self, window):
                    import random
                    return {"above_threshold": True, "class": random.choice(["A", "B", "C"])}

            from python.ml.streaming import StreamingInference
            self.inference_engine = DummyEngine()
            self.streamer = StreamingInference(
                engine=self.inference_engine,
                buffer_size=30,
                stride=10,
                debounce_time=0.5
            )

        # ── WebSocket 서버 ──
        ws_port = sys_cfg.get("websocket_port", 8765)
        self.ws_server = WebSocketServer(port=ws_port, data_dir="data")
        self.ws_server.on_command = self._handle_ws_command
        self.yaw_offset = 0.0
        self._ws_loop: asyncio.AbstractEventLoop = None

        # ── Relative Projection 상태 추적 ──
        self.q_start_writing = None
        self._prev_is_writing = False

        # ── 콜백 ──
        self.on_frame: callable = None

        # 스트리머 이벤트 콜백 바인딩
        if self.streamer:
            self.streamer.on_text_updated = self._on_streamer_text

    def _handle_ws_command(self, cmd: str):
        """웹 클라이언트로부터의 명령 처리."""
        if cmd == "reset_yaw":
            self.yaw_offset = getattr(self, 'current_yaw', 0.0)
            self._trigger_super_z = True
            logger.info(f"🎯 3D Super Z-calibration triggered (UI Z key)!")
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
                # OLED에도 업데이트 전송
                display_text = self.streamer._current_sentence[-1] if self.streamer._current_sentence else ""
                self.receiver.send_to_device(f"{display_text},0.0")
        elif cmd == "clear_all_text":
            if self.streamer:
                self.streamer._current_sentence = ""
                self.streamer._last_emitted_char = None
                logger.info("🗑️ All text cleared")
                self.receiver.send_to_device(",0.0")

    def _on_streamer_text(self, sentence: str, char: str):
        """스트리밍 추론 콜백: 새 글자 인식 시 WS/OLED 전송"""
        # WebSocket으로 전송
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

        # ESP32 OLED로도 전송
        self.receiver.send_to_device(f"{char},95.0")

    def _on_packet(self, data: bytes, addr: tuple):
        """Serial 패킷 수신 콜백."""
        packet = self.parser.parse(data)
        if packet is None or not packet.valid:
            return

        # ── 1. 캘리브레이션 ──
        if not self._calibrated:
            self.calibrator.feed(packet)
            if self.calibrator.is_complete:
                self._calibration_result = self.calibrator.result
                self._calibrated = True
                logger.info("✅ Calibration complete! Starting pipeline.")
                self.receiver.send_to_device("OLED|CAL|SERIAL|DONE|100.0")
            else:
                progress = self.calibrator.progress
                if self._frame_count % 50 == 0:
                    logger.info(f"📐 Calibrating... {progress*100:.0f}%")
                self._frame_count += 1
            return

        # ── 2. 바이어스 보정 ──
        packet = apply_bias(packet, self._calibration_result)

        # ── 3. 저역 통과 필터링 ──
        # NOTE: 자동 자이로 바이어스 보정은 OP-1F 모델에서 제거됨.
        # OP-1F는 적분을 사용하지 않고 Madgwick AHRS의 gradient descent가
        # 자체적으로 자이로 드리프트를 보정하므로, 외부 바이어스 감산은
        # Madgwick의 수렴을 방해하여 오히려 비틀림을 악화시킴.
        s1_accel = self.lpf_s1_accel.update(packet.s1.accel)
        s1_gyro = self.lpf_s1_gyro.update(packet.s1.gyro)
        s2_accel = self.lpf_s2_accel.update(packet.s2.accel)
        s2_gyro = self.lpf_s2_gyro.update(packet.s2.gyro)
        s3_accel = self.lpf_s3_accel.update(packet.s3.accel)
        s3_gyro = self.lpf_s3_gyro.update(packet.s3.gyro)

        # ── 4. AHRS 쿼터니언 추정 ──  (기존 §4는 §3으로 병합)
        if self._frame_count % 100 == 0:
            logger.info(
                f"🔍 S3 raw → accel={s3_accel}, gyro={s3_gyro}, mag={packet.s3.mag}"
            )
        q_s1 = self.madgwick_s1.update_imu(s1_accel, s1_gyro)
        q_s2 = self.madgwick_s2.update_imu(s2_accel, s2_gyro)
        q_s3 = self.madgwick_s3.update_marg(s3_accel, s3_gyro, packet.s3.mag)

        # ── 5. 운동학적 사슬 업데이트 ──
        self.kinematic_chain.update_joint("forearm", q_s1, s1_accel)
        self.kinematic_chain.update_joint("hand", q_s2, s2_accel)
        self.kinematic_chain.update_joint("finger", q_s3, s3_accel)

        if not self.kinematic_chain.is_chain_valid():
            logger.warning("⚠️ Kinematic chain out of bounds — possible drift")

        # ── 6. 최첨단 모델: OP-1F (Orthographic Pointing + 1-Euro Filter) ──
        # 밀림(Sliding)을 유발하는 강제 보정 코드를 철거하고, 6축 본연의 구면 정사영 투영과
        # 수전증(Jitter)을 방지하는 1-Euro 적응형 필터를 결합합니다.

        q_s3_ray = self.madgwick_s3_ray.update_imu(s3_accel, s3_gyro)

        # 6.4 구면 정사영 투영 (Orthographic Ray Cast) & 오토 영점 캘리브레이션 (판서 모드)
        current_is_writing = (packet.button > 0)
        forward_local = np.array([0.0, -1.0, 0.0])  # 센서의 앞면

        # 사용자가 Z키(또는 리셋)를 눌러서 3D 영점 초기화를 요청한 경우
        if getattr(self, '_trigger_super_z', False):
            self.q_start_writing = q_s3_ray.copy()
            self.one_euro_filter.reset()

            # ── 화면 기저 벡터(Screen Basis) 계산 ──
            # 캘리브레이션 자세의 포인팅 방향을 기준으로,
            # 월드 수직(중력 반대)과 교차곱하여 화면의 가로/세로 축을 정의.
            # → 센서가 어떻게 기울어져 있어도 화면 좌표는 항상 수평/수직 보장.
            f_start = quat.rotate_vector(forward_local, q_s3_ray)
            world_up = np.array([0.0, 0.0, 1.0])

            screen_right = np.cross(world_up, f_start)
            norm_r = np.linalg.norm(screen_right)
            if norm_r > 1e-6:
                screen_right /= norm_r
            else:
                screen_right = np.array([1.0, 0.0, 0.0])

            screen_up = np.cross(f_start, screen_right)
            norm_u = np.linalg.norm(screen_up)
            if norm_u > 1e-6:
                screen_up /= norm_u
            else:
                screen_up = np.array([0.0, 0.0, 1.0])

            self._screen_right = screen_right
            self._screen_up = screen_up
            self._trigger_super_z = False



        if self.q_start_writing is not None:
            # ── 월드 좌표 정사영 (World-Aligned Orthographic Projection) ──
            # 현재 포인팅 방향을 월드 좌표의 화면 기저 벡터에 dot하여
            # 화면 좌우(ray_x)와 상하(ray_z)를 직접 계산.
            # 로컬 좌표계를 거치지 않으므로 센서 기울기에 의한 비틀림이 원천 차단됨.
            forward_current = quat.rotate_vector(forward_local, q_s3_ray)
            ray_x = float(np.dot(forward_current, self._screen_right))
            ray_z = float(np.dot(forward_current, self._screen_up))

        else:
            # 부팅 직후 아직 아무 영점도 안 잡았을 때 (기존 방식 1회용)
            r, p, y = quat.to_euler(q_s3_ray)
            self.current_yaw = y
            q_drift_free = quat.from_euler(r, p, y - self.yaw_offset)
            forward_world = quat.rotate_vector(forward_local, q_drift_free)
            ray_x = forward_world[0]
            ray_z = forward_world[2]

        # 1-Euro Filter 적용 (미세한 손떨림은 완전히 굳히고, 빠른 스윙은 지연 없이)
        smoothed_ray = self.one_euro_filter.update(np.array([ray_x, ray_z]))

        # 배율(Scale)을 기존 5.0에서 2.5로 절반 감소시켜 세밀한 속도 제어 확보
        ray_hit = [float(smoothed_ray[0] * 2.5), float(smoothed_ray[1] * 2.5)]

        # ── 7. 궤적 추정 (기존 호환) ──
        accel_world = quat.rotate_vector(s3_accel, q_s3)
        position = self.trajectory.update(accel_world, s3_gyro, packet.button)

        # ── 8. 결과 패키징 ──
        self._frame_count += 1
        frame_data = {
            "timestamp": packet.timestamp_ms,
            "frame": self._frame_count,
            "ray_hit": ray_hit,  # 레이저 포인터 좌표 [x, y]
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
            "input_mode": getattr(self.trajectory, "input_mode", "WRITE"),
            "is_writing": getattr(self.trajectory, "is_writing", False), # 웹 UI(대시보드) 시각화 보존을 위해 무조건 True
            "is_character_writing": getattr(self.trajectory, "is_writing", False) and getattr(self.trajectory, "input_mode", "WRITE") == "WRITE",
            "is_dragging": getattr(self.trajectory, "is_writing", False) and getattr(self.trajectory, "input_mode", "WRITE") == "MOUSE",
        }

        if self.on_frame:
            self.on_frame(frame_data)

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
            f"Sync losses: {recv_stats.get('sync_losses', 0)} | "
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

        # 패킷 수신 콜백 등록
        self.receiver.on_packet = self._on_packet

        logger.info("=" * 55)
        logger.info("  AirWriting Pipeline v4.0 — Serial Mode")
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
    ap = argparse.ArgumentParser(description="AirWriting Pipeline v4.0 — Serial Mode")
    ap.add_argument("--config-dir", default=None, help="Config directory")
    ap.add_argument("--port", default=None, help="Serial port (e.g. COM3, /dev/ttyUSB0). Auto-detect if not set.")
    ap.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate (default: 921600)")
    ap.add_argument("--no-calibration", action="store_true", help="Skip calibration")
    ap.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("airwriting_serial.log", encoding="utf-8"),
        ],
    )

    if args.list_ports:
        ports = list_serial_ports()
        if ports:
            print("사용 가능한 시리얼 포트:")
            for p in ports:
                print(f"  {p}")
        else:
            print("사용 가능한 시리얼 포트가 없습니다.")
        return

    config = load_config(args.config_dir)
    pipeline = AirWritingSerialPipeline(
        config,
        serial_port=args.port,
        baudrate=args.baudrate,
    )

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
