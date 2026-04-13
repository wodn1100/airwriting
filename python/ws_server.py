"""
ws_server.py — WebSocket 서버 + 학습 파이프라인 명령 처리

기능:
1. 실시간 프레임 데이터 브로드캐스트 (Python → 웹)
2. 클라이언트 명령 수신 (웹 → Python):
   - start_recording: 스트로크 기록 시작
   - stop_recording: 기록 종료 + 데이터셋 저장
   - start_training: 모델 학습 (3초)
   - get_sample_count: 현재 수집된 샘플 수
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Set, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import serve as ws_serve
    HAS_WS = True
except ImportError:
    HAS_WS = False
    logger.warning("websockets not installed. pip install websockets")


class WebSocketServer:
    """
    WebSocket 서버 + 반자동화 학습 파이프라인.
    """

    def __init__(self, port: int = 8765, data_dir: str = "data"):
        if not HAS_WS:
            raise ImportError("websockets required. pip install websockets")
        self.port = port
        self._clients: Set = set()
        self._server = None
        self._running = False

        # 학습 파이프라인 상태
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._recording = False
        self._record_label: str = ""
        self._record_frames: list = []

        # 데이터셋 ( lazy load )
        self._dataset = None

        # 콜백: 메인 파이프라인에서 호출
        self.on_command: Optional[callable] = None

    def _get_dataset(self):
        """데이터셋 lazy 로드."""
        if self._dataset is None:
            from python.ml.dataset import AirWritingDataset
            self._dataset = AirWritingDataset(data_dir=str(self.data_dir))
            self._dataset.load()  # 기존 데이터 로드 시도
        return self._dataset

    async def _handler(self, websocket):
        """클라이언트 연결 핸들러."""
        self._clients.add(websocket)
        addr = websocket.remote_address
        logger.info(f"🌐 Client connected: {addr} (total: {len(self._clients)})")

        try:
            async for message in websocket:
                await self._handle_command(websocket, message)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"🌐 Client disconnected: {addr} (total: {len(self._clients)})")

    async def _handle_command(self, websocket, message: str):
        """클라이언트 → 서버 명령 처리."""
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            return

        cmd_type = cmd.get("type", "")

        if cmd_type == "start_recording":
            self._recording = True
            self._record_label = cmd.get("label", "?")
            self._record_frames.clear()
            logger.info(f"⏺ Recording started: label='{self._record_label}'")

        elif cmd_type == "stop_recording":
            self._recording = False
            if len(self._record_frames) >= 5:
                # 스트로크를 데이터셋에 저장
                stroke = np.array(self._record_frames, dtype=np.float32)
                dataset = self._get_dataset()
                dataset.add_sample(stroke, self._record_label)
                dataset.save()

                # 개별 스트로크도 JSON 메타로 저장
                stroke_meta = {
                    "label": self._record_label,
                    "frames": len(self._record_frames),
                    "channels": stroke.shape[1] if len(stroke.shape) > 1 else 0,
                    "timestamp": time.time(),
                }
                meta_path = self.data_dir / "strokes_log.jsonl"
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stroke_meta, ensure_ascii=False) + "\n")

                logger.info(
                    f"💾 Stroke saved: '{self._record_label}' "
                    f"({len(self._record_frames)} frames)"
                )
            else:
                logger.warning(f"Stroke too short: {len(self._record_frames)} frames")

            self._record_frames.clear()
            await self._send_sample_count(websocket)

        elif cmd_type == "record_frame":
            if self._recording:
                frame = cmd.get("frame", {})
                vec = self._extract_train_vector(frame)
                self._record_frames.append(vec)

        elif cmd_type == "start_training":
            duration = cmd.get("duration_sec", 3)
            logger.info(f"🧠 Training requested ({duration}s)")
            main_loop = asyncio.get_event_loop()
            
            # 별도 스레드에서 학습 실행
            thread = threading.Thread(
                target=self._run_training,
                args=(websocket, duration, main_loop),
                daemon=True,
            )
            thread.start()

        elif cmd_type == "get_sample_count":
            await self._send_sample_count(websocket)

        elif cmd_type == "reset_yaw":
            if self.on_command:
                self.on_command("reset_yaw")

        elif cmd_type == "erase_last_char":
            logger.info("⌫ Erase last char requested")
            if self.on_command:
                self.on_command("erase_last_char")

        elif cmd_type == "clear_all_text":
            logger.info("🗑️ Clear all text requested")
            if self.on_command:
                self.on_command("clear_all_text")

        elif cmd_type == "get_dataset_info":
            await self._send_dataset_info(websocket)

    def _extract_train_vector(self, frame: dict) -> list:
        """프레임에서 학습 벡터 추출 (dataset.py와 동일 구조)."""
        vec = []
        # S1/S2/S3 raw (18축)
        if "raw_sensors" in frame:
            for key in ["s1", "s2", "s3"]:
                s = frame["raw_sensors"].get(key, {})
                vec.extend([
                    s.get("ax", 0), s.get("ay", 0), s.get("az", 0),
                    s.get("gx", 0), s.get("gy", 0), s.get("gz", 0),
                ])
            # S3 mag (3축)
            s3 = frame["raw_sensors"].get("s3", {})
            vec.extend([s3.get("mx", 0), s3.get("my", 0), s3.get("mz", 0)])
        else:
            vec.extend([0] * 21)

        # 쿼터니언 (4축)
        orientations = frame.get("orientations", {})
        finger_q = orientations.get("finger", [1, 0, 0, 0])
        vec.extend(finger_q)

        # fingertip (3축)
        tip = frame.get("fingertip", [0, 0, 0])
        vec.extend(tip[:3])

        return vec  # 28축

    def _run_training(self, websocket, duration_sec: float, main_loop=None):
        """학습 실행 (별도 스레드)."""
        try:
            dataset = self._get_dataset()
            if dataset.num_samples < 2 or dataset.num_classes < 2:
                err_msg = f"Not enough data: {dataset.num_samples} samples, {dataset.num_classes} classes (need ≥2 each)"
                logger.warning(err_msg)
                if main_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast({"type": "training_error", "message": err_msg}), main_loop
                    )
                return

            # PyTorch 사용 가능한지 확인
            try:
                import torch
            except ImportError:
                logger.error("PyTorch not installed. pip install torch")
                return

            from python.ml.train import train_model
            start = time.time()

            # 콜백을 통해 매 에포크의 통계를 웹 클라이언트로 브로드캐스트
            def send_epoch_metrics(ep, epochs, loss, t_acc, v_acc, class_metrics):
                msg = {
                    "type": "training_metrics_per_class",
                    "epoch": ep,
                    "total_epochs": epochs,
                    "loss": loss,
                    "train_acc": t_acc,
                    "val_acc": v_acc,
                    "classes": class_metrics
                }
                if main_loop:
                    asyncio.run_coroutine_threadsafe(self.broadcast(msg), main_loop)

            # 학습 (duration_sec 동안 — epoch 수를 시간으로 제한)
            # 3초면 약 20-50 epoch 가능 (데이터 크기에 따라)
            best_acc = train_model(
                data_dir=str(self.data_dir),
                epochs=50,
                batch_size=16,
                lr=0.001,
                model_save_dir="models",
                epoch_callback=send_epoch_metrics
            )

            # train_model이 None을 반환하는 실패 케이스 대비
            best_acc = best_acc if best_acc is not None else 0.0

            elapsed = time.time() - start
            logger.info(f"✅ Training completed in {elapsed:.1f}s")

            # 결과를 클라이언트에 전송
            # (asyncio loop에서 실행해야 함)
            if main_loop:
                asyncio.run_coroutine_threadsafe(
                    self.broadcast({
                        "type": "training_complete",
                        "accuracy": best_acc,
                        "elapsed_sec": elapsed,
                    }),
                    main_loop,
                )
                  
            # 🧠 백엔드 파이썬 엔진에서 새 모델을 즉각 Hot-Reload 하도록 명령!
            if self.on_command:
                self.on_command("reload_model")

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)

    async def _send_sample_count(self, websocket):
        """현재 샘플 수를 클라이언트에 전송."""
        dataset = self._get_dataset()
        try:
            await websocket.send(json.dumps({
                "type": "sample_count",
                "count": dataset.num_samples,
                "classes": dataset.num_classes,
                "summary": dataset.summary(),
            }))
        except Exception:
            pass

    async def _send_dataset_info(self, websocket):
        """Dashboard용 per-class 데이터셋 정보 전송."""
        dataset = self._get_dataset()
        try:
            # per-class 카운트 계산
            class_counts = {}
            for label_idx in dataset.y:
                name = dataset.classes[label_idx]
                class_counts[name] = class_counts.get(name, 0) + 1

            classes_list = [
                {"name": name, "count": count}
                for name, count in sorted(class_counts.items())
            ]

            await websocket.send(json.dumps({
                "type": "dataset_info",
                "total": dataset.num_samples,
                "num_classes": dataset.num_classes,
                "classes": classes_list,
            }))
            logger.info(f"📊 Dataset info sent: {dataset.num_samples} samples, {dataset.num_classes} classes")
        except Exception:
            pass

    async def start(self):
        """서버 시작."""
        self._server = await ws_serve(self._handler, "0.0.0.0", self.port)
        self._running = True
        logger.info(f"🌐 WebSocket server on ws://0.0.0.0:{self.port}")

    async def broadcast(self, data: dict):
        """모든 클라이언트에 브로드캐스트."""
        if not self._clients:
            return
        message = json.dumps(data, default=_json_serialize)
        disconnected = set()
        for client in self._clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
        self._clients -= disconnected

    async def stop(self):
        """서버 종료."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def client_count(self) -> int:
        return len(self._clients)


def _json_serialize(obj):
    """NumPy 타입 JSON 직렬화."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
