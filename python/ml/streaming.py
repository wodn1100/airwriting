"""
streaming.py — 연속 문장 스트리밍 인식 (Sliding Window) 로직

이 모듈은 모션의 연속 스트림을 받아 주기적으로(스트라이드) 추론을 실행하고,
문장을 조립하며, 띄어쓰기 및 중복 글자 필터링(Debouncing)을 관리합니다.
"""

import time
import logging
from typing import Optional, List, Callable

import numpy as np

logger = logging.getLogger(__name__)

class StreamingInference:
    def __init__(
        self,
        engine,
        buffer_size: int = 150,
        stride: int = 50,
        debounce_time: float = 0.8,
        char_timeout: float = 0.6,
        space_timeout: float = 2.0,
    ):
        """
        Args:
            engine: InferenceEngine 인스턴스 (predict 메서드 보유)
            buffer_size: 추론에 필요한 프레임 수 (Window 크기)
            stride: 추론 시도 주기 (프레임 단위)
            debounce_time: 동일 문자의 중복 출력을 무시하는 최소 시간 (초)
            space_timeout: is_writing=False가 유지되면 띄어쓰기로 판정할 시간 (초)
        """
        self.engine = engine
        self.buffer_size = buffer_size
        self.stride = stride
        self.debounce_time = debounce_time
        self.char_timeout = char_timeout
        self.space_timeout = space_timeout

        self._buffer: List[np.ndarray] = []
        self._frame_count = 0
        self._last_predict_count = 0

        # State tracking
        self._last_emitted_char: Optional[str] = None
        self._last_emit_time: float = 0.0
        self._last_writing_time: float = time.time()
        self._space_emitted = False

        self.on_text_updated: Optional[Callable[[str, str], None]] = None
        self._current_sentence = ""

    def _extract_vector(self, frame_data: dict) -> np.ndarray:
        """프레임에서 28축 벡터를 추출합니다."""
        vec = []
        if "raw_sensors" in frame_data:
            for key in ["s1", "s2", "s3"]:
                s = frame_data["raw_sensors"].get(key, {})
                vec.extend([
                    s.get("ax", 0), s.get("ay", 0), s.get("az", 0),
                    s.get("gx", 0), s.get("gy", 0), s.get("gz", 0),
                ])
            # S3 mag
            s3 = frame_data["raw_sensors"].get("s3", {})
            vec.extend([s3.get("mx", 0), s3.get("my", 0), s3.get("mz", 0)])
        else:
            vec.extend([0]*21)

        orientations = frame_data.get("orientations", {})
        finger_q = orientations.get("finger", [1,0,0,0])
        vec.extend(finger_q)

        tip = frame_data.get("fingertip", [0,0,0])
        vec.extend(tip[:3])

        return np.array(vec, dtype=np.float32)

    def process_frame(self, frame_data: dict):
        """매 프레임 호출되어 텍스트 스트리밍을 제어합니다."""
        # 안드로이드/마우스 모드가 아닌 '글쓰기 모드' 일때만 버퍼에 프레임 수집
        is_writing = frame_data.get("is_character_writing", frame_data.get("is_writing", False))
        now = time.time()

        if is_writing:
            self._last_writing_time = now
            self._space_emitted = False
            self._frame_count += 1

            vec = self._extract_vector(frame_data)
            self._buffer.append(vec)

            # NOTE: 여기서 섣불리(슬라이딩 윈도우) 예측하지 않고 인텐트(버튼 뗄 때) 기반으로 변경합니다.
            # 용량 초과 시 과거 데이터 삭제 (메모리 폭발 방지용 하드 리밋, 150프레임에서 1000프레임으로 넉넉히 상향)
            if len(self._buffer) > 1000:
                self._buffer.pop(0)

        else:    # 글씨를 안 쓰고 있을 때 타임아웃 판정
            duration_since_writing = now - self._last_writing_time
            
            # 1. 단일 글자 완성 판정 (char_timeout)
            if duration_since_writing > self.char_timeout and len(self._buffer) > 0:
                if len(self._buffer) > 25:  # 더블클릭 연타(보통 10~15프레임)가 글자로 인식되는 것을 원천 차단
                    self._run_inference()
                self._buffer.clear() # 추론 완료 또는 노이즈면 버퍼 비움

            # 2. 띄어쓰기 판정 (space_timeout)
            if duration_since_writing > self.space_timeout and not self._space_emitted:
                if len(self._current_sentence) > 0 and not self._current_sentence.endswith(" "):
                    self._emit_char(" ")
                self._space_emitted = True
                self._last_emitted_char = None # 새 단어이므로 중복방지 해제

    def _run_inference(self):
        """엔진을 돌려 문자를 가져오고 디바운싱합니다."""
        if not self.engine or len(self._buffer) == 0:
            return

        window = np.vstack(self._buffer)
        result = self.engine.predict(window)

        if result.get("above_threshold") and result.get("class"):
            char = result["class"]
            now = time.time()

            # Debouncing: 동일 글자가 짧은 시간 내 다시 인식되면 무시
            is_duplicate = (char == self._last_emitted_char) and ((now - self._last_emit_time) < self.debounce_time)
            
            if not is_duplicate:
                self._emit_char(char)
                self._last_emitted_char = char
                self._last_emit_time = now

    def _emit_char(self, char: str):
        if char == "<ERASE>":
            if len(self._current_sentence) > 0:
                self._current_sentence = self._current_sentence[:-1]
            logger.info(f"🔙 Stream Erased -> Sentence: '{self._current_sentence}'")
        else:
            self._current_sentence += char
            logger.info(f"📝 Stream Emitted: '{char}' -> Sentence: '{self._current_sentence}'")
            
        if self.on_text_updated:
            self.on_text_updated(self._current_sentence, char)

    def reset(self):
        self._buffer.clear()
        self._current_sentence = ""
        self._last_emitted_char = None
        self._space_emitted = False
        if self.on_text_updated:
            self.on_text_updated("", "")
