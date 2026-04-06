from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class _DllmTraceCollector:
    def __init__(self) -> None:
        self.base_path = os.getenv("SGLANG_DLLM_TRACE_PATH")
        self.enabled = bool(self.base_path)
        self._lock = threading.Lock()
        self._fh = None
        self._path = None
        self._t0 = time.perf_counter()

    def _ensure_file(self) -> None:
        if not self.enabled or self._fh is not None:
            return

        path = Path(self.base_path)
        if path.suffix:
            out_path = path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")
        else:
            out_path = Path(f"{path}.{os.getpid()}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out_path.open("a", encoding="utf-8")
        self._path = str(out_path)

    @property
    def output_path(self) -> str | None:
        return self._path

    def record(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._ensure_file()
            if self._fh is None:
                return

            record = {
                "event": event,
                "pid": os.getpid(),
                "ts_perf_s": time.perf_counter() - self._t0,
                "ts_unix_s": time.time(),
                **payload,
            }
            self._fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


TRACE_COLLECTOR = _DllmTraceCollector()


def record_dllm_event(event: str, **payload: Any) -> None:
    TRACE_COLLECTOR.record(event, **payload)
