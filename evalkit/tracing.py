"""Minimal tracing: one JSON line per LLM call, so you can debug a bad answer after the fact."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, case_id: str, step: str, latency_ms: float, input_tokens: int = 0,
               output_tokens: int = 0, tool_calls: list[str] | None = None, error: str | None = None) -> None:
        event = {
            "run_id": self.run_id,
            "ts": time.time(),
            "case_id": case_id,
            "step": step,                      # "answer" or "judge"
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": tool_calls or [],
            "error": error,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def read_trace(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
