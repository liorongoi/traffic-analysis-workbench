"""Which ONNX Runtime backend runs the models: CUDA, DirectML or CPU (Q20).

The choice is made at start from what ONNX Runtime offers and what actually works:
a session is created with the preferred backend and the backend it really uses is
read back, because ONNX Runtime falls back to CPU on its own when a GPU library or
driver is missing. ``TWB_ACCELERATOR`` (auto, cuda, directml, cpu) overrides the
order; "auto" tries CUDA, then DirectML, then CPU. The backend in use and the reason
for any fallback are recorded with every run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
NAMES = {"cuda": CUDA, "directml": DML, "cpu": CPU}
ORDER = [CUDA, DML, CPU]
LABEL = {CUDA: "CUDA", DML: "DirectML", CPU: "CPU"}


def _l(p: str) -> str:
    return LABEL.get(p, p.replace("ExecutionProvider", ""))


class AcceleratorError(ValueError):
    pass


def wanted(setting: str | None = None) -> list[str]:
    """Backends to try, best first, for a setting of auto, cuda, directml or cpu."""
    s = (setting or os.environ.get("TWB_ACCELERATOR") or "auto").strip().lower()
    if s == "auto":
        return list(ORDER)
    if s not in NAMES:
        raise AcceleratorError(f"TWB_ACCELERATOR must be auto, cuda, directml or cpu, not '{s}'.")
    return [NAMES[s]] if s == "cpu" else [NAMES[s], CPU]


@dataclass
class SessionInfo:
    requested: list[str]
    used: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        first = self.used[0] if self.used else CPU
        return {CUDA: "cuda", DML: "directml"}.get(first, "cpu")

    def record(self) -> dict:
        return {"accelerator": self.accelerator, "requested": self.requested, "used": self.used, "notes": self.notes}


def make_session(weights: Path | str, setting: str | None = None, available: list[str] | None = None):
    """Create an inference session on the best working backend. Returns (session, SessionInfo)."""
    import onnxruntime as ort

    avail = available if available is not None else ort.get_available_providers()
    if CUDA in avail and available is None and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()        # CUDA and cuDNN libraries installed as Python packages (the "cuda" bundle)
        except Exception:
            pass
    want = wanted(setting)
    notes = []
    tried = [p for p in want if p in avail]
    for p in want:
        if p not in avail and p != CPU:
            notes.append(f"{_l(p)} is not in this ONNX Runtime build")
    if CPU not in tried:
        tried.append(CPU)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(str(weights), opts, providers=tried)
    except Exception as exc:          # a GPU backend that fails to start: fall back to CPU, and say so
        if tried == [CPU]:
            raise
        notes.append(f"{_l(tried[0])} failed to start ({type(exc).__name__}: {str(exc)[:200]}); using CPU")
        sess = ort.InferenceSession(str(weights), opts, providers=[CPU])
        tried = [CPU]
    used = sess.get_providers()
    if tried[0] != CPU and used[0] != tried[0]:
        notes.append(f"{_l(tried[0])} was offered but could not be used (driver or library missing); using {_l(used[0])}")
    return sess, SessionInfo(tried, used, notes)
