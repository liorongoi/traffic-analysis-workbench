"""Which ONNX Runtime backend runs the models: CUDA, DirectML, Core ML or CPU (Q20).

The choice is made at start from what ONNX Runtime offers and what actually works:
a session is created with the preferred backend and the backend it really uses is
read back, because ONNX Runtime falls back to CPU on its own when a GPU library or
driver is missing. ``TWB_ACCELERATOR`` (auto, cuda, directml, coreml, cpu) overrides the
order; "auto" tries CUDA, then DirectML, then Core ML, then CPU. Core ML (Macs) runs a model
on the Apple GPU, the Neural Engine and the processor together; on "auto" it is kept only
when it is measured faster than the processor alone for that model, since Core ML cannot
take every layer of every model. The backend in use and the reason for any fallback are
recorded with every run.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
COREML = "CoreMLExecutionProvider"
NAMES = {"cuda": CUDA, "directml": DML, "coreml": COREML, "cpu": CPU}
ORDER = [CUDA, DML, COREML, CPU]
LABEL = {CUDA: "CUDA", DML: "DirectML", COREML: "Core ML", CPU: "CPU"}
# Core ML on every part of the chip (GPU, Neural Engine, processor), compiled models kept between runs.
COREML_OPTIONS = {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}
_COREML_CHOICE: dict[str, bool] = {}     # per model file: Core ML measured faster than the processor


def _l(p: str) -> str:
    return LABEL.get(p, p.replace("ExecutionProvider", ""))


def _feed(session) -> dict:
    feed = {}
    dtype_map = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                 "tensor(double)": np.float64, "tensor(uint8)": np.uint8,
                 "tensor(int8)": np.int8, "tensor(int32)": np.int32,
                 "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
    for node in session.get_inputs():
        shape = []
        for i, dim in enumerate(node.shape):
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            elif len(node.shape) == 4 and i in (2, 3):
                shape.append(640)
            else:
                shape.append(1)
        dtype = dtype_map.get(node.type)
        if dtype is None:
            raise RuntimeError(f"unsupported warm-up input type {node.type}")
        feed[node.name] = np.zeros(shape, dtype=dtype)
    if not feed:
        raise RuntimeError("model has no inputs")
    return feed


def _warmup(session) -> None:
    """Run a bounded zero-input inference to catch providers that initialize but cannot execute."""
    session.run(None, _feed(session))


def _seconds(session, runs: int = 3) -> float:
    feed = _feed(session)
    session.run(None, feed)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        t.append(time.perf_counter() - t0)
    return statistics.median(t)


class AcceleratorError(ValueError):
    pass


def wanted(setting: str | None = None) -> list[str]:
    """Backends to try, best first, for a setting of auto, cuda, directml or cpu."""
    s = (setting or os.environ.get("TWB_ACCELERATOR") or "auto").strip().lower()
    if s == "auto":
        return list(ORDER)
    if s not in NAMES:
        raise AcceleratorError(f"TWB_ACCELERATOR must be auto, cuda, directml, coreml or cpu, not '{s}'.")
    return [NAMES[s]] if s == "cpu" else [NAMES[s], CPU]


@dataclass
class SessionInfo:
    requested: list[str]
    used: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def accelerator(self) -> str:
        first = self.used[0] if self.used else CPU
        return {CUDA: "cuda", DML: "directml", COREML: "coreml"}.get(first, "cpu")

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
    if tried[0] == COREML and len(want) > 2 and _COREML_CHOICE.get(str(Path(weights).resolve())) is False:
        tried = [CPU]                 # measured before in this run: the processor alone was faster for this model
        notes.append("Core ML was not faster than the processor for this model; using CPU")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def providers(names: list[str]) -> list:
        if COREML not in names:
            return names
        cache = Path(os.environ.get("TWB_COREML_CACHE") or Path.home() / ".cache" / "lanepulse-coreml")
        try:
            cache.mkdir(parents=True, exist_ok=True)
            o = {**COREML_OPTIONS, "ModelCacheDirectory": str(cache)}
        except OSError:
            o = dict(COREML_OPTIONS)
        return [(p, o) if p == COREML else p for p in names]

    try:
        try:
            sess = ort.InferenceSession(str(weights), opts, providers=providers(tried))
        except Exception:
            if tried[0] != COREML:
                raise
            sess = ort.InferenceSession(str(weights), opts, providers=tried)   # an older Core ML without these options
        if tried[0] != CPU:
            _warmup(sess)
    except Exception as exc:          # a GPU backend that fails to start or execute: fall back to CPU
        if tried == [CPU]:
            raise
        notes.append(f"{_l(tried[0])} failed session or warm-up inference ({type(exc).__name__}: {str(exc)[:200]}); using CPU")
        sess = ort.InferenceSession(str(weights), opts, providers=[CPU])
        tried = [CPU]
    if tried[0] == COREML and len(want) > 2 and sess.get_providers()[0] == COREML:
        # "auto": Core ML only where it is faster than the processor for this model.
        key = str(Path(weights).resolve())
        if key not in _COREML_CHOICE:
            try:
                cpu = ort.InferenceSession(str(weights), opts, providers=[CPU])
                t_ml, t_cpu = _seconds(sess), _seconds(cpu)
                _COREML_CHOICE[key] = t_ml < t_cpu * 0.95
                notes.append(f"Core ML {t_ml * 1000:.0f} ms, processor alone {t_cpu * 1000:.0f} ms per run")
            except Exception as exc:
                _COREML_CHOICE[key] = False
                notes.append(f"Core ML could not be measured ({type(exc).__name__}); using CPU")
        if not _COREML_CHOICE[key]:
            notes.append("Core ML was not faster than the processor for this model; using CPU")
            sess = ort.InferenceSession(str(weights), opts, providers=[CPU])
            tried = [CPU]
    used = sess.get_providers()
    if tried[0] != CPU and used[0] != tried[0]:
        notes.append(f"{_l(tried[0])} was offered but could not be used (driver or library missing); using {_l(used[0])}")
    return sess, SessionInfo(tried, used, notes)
