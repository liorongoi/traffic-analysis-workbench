"""Can this installation train the vehicle detector, and on what? (for the learning page)

Importing PyTorch takes a few seconds and a few hundred MB, so it happens once, in a background thread, the first
time the learning page asks; until then the answer is "checking".
"""
from __future__ import annotations

import importlib.util
import threading

_lock = threading.Lock()
_state: dict | None = None
_thread: threading.Thread | None = None


def _probe() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"state": "unavailable", "reason": "PyTorch is not part of this installation. Install the current LanePulse "
                "version to train here."}
    try:
        from . import train as tr
        dev = tr.pick_device("auto")
        return {"state": "ready", "device_type": dev.type, "device": tr.describe_device(dev),
                "coco_weights": tr.find_coco_weights() is not None,
                # suggested rounds: a GPU does 40 in an hour or two; a CPU needs a night for about 20
                "suggested_epochs": 40 if dev.type in ("cuda", "mps") else 20}
    except Exception as exc:                       # a broken PyTorch must never break the learning page
        return {"state": "unavailable", "reason": f"PyTorch does not start here: {type(exc).__name__}: {exc}"}


def _run() -> None:
    global _state
    r = _probe()
    with _lock:
        _state = r


def status(wait: bool = False) -> dict:
    global _thread
    with _lock:
        if _state is not None:
            return dict(_state)
        if _thread is None:
            _thread = threading.Thread(target=_run, name="training-probe", daemon=True)
            _thread.start()
        t = _thread
    if wait:
        t.join()
        return status()
    return {"state": "checking"}


def reset() -> None:
    """Tests only."""
    global _state, _thread
    with _lock:
        _state, _thread = None, None
