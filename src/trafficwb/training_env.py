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
        yw = [s for s in tr.YOLOX_SIZES if tr.find_coco_weights(s) is not None]
        out = {"state": "ready", "device_type": dev.type, "device": tr.describe_device(dev),
               "coco_weights": "s" in yw, "yolox_weights": yw, "yolox_sizes": list(tr.YOLOX_SIZES),
               **_rfdetr(),
               "detr": _detr(),
               # suggested rounds: a GPU does 40 in an hour or two; a CPU needs a night for about 20
               "suggested_epochs": 40 if dev.type in ("cuda", "mps") else 20}
        out["recommended"] = recommended(out)
        return out
    except Exception as exc:                       # a broken PyTorch must never break the learning page
        return {"state": "unavailable", "reason": f"PyTorch does not start here: {type(exc).__name__}: {exc}"}


def recommended(env: dict) -> dict:
    """The size of each detector a training starts from on this computer, among the starting weights shipped here
    (hardware.levels: the GPU's memory, an Apple GPU, or the processor), and why: {"why", "sizes": {family: size}}."""
    from . import hardware
    lv = hardware.levels()["train"]
    have = {"yolox": env.get("yolox_weights") or [], "rf-detr": env.get("rfdetr_weights") or [],
            **{a: v.get("weights") or [] for a, v in (env.get("detr") or {}).items()}}
    return {"why": lv[1], "level": lv[0], "sizes": {f: hardware.pick(f, lv[0], w) for f, w in have.items()}}


def _rfdetr() -> dict:
    """Can RF-DETR be trained here too (the rfdetr package), and from which released sizes?"""
    try:
        from . import rfdetr_train as rt
        ok = rt.available()
        return {"rfdetr": ok["ok"], "rfdetr_reason": ok["reason"],
                "rfdetr_weights": [s for s in rt.SIZES if rt.find_coco_weights(s) is not None] if ok["ok"] else []}
    except Exception as exc:                       # never break the learning page
        return {"rfdetr": False, "rfdetr_reason": f"RF-DETR does not start here: {type(exc).__name__}: {exc}",
                "rfdetr_weights": []}


def _detr() -> dict:
    """D-FINE and RT-DETRv2 (detr_train.py): can they be trained here, and from which shipped released sizes?
    {arch: {"title", "ok", "reason", "sizes" (all), "weights" (shipped here), "default"}}."""
    try:
        from . import detr_train as dt
        ok = dt.available()
        return {a: {"title": v["title"], "ok": ok["ok"], "reason": ok["reason"], "sizes": list(v["sizes"]),
                    "default": v["default"],
                    "weights": [s for s in v["sizes"] if dt.find_coco_weights(a, s) is not None] if ok["ok"] else []}
                for a, v in dt.ARCHS.items()}
    except Exception as exc:                       # never break the learning page
        return {a: {"title": t, "ok": False, "reason": f"{t} does not start here: {type(exc).__name__}: {exc}",
                    "sizes": [], "default": None, "weights": []}
                for a, t in (("d-fine", "D-FINE"), ("rt-detr", "RT-DETRv2"))}


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
