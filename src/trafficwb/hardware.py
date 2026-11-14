"""What this computer can run and train, and which size of each fast detector suits it.

Every size of YOLOX, RF-DETR, D-FINE and RT-DETRv2 is installed with LanePulse. Which one runs (detection) and which
one a training starts from is chosen here from the hardware, unless the operator chose a size (``choices``):

- detection runs in ONNX Runtime: on an NVIDIA GPU with CUDA (its memory decides), on DirectML (a GPU of unknown
  memory: a middle size), or on the processor (its threads and memory; an Apple Mac runs detection here too).
- training runs in PyTorch: on an NVIDIA GPU (its memory), an Apple GPU (Metal; the Mac's memory, which the GPU
  shares), or the processor (the smallest sizes; slow).

The level (0 slowest .. 4 strongest) maps to one size per detector (SIZES). If that size is not installed, the
nearest smaller one is used, then the nearest larger one.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

FAMILIES = ("yolox", "rf-detr", "d-fine", "rt-detr")
TITLES = {"yolox": "YOLOX", "rf-detr": "RF-DETR", "d-fine": "D-FINE", "rt-detr": "RT-DETRv2"}
# Sizes from smallest to largest, and the size for each level 0..4.
ORDER = {"yolox": ["nano", "tiny", "s", "m", "l", "x"], "rf-detr": ["nano", "small", "medium"],
         "d-fine": ["nano", "small", "medium", "large", "xlarge"], "rt-detr": ["r18", "r34", "r50", "r101"]}
SIZES = {"yolox": ["tiny", "s", "m", "l", "x"], "rf-detr": ["nano", "small", "medium", "medium", "medium"],
         "d-fine": ["nano", "small", "medium", "large", "xlarge"], "rt-detr": ["r18", "r18", "r34", "r50", "r101"]}
TRAINABLE = {"yolox": ["tiny", "s", "m", "l", "x"]}      # YOLOX-nano (depthwise layers) runs but is not trained here
CHOICES_FILE = "detector_sizes.json"


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = M()
            m.l = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2 ** 30, 1)
        except Exception:
            return None
    return None


def _nvidia() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        name, _, mem = line.rpartition(",")
        try:
            gpus.append({"name": name.strip(), "memory_gb": round(float(mem) / 1024, 1)})
        except ValueError:
            continue
    return gpus


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


@lru_cache(maxsize=1)
def detect() -> dict:
    """The hardware as LanePulse sees it (read once per start)."""
    apple = sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")
    return {"system": platform.system(), "machine": platform.machine(), "cpu_threads": os.cpu_count() or 1,
            "memory_gb": _ram_gb(), "nvidia": _nvidia(), "apple_silicon": apple, "onnx": _onnx_providers()}


def levels(hw: dict | None = None) -> dict:
    """{"run": (level, why), "train": (level, why)} for this computer."""
    hw = hw or detect()
    ram = hw.get("memory_gb") or 8
    threads = hw.get("cpu_threads") or 1
    gpu = max((g["memory_gb"] for g in hw.get("nvidia") or []), default=0)
    gname = next((g["name"] for g in hw.get("nvidia") or [] if g["memory_gb"] == gpu), "NVIDIA GPU")
    cpu = (0, f"the processor ({threads} threads, {ram:g} GB)") if threads < 8 or ram < 16 else \
        (1, f"the processor ({threads} threads, {ram:g} GB)")
    if gpu and "CUDAExecutionProvider" in (hw.get("onnx") or []):
        run = (1 if gpu < 4 else 2 if gpu < 8 else 3 if gpu < 16 else 4, f"{gname} ({gpu:g} GB)")
    elif "DmlExecutionProvider" in (hw.get("onnx") or []):
        run = (2, "the graphics card (DirectML)")
    else:
        run = cpu
    if gpu:
        train = (1 if gpu < 6 else 2 if gpu < 12 else 3 if gpu < 20 else 4, f"{gname} ({gpu:g} GB)")
    elif hw.get("apple_silicon"):
        train = (1 if ram < 16 else 2 if ram < 32 else 3, f"the Apple GPU ({ram:g} GB shared)")
    else:
        train = (0, cpu[1])
    return {"run": run, "train": train}


def pick(family: str, level: int, installed: list[str]) -> str | None:
    """The size for a level among those installed: that one, else the nearest smaller, else the nearest larger."""
    want = SIZES[family][max(0, min(level, 4))]
    order = ORDER[family]
    if want in installed:
        return want
    i = order.index(want)
    smaller = [s for s in reversed(order[:i]) if s in installed]
    larger = [s for s in order[i + 1:] if s in installed]
    return (smaller or larger or [None])[0]


def choices(data_root: Path | str | None) -> dict:
    """The operator's own choices ({family: size}); a family not named is automatic."""
    if data_root is None:
        return {}
    try:
        data = json.loads((Path(data_root) / CHOICES_FILE).read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in FAMILIES and v in ORDER[k]}
    except (OSError, ValueError):
        return {}


def set_choice(data_root: Path | str, family: str, size: str | None) -> dict:
    """Choose a size for a family, or None for automatic."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown detector {family}.")
    if size is not None and size not in ORDER[family]:
        raise ValueError(f"{TITLES[family]} comes in {', '.join(ORDER[family])}.")
    cur = choices(data_root)
    if size is None:
        cur.pop(family, None)
    else:
        cur[family] = size
    p = Path(data_root) / CHOICES_FILE
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return cur


def installed_runnable(models_root: Path | None = None) -> dict[str, list[str]]:
    """Released detector packs installed here, by family: yolox-s/0.1.1rc0, rf-detr-small/..., d-fine-small/coco."""
    from . import models
    names = {m["name"] for m in models.installed("vehicle_detection", models_root)}
    return {f: [s for s in ORDER[f] if f"{f}-{s}" in names] for f in FAMILIES}


def run_size(family: str, data_root: Path | str | None = None, models_root: Path | None = None) -> str | None:
    """The size of a family that detection uses here: the operator's choice if installed, else by the hardware."""
    have = installed_runnable(models_root)[family]
    chosen = choices(data_root).get(family)
    if chosen in have:
        return chosen
    return pick(family, levels()["run"][0], have)


def summary(data_root: Path | str | None = None, models_root: Path | None = None,
            trainable: dict[str, list[str]] | None = None) -> dict:
    """For the screens: the hardware, and per family the sizes installed, the size chosen automatically for detection
    and for training, and the operator's choice."""
    hw = detect()
    lv = levels(hw)
    run_have = installed_runnable(models_root)
    chosen = choices(data_root)
    fams = {}
    for f in FAMILIES:
        tr_have = (trainable or {}).get(f, [])
        auto_run = pick(f, lv["run"][0], run_have[f])
        fams[f] = {"title": TITLES[f], "sizes": ORDER[f], "installed": run_have[f], "auto": auto_run,
                   "chosen": chosen.get(f), "runs": chosen.get(f) if chosen.get(f) in run_have[f] else auto_run,
                   "trainable": tr_have, "train_auto": pick(f, lv["train"][0], tr_have)}
    return {"hardware": hw, "run_on": lv["run"][1], "train_on": lv["train"][1], "run_level": lv["run"][0],
            "train_level": lv["train"][0], "families": fams}


def run_pack(family: str, data_root: Path | str | None = None, models_root: Path | None = None):
    """The installed pack of a family that detection uses here (run_size), or None when none of its sizes is installed."""
    from . import models
    size = run_size(family, data_root, models_root)
    if size is None:
        return None
    try:
        return models.find(f"{family}-{size}", root=models_root)
    except models.ModelPackError:
        return None


def yolox_pack(data_root: Path | str | None = None, models_root: Path | None = None):
    """The built-in YOLOX a mission runs when nothing is promoted: the size for this computer, else YOLOX-s."""
    from . import models
    return run_pack("yolox", data_root, models_root) or models.find("yolox-s", root=models_root)
