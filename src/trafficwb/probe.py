"""Hardware probe.

Reports what the machine actually has. Nothing here assumes a GPU works: a
backend is listed as usable only when it is verified (Phase 0 only lists what
is installed; session-level verification with a tiny model comes with the
inference worker).

Usage: trafficwb-probe [--paths D:\\data E:\\exports] [--json out.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

GIB = 1024**3

# RAM tiers from the design. The cut-off between tiers is a proposal, not a measured value.
CONSERVATIVE_MAX_GIB = 24


def _ram_bytes() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except ImportError:
        pass
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return stat.ullTotalPhys, stat.ullAvailPhys
        return None, None
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) * 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
            return total, None
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None, None


def _cpu() -> dict[str, Any]:
    out: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "flags": None,
    }
    try:
        import psutil  # type: ignore

        out["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    out["flags"] = sorted(flags & {"avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "fma", "f16c"})
                    break
    except OSError:
        pass
    return out


def _nvidia() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    q = "name,memory.total,memory.free,driver_version,compute_cap"
    try:
        r = subprocess.run(
            [exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": f"nvidia-smi failed: {exc}"}]
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "vendor": "NVIDIA",
                    "name": parts[0],
                    "vram_total_mib": _int(parts[1]),
                    "vram_free_mib": _int(parts[2]),
                    "driver": parts[3],
                    "compute_capability": parts[4] if len(parts) > 4 else None,
                }
            )
    if r.returncode != 0 and not gpus:
        gpus.append({"error": (r.stderr or r.stdout).strip()[:300]})
    return gpus


def _windows_adapters() -> list[str]:
    if sys.platform != "win32":
        return []
    cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _onnxruntime() -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return {"installed": False, "available_providers": [], "verified_providers": []}
    return {
        "installed": True,
        "version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "verified_providers": [],  # filled by the session test in the inference worker (not in Phase 0)
    }


def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except ValueError:
        return None


def _disks(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            out.append({"path": p, "total_bytes": u.total, "free_bytes": u.free})
        except OSError as exc:
            out.append({"path": p, "error": str(exc)})
    return out


def suggest_profile(ram_total: int | None, gpus: list[dict[str, Any]], ort: dict[str, Any]) -> dict[str, Any]:
    """RAM tier and accelerator are chosen independently."""
    if ram_total is None:
        ram_tier = "unknown"
    elif ram_total / GIB <= CONSERVATIVE_MAX_GIB:
        ram_tier = "conservative"
    else:
        ram_tier = "expanded"
    verified = ort.get("verified_providers") or []
    if "CUDAExecutionProvider" in verified:
        accel = "cuda"
    elif "DmlExecutionProvider" in verified:
        accel = "directml"
    else:
        accel = "cpu"
    notes = []
    if any(g.get("vendor") == "NVIDIA" and "name" in g for g in gpus) and accel == "cpu":
        notes.append("NVIDIA GPU present but no GPU backend verified yet; CPU is used until verification passes.")
    return {"ram_tier": ram_tier, "accelerator": accel, "notes": notes}


def probe(paths: list[str] | None = None) -> dict[str, Any]:
    total, avail = _ram_bytes()
    gpus = _nvidia()
    ort = _onnxruntime()
    return {
        "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "python": sys.version.split()[0],
        "cpu": _cpu(),
        "ram": {"total_bytes": total, "available_bytes": avail},
        "gpus_nvidia": gpus,
        "display_adapters": _windows_adapters(),
        "onnxruntime": ort,
        "disks": _disks(paths or [os.getcwd()]),
        "suggested_profile": suggest_profile(total, gpus, ort),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report workstation hardware for profile selection.")
    ap.add_argument("--paths", nargs="*", default=None, help="folders whose volumes to check for free space")
    ap.add_argument("--json", dest="json_out", help="write the report to this file")
    a = ap.parse_args(argv)
    rep = probe(a.paths)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
