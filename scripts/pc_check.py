"""Checks that this computer can load LanePulse's compiled parts. Run with the installed application's Python.

Loads each one in turn (NumPy, OpenCV, ONNX Runtime, and PyTorch when training is installed) and reports, in plain
words, the first one that does not load and why. On Windows it also reports which Microsoft C++ runtime files
(msvcp140.dll, vcruntime140.dll) were loaded and from where: an old copy in C:\\Windows\\System32 is the usual
reason ONNX Runtime fails with "A dynamic link library (DLL) initialization routine failed", and LanePulse ships its
own copies so it never depends on that one. Prints one JSON line last and exits 0 when everything loads.

    python pc_check.py [--training]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import traceback

# The C++ runtime ONNX Runtime and PyTorch are built with; older copies fail to initialise.
NEEDED = (14, 40)
RUNTIME_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _file_version(path: str) -> str | None:
    import ctypes
    from ctypes import wintypes
    ver = ctypes.WinDLL("version")
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(path, 0, size, buf):
        return None
    ptr, n = ctypes.c_void_p(), wintypes.UINT()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(n)) or not ptr.value:
        return None
    fixed = (wintypes.DWORD * 13).from_address(ptr.value)   # VS_FIXEDFILEINFO
    ms, ls = fixed[2], fixed[3]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def loaded_runtime() -> dict[str, dict]:
    """The C++ runtime files loaded in this process: path and version."""
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.restype = wintypes.HMODULE
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    out = {}
    for name in RUNTIME_DLLS:
        h = k32.GetModuleHandleW(name)
        if not h:
            continue
        buf = ctypes.create_unicode_buffer(1024)
        k32.GetModuleFileNameW(h, buf, 1024)
        out[name] = {"path": buf.value, "version": _file_version(buf.value)}
    return out


def system_runtime() -> dict[str, str | None]:
    """The versions in System32, which the app would use without its own copies."""
    if os.name != "nt":
        return {}
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    return {n: _file_version(os.path.join(sysdir, n)) for n in ("msvcp140.dll", "vcruntime140.dll")
            if os.path.isfile(os.path.join(sysdir, n))}


def _old(version: str | None) -> bool:
    if not version:
        return False
    major, minor = (int(x) for x in version.split(".")[:2])
    return (major, minor) < NEEDED


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true", help="also load PyTorch")
    a = ap.parse_args(argv)
    # OpenVINO's model converter would report usage on import; hide that package, as LanePulse itself does.
    for name in ("openvino_telemetry", "openvino_telemetry.backend", "openvino_telemetry.backend.backend_ga4"):
        sys.modules.setdefault(name, None)
    parts = [("numpy", "NumPy"), ("cv2", "OpenCV"), ("onnxruntime", "ONNX Runtime (runs the models)"),
             ("openvino", "OpenVINO (runs SAM 3.1)"), ("tokenizers", "Tokenizers (SAM 3.1 text prompts)")]
    if a.training:
        parts.append(("torch", "PyTorch (training)"))
    report: dict = {"python": sys.executable, "windows": platform.platform(), "loaded": {}, "failed": None}
    for mod, label in parts:
        try:
            m = importlib.import_module(mod)
            report["loaded"][mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - every failure is reported in words
            report["failed"] = {"part": label, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
            break
    report["cpp_runtime_loaded"] = loaded_runtime()
    report["cpp_runtime_system32"] = system_runtime()
    if report["failed"]:
        f = report["failed"]
        print(f"\nThis computer could not load {f['part']}: {f['error']}")
        loaded = report["cpp_runtime_loaded"].get("msvcp140.dll")
        if loaded and _old(loaded["version"]):
            print(f"The Microsoft C++ runtime in use is {loaded['version']} from {loaded['path']}; "
                  f"LanePulse needs {NEEDED[0]}.{NEEDED[1]} or newer. Its own copy was not used.")
        elif "DLL" in f["error"]:
            print("A Windows library did not load. Share install.log so the cause can be found.")
    else:
        print("This computer can run LanePulse: " + ", ".join(f"{k} {v}" for k, v in report["loaded"].items()))
    for name, info in report["cpp_runtime_loaded"].items():
        print(f"  {name} {info['version']} from {info['path']}")
    print(json.dumps(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
