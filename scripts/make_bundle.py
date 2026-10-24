"""Staging machine only: build the offline install bundle for the workstation (ENV-04, P9).

The bundle holds everything the workstation needs, so it never goes online:
the application wheel (with the built screens), every dependency as a wheel for
the target platform, the verified model packs, install and start scripts, and a
SHA256SUMS file.

    python scripts/build_ui.py
    python scripts/fetch_models.py
    python scripts/make_bundle.py --platform win_amd64 --python 3.12   # Windows workstation
    python scripts/make_bundle.py                                       # this machine's platform

Copy the resulting folder (or zip) to the workstation and run install.ps1 there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--platform", help="pip platform tag, e.g. win_amd64; default: this machine")
    ap.add_argument("--python", help="target Python version, e.g. 3.12; default: this interpreter")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--ort", default="cpu", choices=["cpu", "cuda", "directml", "auto"],
                    help="ONNX Runtime build: cpu, cuda (NVIDIA, about 1.2 GB more), directml (Windows GPUs), or auto "
                         "(both CUDA and DirectML; the installer picks CUDA when the NVIDIA driver is new enough)")
    ap.add_argument("--python-installer", help="official python-3.x.y-amd64.exe to ship for workstations without Python")
    ap.add_argument("--zip", action="store_true", help="also write a .zip of the bundle")
    ap.add_argument("--app-only", action="store_true",
                    help="small update: only the application wheel, for a workstation that already has the runtime and models")
    a = ap.parse_args(argv)

    from trafficwb import __version__
    from trafficwb.models import ModelPack

    static = ROOT / "src" / "trafficwb" / "web" / "static" / "index.html"
    if not static.is_file():
        print("The screens are not built: run scripts/build_ui.py first.", file=sys.stderr)
        return 1
    pyver = a.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    plat = a.platform or "native"
    name = f"trafficwb-offline-{__version__}-py{pyver.replace('.', '')}-{plat}-{a.ort}" + ("-update" if a.app_only else "")
    out = Path(a.out) / name
    if out.exists():
        print(f"{out} exists; remove it or choose another --out. Nothing was overwritten.", file=sys.stderr)
        return 1
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    wh = tmp / "wheelhouse"
    wh.mkdir(parents=True)

    # The application itself (pure Python, any platform).
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wh), str(ROOT)])
    # Dependencies as binary wheels for the target.
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = meta["optional-dependencies"]
    reqs = meta["dependencies"] + extras["probe"] + (extras["cuda"] + extras["directml"] if a.ort == "auto" else extras[a.ort])
    if (a.platform or sys.platform).startswith("win"):
        reqs.append("colorama")       # pulled in on Windows only, which pip cannot see when downloading from elsewhere
    if not a.app_only:
        cmd = [sys.executable, "-m", "pip", "download", "-d", str(wh), "--only-binary=:all:"]
        if a.platform or a.python:
            cmd += ["--python-version", pyver, "--implementation", "cp"]
            if a.platform:
                cmd += ["--platform", a.platform]
        run(cmd + reqs)

    # Model packs, verified before they are copied.
    packs = []
    for card in ([] if a.app_only else sorted(Path(a.models).glob("*/*/model.json"))):
        pack = ModelPack.load(card.parent)            # raises on a checksum mismatch
        dest = tmp / "models" / pack.name / pack.version
        shutil.copytree(card.parent, dest)
        packs.append({"name": pack.name, "version": pack.version, "task": pack.card["task"],
                      "evaluation_only": bool(pack.card.get("evaluation_only"))})
    if not a.app_only and not any(p["task"] == "vehicle_detection" for p in packs):
        print("No vehicle detector pack found: run scripts/fetch_models.py first.", file=sys.stderr)
        return 1

    if a.python_installer:
        shutil.copy2(a.python_installer, tmp / Path(a.python_installer).name)
    for f in ("install.ps1", "install.sh", "INSTALL.txt"):
        shutil.copy2(ROOT / "packaging" / f, tmp / f)
    shutil.copy2(ROOT / "scripts" / "offline_smoke.py", tmp / "offline_smoke.py")
    # The CUDA libraries shipped decide the oldest NVIDIA driver that can run them (Windows minimums from NVIDIA).
    cuda_major = next((int(p.name.split("-")[1].split(".")[0]) for p in wh.glob("nvidia_cuda_runtime*-*.whl")), None)
    cuda_min_driver = {11: "452.39", 12: "528.33", 13: "580.0"}.get(cuda_major) if cuda_major else None
    manifest = {"cuda_major": cuda_major, "cuda_min_driver": cuda_min_driver, "version": __version__, "python": pyver, "platform": plat, "ort": a.ort, "app_only": a.app_only, "models": packs,
                "requirements": reqs, "wheels": sorted(p.name for p in wh.iterdir())}
    (tmp / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [f"{sha(p)}  {p.relative_to(tmp).as_posix()}" for p in sorted(tmp.rglob("*"))
             if p.is_file() and p.name != "SHA256SUMS"]
    (tmp / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.rename(out)
    if a.zip:
        shutil.make_archive(str(out), "zip", out.parent, out.name)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundle: {out} ({size / 1e6:.0f} MB, {len(manifest['wheels'])} wheels, {len(packs)} model packs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
