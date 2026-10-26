# Delivery report: hardening, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- **Offline bundle** (`scripts/make_bundle.py`): application wheel with the screens, every dependency as a wheel for the target platform, verified model packs, install scripts, `bundle.json` and `SHA256SUMS`. Choice of ONNX Runtime build (CPU, CUDA, DirectML, or `auto` with both GPU builds, chosen by the installer from the NVIDIA driver version); optional Python installer; `--app-only` small updates.
- **Installers** (`packaging/install.ps1` for Windows, `packaging/install.sh` for Linux testing): checksum check, per-user install without admin rights, one folder per version, a local add-only wheelhouse, model packs added without replacing, a disconnected install-to-export check before switching, a backup before an update, `current.txt` and `previous.txt` for rollback. `packaging/INSTALL.txt` explains install, rollback, backups and GPU.
- **Disconnected check** (`scripts/offline_smoke.py`).
- **Backups and restore** (`trafficwb.admin`, `trafficwb-admin`): online backup with checksums and schema versions, verify, restore only into an empty folder, optional evidence copy; nightly backup in the server (`--backup-dir`, `--backup-at`); "Back up now" and the last backup on the missions page.
- **Mission deletion**: admin only, exact name typed, never while a job runs, source and exports untouched, audit log entry. On the mission overview and in `trafficwb-admin`.
- **Audit log** in `control.db`, shown to administrators on the missions page.
- **GPU or CPU** (`trafficwb.runtime`): CUDA, then DirectML, then CPU, reading back what ONNX Runtime really uses; fallback reasons recorded with each run and shown on the missions page; `TWB_ACCELERATOR` override. ONNX Runtime moved to the extras `cpu`, `cuda` and `directml`, so exactly one build is installed.
- **Measuring** (`scripts/measure.py`).
- CI: a Windows job builds the win_amd64 bundle and runs the installer and its check with package downloads off and web traffic sent to a dead proxy.
- Rules: `docs/design/deployment.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite. New: `tests/test_admin.py` (backup, verify, restore, damaged backup refused, backup inside the data folder refused, update backs up before migrating and rollback restores the old schema, deletion needs the exact name and is logged, nightly schedule), `tests/test_runtime.py` (backend order, override, missing backend, backend that fails to start), API tests for system, backup now and deletion | Linux cloud container, Python 3.13 | All passed |
| Install from the bundle in a network namespace with no network at all: 26 wheels and 3 model packs installed, then mission, detection on 6 Finnish sample images (45 vehicles), CSV and Excel export, verified backup, web server page and session | Linux cloud container | Passed; the check reported "no network" |
| Update: a 264 kB application-only bundle (version 0.0.2, built for the test) installed over the full 0.0.1 install with no network; the mission created with 0.0.1 opens with 0.0.2; both version folders kept | Linux cloud container | Passed |
| Windows bundle downloads for Python 3.12: CPU (100 MB of wheels), CUDA (ONNX Runtime GPU 1.31 with CUDA 13 and cuDNN 9, 1.3 GB), DirectML (ONNX Runtime DirectML 1.24) | Cloud container (download only) | All resolved |
| Windows install and check, CPU bundle and `auto` bundle (the runner has no NVIDIA driver, so `auto` must choose DirectML) | GitHub Actions `offline-bundle` jobs | See the PR checks |

## Measurements (cloud container, 4 CPU cores, no GPU)

Finnish sample, 143 images (1280×720 day, infrared night), whole pipeline including night analysis, YOLOX-s with 3×2 tiles on CPU:

| Mission | Images per minute | Seconds per image (median / 95th percentile) | Peak memory | Database per image | Evidence per image |
|---|---|---|---|---|---|
| General traffic | 48.9 | 1.09 / 2.07 | 308 MB | 20.5 kB | 0 |
| Plate recognition (1,027 vehicles: 1 plate read, 401 no plate, 625 too small) | 48.2 | 0.96 / 1.19 | 340 MB | 22.8 kB | 0.1 kB |

Model loading took under 0.3 s and scanning 143 files under 0.6 s. These are container figures, not workstation budgets: run `scripts/measure.py` on the workstation, on CPU and on its GPU.

## Not tested

- Anything on the target workstation, a GPU (CUDA or DirectML), or an air-gapped Windows machine. The Windows CI job blocks downloads but the runner itself is online.
- The Python installer path in `install.ps1` (CI already has Python).
- Peak VRAM, 16 GB and 48 GB machine profiles, storage speed.
- At the time of this report, Windows automatic startup was not tested. The current installer starts manually by default; `autostart.ps1` supports Windows user-session startup and an administrator-configured power-on task. LanePulse application sign-in was removed later.
- HTTPS for LAN use (Q21).