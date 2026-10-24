# Deployment, updates, backups and GPU choice

Requirements ENV-01, ENV-04 to ENV-07, ARC-02, ARC-04, HW-01; questions Q20, Q22, Q28.

## Offline bundle (staging machine)

`scripts/make_bundle.py` builds one folder that the workstation installs without any network:

- the application wheel, with the built screens inside;
- every dependency as a binary wheel for the target (`--platform win_amd64 --python 3.12`);
- the model packs, checksum-verified before they are copied;
- `install.ps1` (Windows), `install.sh` (Linux, used for testing), `INSTALL.txt`, `offline_smoke.py`;
- `bundle.json` (versions, models, wheels) and `SHA256SUMS` over every file.

`--ort` picks the ONNX Runtime build: `cpu`, `cuda` (NVIDIA, adds the CUDA and cuDNN libraries as Python packages, about 1.2 GB more), `directml` (any DirectX 12 GPU on Windows), or **`auto`**, which carries both GPU builds and lets the installer choose on the workstation. Checked on 24 Oct 2026: the `cuda` download for Windows and Python 3.12 is ONNX Runtime GPU 1.31 with CUDA 13 and cuDNN 9, which needs an NVIDIA driver of version 580 or later; `directml` is ONNX Runtime DirectML 1.24. The bundle records the oldest driver its CUDA libraries need (`cuda_min_driver` in `bundle.json`).

**`auto` is the recommended bundle for the A4000 workstation**, since its driver version is not known (Q20). The installer reads the driver version with `nvidia-smi`: version 580 or later gets the CUDA build; an older driver, or no NVIDIA driver, gets DirectML. If the CUDA build is chosen but the install check shows the models did not run on CUDA, the installer replaces it with the DirectML build and says so. `-Ort cuda|directml|cpu` overrides the choice.

`--python-installer` ships the official Python installer for a workstation without Python. `--app-only` builds a small update with only the application wheel.

## Install and update (workstation)

`install.ps1`:

1. checks every file against `SHA256SUMS`;
2. finds Python (given, the `py` launcher, `python` on the path, or the installer in the bundle, installed for the current user without admin rights);
3. copies the wheels into a local wheelhouse (`<target>\wheelhouse`, add-only), creates a new environment `<target>\app-<version>` and installs from the local wheelhouse with downloads off. Earlier versions' folders stay;
4. adds model packs that are not yet there (existing versions are never replaced);
5. chooses the ONNX Runtime build (see above), then runs the disconnected install-to-export check (`offline_smoke.py`): synthetic images, a mission, scan, detection with the installed detector, CSV and Excel export, a verified backup, and the web server's page and session answer. On failure the previous version stays in use;
6. on an update, backs up the data with the previous version, then writes `start.cmd` for the new version and records `current.txt` and `previous.txt`.

No admin rights, Docker, WSL or Windows service are needed. Running as a service or at sign-in is not built yet (ENV-08: needs your installation permissions).

Because wheels collect in the local wheelhouse, an `--app-only` update (a few hundred kB) installs as long as its dependencies were shipped before. If an update needs a new dependency, use a full bundle.

## Roll back

Point `start.cmd` back at the folder named in `previous.txt`. If the update changed the database format, restore the backup taken just before the update into a new data folder (`trafficwb-admin restore`) and point `--data` at it. Each mission database also keeps its own copy from just before every format change (`missions/<id>/backups/`).

## Backups (ARC-02, Q22)

- `trafficwb-admin backup --data D --to E:\TrafficBackups [--with-evidence]` copies `control.db` and every mission database with SQLite's online backup (safe while the server runs), writes a manifest with checksums and schema versions, and renames the folder from `.partial` when complete. Backups inside the data folder are refused.
- `trafficwb-admin verify <backup>` checks checksums and runs SQLite's integrity check.
- `trafficwb-admin restore <backup> --data <new folder>` verifies first and only writes into an empty or new folder.
- The server takes a nightly backup when started with `--backup-dir` (time with `--backup-at`, default 02:00 local). If the server was off at that time, the backup runs at its next check that day. "Back up now" and the last backup are on the missions page.
- Old backups are never deleted automatically (Q28 default: keep everything).

## Deleting a mission (Q28)

Only an administrator, only after typing the mission's exact name, and never while a job runs. The mission's database, crops and clips are removed; the source folder and exported files are not touched. The audit log keeps who, when, the mission's name, folders and row counts. `trafficwb-admin delete-mission` does the same from the command line, and `trafficwb-admin audit` lists the log.

## GPU or CPU (Q20, HW-01)

At start, each model session tries CUDA, then DirectML, then CPU, using only the backends the installed ONNX Runtime build offers, and reads back the backend it really got (ONNX Runtime falls back to CPU by itself when a driver or library is missing). The backend and the reason for any fallback are stored with every run and shown on the missions page. `TWB_ACCELERATOR=cpu|cuda|directml|auto` overrides the order.

## Measuring

`scripts/measure.py --images <folder> --work <new folder>` runs the whole pipeline on a folder and reports images per minute, seconds per image (median and 95th percentile), peak memory and storage per image, with the backend used. Run it on the workstation for the real budgets; figures from the cloud container are in `docs/delivery/P9-hardening-part1.md`.
