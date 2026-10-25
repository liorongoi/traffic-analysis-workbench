# Delivery report: Phase 0, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.timeparse`: UTC capture time from two filename patterns (spec `DDMMYYYY-HHmm-suffix`, minute precision; Fintraffic `<camera>_YYYYMMDDTHHMMSSZ`, second precision). Impossible dates rejected. No filesystem-time fallback.
- `trafficwb.localtime`: UTC → local with IANA rules per instant.
- `trafficwb.schedule`: weekday windows, overnight windows, several windows per day, date ranges, half-open whole-interval boundary rule.
- `trafficwb-probe`: CPU, RAM, NVIDIA GPU (via `nvidia-smi`), Windows display adapters, ONNX Runtime providers, disk free space, suggested RAM tier and accelerator.
- `trafficwb-inventory`: read-only sample report (format, dimensions, EXIF, partial-JPEG check, SHA-256, name parsing, intervals per camera).
- CI workflow: Windows and Linux, Python 3.11 and 3.12.

## Actually tested

| What | Where | Result |
|---|---|---|
| 56 unit tests (`pytest -v`) | Linux cloud container, Python 3.13.17, tzdata 2026e | 56 passed |
| Same tests | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Inventory on the Fintraffic sample (`trafficwb-inventory <sample>/images --json inv.json`) | Linux VM on the owner's computer, Python 3.10.12, sample folder mounted read-only | 143 JPEGs, all 1280×720, 0 with EXIF, 143/143 names parsed with second precision, 0 image errors, 0 identical-content groups, range 2026-10-20T19:45:13Z to 2026-10-21T19:35:14Z, intervals 141 × 10 min and 1 × 20 min |
| Inventory checksums and capture times vs the sample's own `images.csv` | same | 0 mismatches out of 143 |
| `trafficwb-probe` | Linux cloud container | Runs; reports 4 cores, ~16 GB RAM, no GPU. This is not the target workstation. |

## Not tested

- Nothing has run on the target Windows workstation, on 16 GB or 48 GB target hardware, or on an RTX A4000.
- GPU backend verification (creating an inference session per provider) is not built; the probe only lists installed providers.
- No video samples yet, so the video path of the inventory (ffprobe) has not run on real data.
- Offline install, packaging and no-admin start are not built.

## Known limitations

- The RAM tier cut-off (24 GiB) is a proposal, not a measurement.
- Only two filename patterns exist. Other naming conventions are reported as malformed.
- The Fintraffic sample is from Finland, outside the four target countries; it is used for format and night-image work only.
