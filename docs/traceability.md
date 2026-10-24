# Requirements traceability

Requirement IDs come from the design package (`docs/design/design-package.html`, section 01).
"Tested" means an automated test in this repository passed in CI or in a recorded local run. Nothing is marked tested on the basis of inspection alone.

| Req | Implementation | Tests | Status |
|---|---|---|---|
| TIME-01 filename UTC, precision kept, suffix opaque | `src/trafficwb/timeparse.py` | `tests/test_timeparse.py` | Tested (unit) |
| TIME-02 no filesystem fallback, malformed handling | `timeparse.parse_capture_time` raises `MalformedName` | `tests/test_timeparse.py` | Tested (unit). Parse preview UI not built. |
| TIME-04 IANA zone id, offset, DST status | `src/trafficwb/localtime.py` | `tests/test_localtime.py` | Conversion tested. Map UI not built. |
| TIME-05 per-timestamp rules | `localtime.to_local` | `tests/test_localtime.py` | Tested for IL, LV, LB, IR, FI. Manual override not built. |
| IMG-02 schedules, overnight, multiple windows, DST | `src/trafficwb/schedule.py` | `tests/test_schedule.py` | Tested (unit) |
| DER-02 minute-precision boundary rule | `schedule.Schedule.contains_local` | `tests/test_schedule.py` | Tested (unit) |
| HW-01 hardware detection | `src/trafficwb/probe.py` | `tests/test_probe.py` | Runs on Linux and CI runners. Not run on the target Windows workstation. GPU backend verification not built. |
| HW-02 RAM tier independent of GPU | `probe.suggest_profile` | `tests/test_probe.py` | Tested (unit). Tier cut-off is a proposal. |
| MIS-06 / ING-02 format definition from samples | `src/trafficwb/inventory.py` | `tests/test_inventory.py` | Tested (unit), run on the Fintraffic sample (see delivery report) |
| ING-03 identical bytes are distinct captures | `inventory` keeps per-file rows | `tests/test_inventory.py` | Tested (unit) |

All other requirements: not started.
