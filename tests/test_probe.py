from trafficwb.probe import GIB, probe, suggest_profile


def test_probe_runs_and_reports_core_fields(tmp_path):
    rep = probe([str(tmp_path)])
    assert rep["cpu"]["logical_cores"]
    assert rep["ram"]["total_bytes"] is None or rep["ram"]["total_bytes"] > 0
    assert rep["disks"][0]["free_bytes"] >= 0
    assert rep["suggested_profile"]["accelerator"] in {"cpu", "cuda", "directml"}


def test_ram_tier_independent_of_gpu():
    ort = {"verified_providers": []}
    assert suggest_profile(16 * GIB, [], ort)["ram_tier"] == "conservative"
    assert suggest_profile(48 * GIB, [], ort)["ram_tier"] == "expanded"
    gpu = [{"vendor": "NVIDIA", "name": "NVIDIA RTX A4000"}]
    p = suggest_profile(16 * GIB, gpu, ort)
    assert p["ram_tier"] == "conservative" and p["accelerator"] == "cpu"  # GPU present but not verified
    assert p["notes"]
    assert suggest_profile(16 * GIB, gpu, {"verified_providers": ["CUDAExecutionProvider"]})["accelerator"] == "cuda"
