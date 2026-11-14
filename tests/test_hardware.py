"""Which size of each fast detector suits this computer (hardware.py), the operator's override, and its screens."""
import pytest
from fastapi.testclient import TestClient
from test_learning import _pack
from test_web_api import H1, Fake, direct_access

from trafficwb import hardware as hw
from trafficwb.web.app import create_app

CPU_SMALL = {"cpu_threads": 4, "memory_gb": 8, "nvidia": [], "apple_silicon": False, "onnx": ["CPUExecutionProvider"]}
CPU_BIG = CPU_SMALL | {"cpu_threads": 16, "memory_gb": 32}
CUDA = lambda gb: CPU_BIG | {"nvidia": [{"name": "RTX", "memory_gb": gb}],             # noqa: E731
                             "onnx": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
APPLE = lambda gb: CPU_SMALL | {"cpu_threads": 10, "memory_gb": gb, "apple_silicon": True}                 # noqa: E731


@pytest.mark.parametrize("machine,run,train", [
    (CPU_SMALL, 0, 0), (CPU_BIG, 1, 0), (CUDA(3), 1, 1), (CUDA(6), 2, 2), (CUDA(12), 3, 3), (CUDA(24), 4, 4),
    (APPLE(8), 0, 1), (APPLE(16), 1, 2), (APPLE(64), 1, 3),
    (CPU_BIG | {"onnx": ["DmlExecutionProvider", "CPUExecutionProvider"]}, 2, 0),
    # an NVIDIA card without CUDA in ONNX Runtime still trains on it (PyTorch), but detection stays on the processor
    (CPU_SMALL | {"nvidia": [{"name": "RTX", "memory_gb": 8}]}, 0, 2),
])
def test_levels_follow_the_hardware(machine, run, train):
    lv = hw.levels(machine)
    assert (lv["run"][0], lv["train"][0]) == (run, train)
    assert lv["run"][1] and lv["train"][1]


def test_sizes_per_level_and_nearest_installed():
    assert [hw.pick("yolox", lv, hw.ORDER["yolox"]) for lv in range(5)] == ["tiny", "s", "m", "l", "x"]
    assert [hw.pick("rt-detr", lv, hw.ORDER["rt-detr"]) for lv in range(5)] == ["r18", "r18", "r34", "r50", "r101"]
    assert hw.pick("yolox", 4, ["tiny", "s"]) == "s"                     # the nearest smaller first
    assert hw.pick("yolox", 0, ["s", "m"]) == "s"                        # then the nearest larger
    assert hw.pick("d-fine", 2, []) is None
    assert all(len(hw.SIZES[f]) == 5 and set(hw.SIZES[f]) <= set(hw.ORDER[f]) for f in hw.FAMILIES)


def test_operator_choice_wins_when_installed(tmp_path, monkeypatch):
    models = tmp_path / "models"
    for n in ("yolox-tiny", "yolox-s", "yolox-x"):
        _pack(models, n, "0.1.1rc0")
    _pack(models, "rf-detr-small", "1")
    monkeypatch.setattr(hw, "detect", lambda: CPU_SMALL)
    assert hw.installed_runnable(models)["yolox"] == ["tiny", "s", "x"]
    assert hw.run_size("yolox", tmp_path, models) == "tiny"              # a small CPU: the smallest
    assert hw.run_size("rf-detr", tmp_path, models) == "small"           # nano missing: the nearest larger
    hw.set_choice(tmp_path, "yolox", "x")
    assert hw.run_size("yolox", tmp_path, models) == "x"
    assert hw.yolox_pack(tmp_path, models).name == "yolox-x"
    hw.set_choice(tmp_path, "yolox", "m")                                 # chosen but not installed: automatic
    assert hw.run_size("yolox", tmp_path, models) == "tiny"
    hw.set_choice(tmp_path, "yolox", None)
    assert hw.choices(tmp_path) == {}
    with pytest.raises(ValueError, match="comes in"):
        hw.set_choice(tmp_path, "yolox", "huge")
    with pytest.raises(ValueError, match="Unknown detector"):
        hw.set_choice(tmp_path, "yolo9", "s")
    assert hw.run_pack("d-fine", tmp_path, models) is None


def test_screens_show_and_change_the_sizes(tmp_path, monkeypatch):
    models = tmp_path / "models"
    for n in ("yolox-tiny", "yolox-s"):
        _pack(models, n, "0.1.1rc0")
    monkeypatch.setenv("TWB_MODELS", str(models))
    monkeypatch.setattr(hw, "detect", lambda: CUDA(24) | {"onnx": ["CPUExecutionProvider"]})
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    h = c.get("/api/hardware").json()
    y = h["families"]["yolox"]
    assert y["installed"] == ["tiny", "s"] and y["auto"] == "s" and y["runs"] == "s" and y["chosen"] is None
    assert h["train_level"] == 4 and "RTX" in h["train_on"] and "processor" in h["run_on"]
    r = c.put("/api/hardware/sizes", json={"family": "yolox", "size": "tiny"}, headers=H1).json()
    assert r["families"]["yolox"]["runs"] == "tiny" and r["families"]["yolox"]["chosen"] == "tiny"
    assert c.put("/api/hardware/sizes", json={"family": "yolox", "size": "huge"}, headers=H1).status_code == 400
    r = c.put("/api/hardware/sizes", json={"family": "yolox", "size": None}, headers=H1).json()
    assert r["families"]["yolox"]["runs"] == "s"
