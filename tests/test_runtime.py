import pytest

from trafficwb.models import find
from trafficwb.runtime import CPU, CUDA, DML, AcceleratorError, make_session, wanted


def test_wanted_order_and_override(monkeypatch):
    monkeypatch.delenv("TWB_ACCELERATOR", raising=False)
    assert wanted() == [CUDA, DML, CPU]
    assert wanted("cpu") == [CPU]
    assert wanted("cuda") == [CUDA, CPU]
    monkeypatch.setenv("TWB_ACCELERATOR", "DirectML")
    assert wanted() == [DML, CPU]
    with pytest.raises(AcceleratorError):
        wanted("tpu")


def _weights():
    try:
        return find("plate-ocr-cct-s-v2-global").weights
    except Exception:
        pytest.skip("plate reader pack not installed")


def test_cpu_session_and_missing_gpu_noted():
    s, info = make_session(_weights(), "auto")
    assert info.used[-1] == CPU and info.accelerator in ("cpu", "cuda", "directml")
    s, info = make_session(_weights(), "cuda", available=[CPU])
    assert info.accelerator == "cpu" and any("CUDA" in n for n in info.notes)


def test_gpu_offered_but_unusable_falls_back(monkeypatch):
    # A build that lists CUDA but cannot start it: the session must still open on CPU and say why.
    import onnxruntime as ort
    real = ort.InferenceSession

    def fake(path, opts=None, providers=None):
        if providers and providers[0] == CUDA:
            raise RuntimeError("libcudnn not found")
        return real(path, opts, providers=providers)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    s, info = make_session(_weights(), "auto", available=[CUDA, CPU])
    assert info.used == [CPU] and "failed to start" in info.notes[-1]
