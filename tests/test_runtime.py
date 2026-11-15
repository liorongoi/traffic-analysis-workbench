import pytest

from trafficwb import runtime
from trafficwb.models import find
from trafficwb.runtime import COREML, CPU, CUDA, DML, AcceleratorError, make_session, wanted


def test_wanted_order_and_override(monkeypatch):
    monkeypatch.delenv("TWB_ACCELERATOR", raising=False)
    assert wanted() == [CUDA, DML, COREML, CPU]
    assert wanted("coreml") == [COREML, CPU]
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
    assert info.used == [CPU] and "failed session" in info.notes[-1]


def test_gpu_session_that_fails_first_inference_falls_back(monkeypatch):
    import onnxruntime as ort
    real = ort.InferenceSession

    class BrokenGpu:
        def get_inputs(self):
            return [type("Input", (), {"name": "input", "shape": [1, 3, 640, 640], "type": "tensor(float)"})()]

        def run(self, *_args):
            raise RuntimeError("provider failed during inference")

    def fake(path, opts=None, providers=None):
        if providers and providers[0] == CUDA:
            return BrokenGpu()
        return real(path, opts, providers=providers)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    _, info = make_session(_weights(), "auto", available=[CUDA, CPU])
    assert info.used == [CPU] and "warm-up inference" in info.notes[-1]


def _fake_coreml(monkeypatch, coreml_slower: bool):
    # Core ML exists only on Macs: a stand-in session that reports Core ML and is slower or faster than the CPU one.
    import time

    import onnxruntime as ort
    real = ort.InferenceSession

    class Timed:
        def __init__(self, inner, providers, delay):
            self.inner, self.providers, self.delay = inner, providers, delay

        def get_inputs(self):
            return self.inner.get_inputs()

        def get_providers(self):
            return self.providers

        def run(self, *a):
            time.sleep(self.delay)
            return self.inner.run(*a)

    def fake(path, opts=None, providers=None):
        names = [p[0] if isinstance(p, tuple) else p for p in providers or []]
        if names and names[0] == COREML:
            assert dict(providers[0][1])["MLComputeUnits"] == "ALL"   # GPU, Neural Engine and processor
            return Timed(real(path, opts, providers=[CPU]), [COREML, CPU], 0.05 if coreml_slower else 0)
        return Timed(real(path, opts, providers=providers), [CPU], 0 if coreml_slower else 0.05)

    monkeypatch.setattr(ort, "InferenceSession", fake)
    monkeypatch.setattr(runtime, "_COREML_CHOICE", {})


def test_coreml_kept_on_auto_only_when_faster(monkeypatch, tmp_path):
    monkeypatch.setenv("TWB_COREML_CACHE", str(tmp_path))
    _fake_coreml(monkeypatch, coreml_slower=False)
    _, info = make_session(_weights(), "auto", available=[COREML, CPU])
    assert info.accelerator == "coreml" and any("Core ML" in n and "ms" in n for n in info.notes)
    _fake_coreml(monkeypatch, coreml_slower=True)
    _, info = make_session(_weights(), "auto", available=[COREML, CPU])
    assert info.accelerator == "cpu" and "not faster" in info.notes[-1]
    _, again = make_session(_weights(), "auto", available=[COREML, CPU])      # remembered: no second measurement
    assert again.accelerator == "cpu" and again.requested == [CPU]
    _, forced = make_session(_weights(), "coreml", available=[COREML, CPU])   # asked for by name: not measured
    assert forced.accelerator == "coreml"
