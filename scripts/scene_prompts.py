"""Staging machine only: make src/trafficwb/scene_prompts.json, the two text embeddings the indoor/outdoor sorter
compares each image with: indoor, outdoor seen from inside (through a window or door), and outdoor. Run once when the prompts or the CLIP model change; the app itself never needs the text
model.

    pip install tokenizers onnxruntime numpy
    python scripts/scene_prompts.py --text-model text_model.onnx --tokenizer tokenizer.json

text_model.onnx and tokenizer.json come from https://huggingface.co/Xenova/clip-vit-base-patch32 at the commit
fetch_models.py pins (onnx/text_model.onnx, the full-precision text encoder). The offsets are kept from the existing
file unless --bias is given (three numbers: indoor, window, outdoor); they were chosen on half of the photos in
docs/decisions/indoor-outdoor.md and checked on the other half.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "trafficwb" / "scene_prompts.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bias", type=float, nargs=3)
    a = ap.parse_args(argv)
    import onnxruntime as ort
    from tokenizers import Tokenizer
    cur = json.loads(OUT.read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(a.tokenizer)
    pad = tok.token_to_id("<|endoftext|>")
    sess = ort.InferenceSession(a.text_model)

    def embed(texts: list[str]) -> list[float]:
        ids = np.full((len(texts), 77), pad, np.int64)
        for i, t in enumerate(texts):
            x = tok.encode(t).ids[:77]
            ids[i, :len(x)] = x
        e = sess.run(None, {"input_ids": ids})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        m = e.mean(0)
        return [round(float(v), 6) for v in m / np.linalg.norm(m)]

    for c in cur["classes"]:
        c["embedding"] = embed(c["prompts"])
    if a.bias is not None:
        for c, b in zip(cur["classes"], a.bias):
            c["bias"] = b
    OUT.write_text(json.dumps(cur), encoding="utf-8")
    print(f"ok {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
