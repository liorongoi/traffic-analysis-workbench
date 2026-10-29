"""A tiny stand-in for the SAM 3.1 pack: the same graph inputs and outputs, a few kB, runnable by OpenVINO.

The "decoder" outlines the bright parts of the image (one query) and scores every prompt alike, except prompts
whose first word token is ``nothing``, which score low; that is enough to test the pipeline around the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

S = 64        # input size of the fake model; masks come out at the same size


def _save(graph, path: Path) -> None:
    import onnx
    from onnx import helper
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def build(dest: Path, name: str = "sam3.1-w8a16", version: str = "test") -> Path:
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    from trafficwb.models import sha256_file

    d = Path(dest) / name / version
    d.mkdir(parents=True, exist_ok=True)
    f = TensorProto.FLOAT
    # vision: mean over colour channels, kept at full size for every "FPN level"
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["m"], axes=[1], keepdims=1),
         *[helper.make_node("Identity", ["m"], [o]) for o in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2")]],
        "vision", [helper.make_tensor_value_info("pixel_values", f, [1, 3, S, S])],
        [helper.make_tensor_value_info(o, f, [1, 1, S, S]) for o in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2")],
        [])
    _save(g, d / "vision-encoder.onnx")
    # text: token ids as features; the mask from the attention mask
    g = helper.make_graph(
        [helper.make_node("Cast", ["input_ids"], ["idf"], to=f),
         helper.make_node("Unsqueeze", ["idf", "ax2"], ["text_features"]),
         helper.make_node("Greater", ["attention_mask", "zero"], ["text_mask"])],
        "text", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 32]),
                 helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 32])],
        [helper.make_tensor_value_info("text_features", f, [1, 32, 1]),
         helper.make_tensor_value_info("text_mask", TensorProto.BOOL, [1, 32])],
        [numpy_helper.from_array(np.array([2], np.int64), "ax2"), numpy_helper.from_array(np.array(0, np.int64), "zero")])
    _save(g, d / "text-encoder.onnx")
    # decoder: mask logits = 20 * brightness (the input is normalised to -1..1, so bright is positive); one query;
    # the score is low when the first word is token id 3 ("nothing")
    g = helper.make_graph(
        [helper.make_node("Mul", ["fpn_feat_1", "zf"], ["z1"]),
         helper.make_node("Mul", ["fpn_feat_2", "zf"], ["z2"]),
         helper.make_node("Mul", ["fpn_pos_2", "zf"], ["z3"]),
         helper.make_node("Sum", ["fpn_feat_0", "z1", "z2", "z3"], ["f0"]),
         helper.make_node("Mul", ["f0", "k"], ["pred_masks"]),
         helper.make_node("Identity", ["box"], ["pred_boxes"]),
         helper.make_node("Gather", ["prompt_features", "one"], ["tok"], axis=1),           # [1, 1]
         helper.make_node("Equal", ["tok", "nothing_id"], ["is_nothing"]),
         helper.make_node("Where", ["is_nothing", "low", "high"], ["pred_logits"]),
         helper.make_node("Cast", ["prompt_mask"], ["pm"], to=f),
         helper.make_node("ReduceMax", ["pm"], ["pmax"], keepdims=1),
         helper.make_node("Reshape", ["pmax", "shape11"], ["pmax11"]),
         helper.make_node("Mul", ["pmax11", "high"], ["presence_logits"]),
         ],
        "decoder",
        [helper.make_tensor_value_info(n, f, [1, 1, S, S]) for n in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2")]
        + [helper.make_tensor_value_info("prompt_features", f, [1, 32, 1]),
           helper.make_tensor_value_info("prompt_mask", TensorProto.BOOL, [1, 32])],
        [helper.make_tensor_value_info("pred_masks", f, [1, 1, S, S]),
         helper.make_tensor_value_info("pred_boxes", f, [1, 1, 4]),
         helper.make_tensor_value_info("pred_logits", f, [1, 1]),
         helper.make_tensor_value_info("presence_logits", f, [1, 1])],
        [numpy_helper.from_array(np.array(20.0, np.float32), "k"),
         numpy_helper.from_array(np.array([[[0.25, 0.5, 0.75, 1.0]]], np.float32), "box"),
         numpy_helper.from_array(np.array(1, np.int64), "one"),
         numpy_helper.from_array(np.array(0.0, np.float32), "zf"),
         numpy_helper.from_array(np.array([1, 1], np.int64), "shape11"),
         numpy_helper.from_array(np.array(3.0, np.float32), "nothing_id"),
         numpy_helper.from_array(np.array([[-6.0]], np.float32), "low"),
         numpy_helper.from_array(np.array([[4.0]], np.float32), "high")])
    _save(g, d / "prompt-decoder.onnx")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "[UNK]": 2, "nothing": 3, "road": 4, "car": 5, "van": 6,
             "truck": 7, "bus": 8, "motorcycle": 9, "bicycle": 10, "parking": 11, "lot": 12}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<|startoftext|> $A <|endoftext|>",
                                                       special_tokens=[("<|startoftext|>", 0), ("<|endoftext|>", 1)])
    tok.save(str(d / "tokenizer.json"))
    (d / "LICENSE").write_text("SAM License (test copy)\n", encoding="utf-8")
    files = {n: sha256_file(d / n) for n in ("text-encoder.onnx", "prompt-decoder.onnx", "tokenizer.json", "LICENSE")}
    card = {"name": name, "version": version, "task": "segmentation", "license": "SAM License (test)",
            "weights": "vision-encoder.onnx", "sha256": sha256_file(d / "vision-encoder.onnx"), "files": files,
            "graphs": {"vision": "vision-encoder.onnx", "text": "text-encoder.onnx", "decoder": "prompt-decoder.onnx"},
            "tokenizer": "tokenizer.json", "input_size": S, "text_length": 32, "pad_token_id": 1, "box_format": "xyxy",
            "precision": "fp32 (test)"}
    (d / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return d
