#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "timm",
#     "onnx",
#     "onnxscript",
#     "onnxruntime",
# ]
# ///
"""Export any timm model as an ONNX embedding extractor.

Examples
--------
  ./export_mobilenet.py                                  # same as the old script: 224, batch 1
  ./export_mobilenet.py --size 288                       # bigger input
  ./export_mobilenet.py --size native                    # model's own training size
  ./export_mobilenet.py --size test                      # model's recommended test size
  ./export_mobilenet.py --model mobilenetv3_small_100    # other variant
  ./export_mobilenet.py --batch dynamic                  # embed N subjects per call
  ./export_mobilenet.py --bake-preprocess --l2norm       # feed pixel/255, get unit vectors
  ./export_mobilenet.py --list-models 'mobilenetv3*'     # what exists in timm
  ./export_mobilenet.py --info                           # print the model's data config only

A <output>.json sidecar is written with mean/std/size/embedding dim and the
SQUARE_BG value to paste into variables.rs.
"""

import argparse
import json
import sys
from pathlib import Path

import onnx
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("model")
    g.add_argument("--model", default="mobilenetv3_large_100", help="timm model name")
    g.add_argument("--no-pretrained", action="store_true",
                   help="random weights (offline testing only; embeddings are meaningless)")
    g.add_argument("--checkpoint", metavar="PATH",
                   help="load weights from a local checkpoint on top of the model")
    g.add_argument("--pool", default="avg", choices=["avg", "max", "avgmax", "catavgmax"],
                   help="global pooling. catavgmax doubles the embedding size (default: avg)")
    g.add_argument("--model-kwargs", default="{}", metavar="JSON",
                   help='extra timm.create_model kwargs, e.g. \'{"drop_rate": 0.0}\'')

    g = p.add_argument_group("input")
    g.add_argument("--size", default="224",
                   help="N (square), HxW, 'native' (pretrained input_size) or 'test' "
                        "(pretrained test_input_size). Default: 224")
    g.add_argument("--batch", default="1",
                   help="N for a fixed batch size, or 'dynamic'. Default: 1")
    g.add_argument("--dynamic-size", action="store_true",
                   help="EXPERIMENTAL: also make height/width dynamic (--size is the trace size)")
    g.add_argument("--mean", type=float, nargs=3, metavar=("R", "G", "B"),
                   help="override normalization mean (default: from the model's data config)")
    g.add_argument("--std", type=float, nargs=3, metavar=("R", "G", "B"),
                   help="override normalization std (default: from the model's data config)")

    g = p.add_argument_group("graph")
    g.add_argument("--bake-preprocess", action="store_true",
                   help="fold (x - mean) / std into the graph: feed RGB in 0..1 (NCHW, f32)")
    g.add_argument("--l2norm", action="store_true",
                   help="fold L2 normalization of the embedding into the graph (cosine = dot)")

    g = p.add_argument_group("export")
    g.add_argument("--out", metavar="FILE",
                   help="output path (default: <model>_<H>x<W>_embedding.onnx)")
    g.add_argument("--opset", type=int, default=18)
    g.add_argument("--ir-version", type=int, default=9,
                   help="IR version to stamp for older ONNX Runtime builds; 0 = leave as is")
    g.add_argument("--input-name", default="image")
    g.add_argument("--output-name", default="embedding")
    g.add_argument("--no-verify", action="store_true", help="skip torch.onnx.export verify=True")
    g.add_argument("--no-ort-check", action="store_true",
                   help="skip the final onnxruntime-vs-torch comparison")
    g.add_argument("--no-sidecar", action="store_true", help="don't write the .json file")

    g = p.add_argument_group("info")
    g.add_argument("--list-models", metavar="PATTERN",
                   help="list pretrained timm models matching PATTERN and exit")
    g.add_argument("--info", action="store_true",
                   help="print the model's data config and exit (no export)")
    return p.parse_args()


def parse_size(text, cfg):
    t = text.strip().lower()
    native = tuple(cfg["input_size"][1:])
    if t == "native":
        return native
    if t == "test":
        return tuple(cfg.get("test_input_size", cfg["input_size"])[1:])
    if "x" in t:
        h, w = t.split("x")
        return int(h), int(w)
    return int(t), int(t)


# ---------------------------------------------------------------------------
# Wrapper: lets us bake preprocessing / L2 norm in and fixes the input name
# ---------------------------------------------------------------------------
class Embedder(nn.Module):
    def __init__(self, model, mean, std, bake, l2):
        super().__init__()
        self.model, self.bake, self.l2 = model, bake, l2
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    # NOTE: the argument name is part of the dynamic_shapes key below.
    def forward(self, image):
        x = (image - self.mean) / self.std if self.bake else image
        e = self.model(x)
        return F.normalize(e, dim=1) if self.l2 else e


def main():
    a = parse_args()

    if a.list_models:
        for n in timm.list_models(a.list_models, pretrained=True):
            print(n)
        return

    print(f"Loading {'random-init' if a.no_pretrained else 'pretrained'} {a.model}...")
    model = timm.create_model(
        a.model,
        pretrained=not a.no_pretrained,
        num_classes=0,
        global_pool=a.pool,
        **json.loads(a.model_kwargs),
    )
    if a.checkpoint:
        print(f"Loading checkpoint {a.checkpoint}")
        timm.models.load_checkpoint(model, a.checkpoint, strict=False)
    model.eval()

    cfg = timm.data.resolve_data_config({}, model=model)
    mean = list(a.mean or cfg["mean"])
    std = list(a.std or cfg["std"])
    h, w = parse_size(a.size, cfg)

    print("Model data config:")
    for k in ("input_size", "test_input_size", "mean", "std", "interpolation", "crop_pct"):
        if k in cfg:
            print(f"  {k}: {cfg[k]}")
    if a.info:
        return
    if h % 32 or w % 32:
        print(f"WARNING: {h}x{w} is not a multiple of 32 (the network's total stride); "
              "it will run but feature maps get rounded.", file=sys.stderr)

    dynamic_batch = a.batch.lower() == "dynamic"
    # torch.export specializes batch 1, so trace a dynamic batch with 2.
    n = 2 if dynamic_batch else int(a.batch)
    dummy = torch.randn(n, 3, h, w)

    net = Embedder(model, mean, std, a.bake_preprocess, a.l2norm).eval()
    with torch.inference_mode():
        ref = net(dummy)
    emb_dim = ref.shape[1]

    dynamic_shapes = None
    if dynamic_batch or a.dynamic_size:
        Dim = torch.export.Dim
        dims = {}
        if dynamic_batch:
            dims[0] = Dim("batch", min=1, max=256)
        if a.dynamic_size:
            dims[2] = Dim.AUTO
            dims[3] = Dim.AUTO
        dynamic_shapes = {"image": dims}

    out = Path(a.out or f"{a.model}_{h}x{w}_embedding.onnx")
    print(f"Exporting ONNX -> {out}  (input {tuple(dummy.shape)}"
          f"{', dynamic batch' if dynamic_batch else ''}"
          f"{', dynamic size' if a.dynamic_size else ''})")
    kwargs = dict(
        input_names=[a.input_name],
        output_names=[a.output_name],
        opset_version=a.opset,
        dynamo=True,
        verify=not a.no_verify,
        external_data=False,
    )
    if dynamic_shapes:
        kwargs["dynamic_shapes"] = dynamic_shapes
    torch.onnx.export(net, (dummy,), str(out), **kwargs)

    if a.ir_version:
        print("Adjusting ONNX IR version for ORT compatibility...")
        m = onnx.load(str(out))
        print(f"  original IR version: {m.ir_version}")
        m.ir_version = a.ir_version
        onnx.checker.check_model(m)
        onnx.save(m, str(out))
        print(f"  final IR version: {m.ir_version}")

    if not a.no_ort_check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print(f"ORT input : {inp.name} {inp.shape}")
        print(f"ORT output: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
        batches = [n] if not dynamic_batch else [1, 3]
        for b in batches:
            x = torch.randn(b, 3, h, w)
            with torch.inference_mode():
                t = net(x)
            o = torch.from_numpy(sess.run(None, {inp.name: x.numpy()})[0])
            cos = F.cosine_similarity(t, o, dim=1).min().item()
            print(f"  batch {b}: min cosine(torch, ort) = {cos:.6f}, "
                  f"max |diff| = {(t - o).abs().max().item():.2e}")

    if not a.no_sidecar:
        side = {
            "model": a.model,
            "pretrained": not a.no_pretrained,
            "file": out.name,
            "input_name": a.input_name,
            "output_name": a.output_name,
            "input_shape": ["batch" if dynamic_batch else n, 3, h, w],
            "layout": "NCHW, RGB, f32",
            "embedding_dim": int(emb_dim),
            "l2_normalized": a.l2norm,
            "preprocess_baked": a.bake_preprocess,
            # What the Rust side must do before feeding the model:
            "input_range": "0..1 (no further normalization)" if a.bake_preprocess
                           else "0..1, then (x - mean) / std",
            "mean": mean,
            "std": std,
            "interpolation": cfg.get("interpolation"),
            # Paste into variables.rs: padding colour that normalizes to zero.
            "square_bg": [round(c * 255) for c in mean],
            "opset": a.opset,
            "ir_version": a.ir_version or None,
        }
        sp = out.with_suffix(".json")
        sp.write_text(json.dumps(side, indent=2))
        print(f"Wrote {sp}")

    print()
    print(f"Exported: {out}")
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(ref.shape)}   (embedding dim {emb_dim})")
    print(f"Rust: instance_squares(&img, Some({h}))   SQUARE_BG = "
          f"{[round(c * 255) for c in mean]}")
    if h != w:
        print("NOTE: non-square input; instance_squares produces squares, resize to "
              f"{h}x{w} yourself or use a square --size.")


if __name__ == "__main__":
    main()
