#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "onnx",
#     "onnxruntime",
#     "numpy",
#     "pillow",
# ]
# ///
"""Static int8 quantization (QDQ) of U2-Net for ONNX Runtime CPU.

  uv run quantize_u2net.py u2net.onnx calibration_photos/ u2net_int8.onnx

Dependencies are declared inline (PEP 723); uv installs them automatically.
Calibration: ~16-32 representative photos (use frames from the kiosk camera if you can).
Tested on rembg's u2net.onnx (opset 11): masks matched fp32 at IoU ~0.995.
u2netp does NOT survive int8 (IoU ~0.6): only use this on u2net.
"""
import argparse, glob, os, tempfile
import numpy as np
import onnx
from onnx import version_converter
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

MEAN = np.array([0.485, 0.456, 0.406], "f4")
STD = np.array([0.229, 0.224, 0.225], "f4")

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("photos"); p.add_argument("out")
p.add_argument("--max-images", type=int, default=24)
p.add_argument("--input-name", default="input.1")
p.add_argument("--ir-version", type=int, default=9, help="0 = leave as is")
p.add_argument("--size", type=int, default=320,
               help="square input size, multiple of 32 (320 = original; 256/288 are faster)")
a = p.parse_args()
assert a.size % 32 == 0, "--size must be a multiple of 32 (encoder downsamples 5x)"
SIZE = a.size


def prep(im):  # identical to the Rust preprocessing (squash to SIZE, /max, mean/std)
    x = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), "f4") / 255.0
    x = x / max(x.max(), 1e-6)
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype("f4")


files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(a.photos, f"*.{ext}")))
step = max(1, len(files) // a.max_images)
files = files[::step][: a.max_images]
assert files, "no calibration photos found"
print(f"calibrating on {len(files)} photos")


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter([prep(Image.open(f)) for f in files])

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {a.input_name: x}


with tempfile.TemporaryDirectory() as tmp:
    m = onnx.load(a.model)
    # Rewrite the declared H/W of the graph inputs/outputs; drop stale 320-sized intermediate
    # shapes so shape inference in quant_pre_process recomputes them for SIZE.
    for t in list(m.graph.input) + list(m.graph.output):
        d = t.type.tensor_type.shape.dim
        if len(d) == 4:
            d[2].dim_value = SIZE
            d[3].dim_value = SIZE
    del m.graph.value_info[:]
    print("input size", SIZE)
    print("opset", [o.version for o in m.opset_import], "-> 13 (per-channel needs >= 13)")
    onnx.save(version_converter.convert_version(m, 13), f"{tmp}/op13.onnx")
    quant_pre_process(f"{tmp}/op13.onnx", f"{tmp}/pre.onnx", skip_symbolic_shape=True)
    quantize_static(f"{tmp}/pre.onnx", a.out, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv"])

# Newer onnx stamps the model with extra operator-set domains (e.g. ai.onnx.ml v5) that no node
# uses. Older ONNX Runtime builds (e.g. 1.17) refuse to load such files, so keep only domains
# that nodes actually use (here: just the default ai.onnx).
m = onnx.load(a.out)
used = {n.domain or "ai.onnx" for n in m.graph.node}
keep = [o for o in m.opset_import if (o.domain or "ai.onnx") in used]
del m.opset_import[:]
m.opset_import.extend(keep)
print("opsets:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
if a.ir_version:
    print("IR version", m.ir_version, "->", a.ir_version)
    m.ir_version = a.ir_version
onnx.checker.check_model(m)
onnx.save(m, a.out)
print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.0f} MB)")
