
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "timm",
#     "onnx",
#     "onnxscript",
# ]
# ///

import torch
import timm

MODEL_NAME = "mobilenetv3_large_100"
OUTPUT_FILE = "mobilenetv3_embedding.onnx"

print(f"Loading pretrained {MODEL_NAME}...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
    global_pool="avg",
)

model.eval()

dummy = torch.randn(1, 3, 224, 224)

print("Exporting ONNX...")

torch.onnx.export(
    model,
    (dummy,),
    OUTPUT_FILE,
    input_names=["image"],
    output_names=["embedding"],
    opset_version=18,
    dynamo=True,
    verify=True,
    external_data=False,
)

print(f"Exported: {OUTPUT_FILE}")

# Verify output dimensions
with torch.inference_mode():
    output = model(dummy)

print(f"Input shape:  {tuple(dummy.shape)}")
print(f"Output shape: {tuple(output.shape)}")
