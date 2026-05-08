import sys, os, torch, torch.nn as nn
import segmentation_models_pytorch as smp
import onnx
from onnxsim import simplify  # pip install onnxsim

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEIGHTS_PATH = r"C:\Users\User\Downloads\pls work\mass_best_model (3).pth"
OUTPUT_PATH  = r"C:\Users\User\Downloads\pls work\model.onnx"
IMG_SIZE     = 320


class SafeSiLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)  # ReLU is universally supported in ort-web

def replace_activations(model):
    for name, module in model.named_children():
        if isinstance(module, (nn.SiLU, nn.Hardswish)):
            setattr(model, name, SafeSiLU())
        else:
            replace_activations(module)

class SegmentationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=(128, 64, 32, 16, 8),
            decoder_attention_type=None,
        )

    def forward(self, x):
        features    = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        return self.unet.segmentation_head(decoder_out)

print("Loading model...")
model = SegmentationUNet()
state = torch.load(WEIGHTS_PATH, map_location="cpu")
if isinstance(state, dict):
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            print(f"  Unwrapping key '{key}'...")
            state = state[key]
            break
model.load_state_dict(state)

# Patch activations BEFORE eval() and export
print("Patching SiLU/HardSwish → ReLU for ort-web compatibility...")
replace_activations(model)

model.eval()
print("Weights loaded OK")

print("Exporting to ONNX...")
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        OUTPUT_PATH,
        export_params=True,
        opset_version=12,       
        do_constant_folding=False,  # Avoid unsupported fused ops in browser
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={              # Required for ort-web to handle input properly
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
        verbose=False,
    )
print("Raw export done.")

# Simplify the graph — removes redundant nodes 
print("Simplifying ONNX graph (onnxsim)")
m = onnx.load(OUTPUT_PATH)
m_simplified, check = simplify(m)
if check:
    print("Simplification OK")
    m = m_simplified
else:
    print("Simplification failed, using original (may still work)")

# Inline all weights — no .data sidecar file
print("Inlining weights...")
onnx.save(m, OUTPUT_PATH, save_as_external_data=False)

# Validate
print("Validating")
onnx.checker.check_model(OUTPUT_PATH)

size_mb = round(os.path.getsize(OUTPUT_PATH) / 1e6, 1)
print(f"Done! model.onnx = {size_mb} MB")
print("This file should load correctly in ort-web (WASM or WebGL backend).")
