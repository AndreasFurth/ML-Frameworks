"""
Template: export a PyTorch model to ONNX and verify with onnxruntime.
"""

import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = TinyNet().eval()
    dummy = torch.randn(1, 4)

    torch.onnx.export(
        model,
        dummy,
        "tinynet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    print("Exported tinynet.onnx")

    # Optional verification
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession("tinynet.onnx")
        outputs = ort_session.run(None, {"input": dummy.numpy()})
        print("ONNX runtime output shape:", outputs[0].shape)
    except Exception as exc:
        print("ONNX Runtime not available:", exc)


if __name__ == "__main__":
    main()
