"""
Template: export a PyTorch model to TorchScript via tracing or scripting.
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

    traced = torch.jit.trace(model, dummy)
    traced.save("tinynet_traced.pt")
    print("Saved tinynet_traced.pt")

    scripted = torch.jit.script(model)
    scripted.save("tinynet_scripted.pt")
    print("Saved tinynet_scripted.pt")


if __name__ == "__main__":
    main()
