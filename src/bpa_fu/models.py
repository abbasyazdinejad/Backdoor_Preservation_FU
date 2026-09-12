"""Model definitions.

SimpleCNN reproduces the architecture of the original implementation exactly:
Conv(3x3,32)-ReLU-MaxPool-Conv(3x3,64)-ReLU-MaxPool-Flatten-FC(4096,128)-ReLU-FC(128,K).

ResNet18CIFAR is the CIFAR-adapted ResNet-18 (He et al., 2016; CIFAR variant as in the common
32x32 adaptations): a 3x3 stride-1 stem convolution, no ImageNet max-pooling, four stages of two
BasicBlocks (64, 128, 256, 512 channels), global average pooling and a linear classifier.
BatchNorm layers are ordinary nn.BatchNorm2d; their running statistics are part of the state dict
and are aggregated by FedAvg like every other floating-point tensor, while num_batches_tracked
(int64) is kept from the global model.

parameter_groups(model) returns the stable module groups used by the architecture-independent
layer-wise analyses (alignment energy, checkpoint checks); every parameter tensor of the model
belongs to exactly one group.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stride-1 stem, no max-pooling, [2, 2, 2, 2] BasicBlocks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Final feature map (N, 512, 4, 4) after the last residual stage (post-ReLU)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


ARCHITECTURES = {"simplecnn": SimpleCNN, "resnet18": ResNet18CIFAR}

# Stable module groups for the layer-wise analyses: (group name, state-dict key prefixes).
PARAMETER_GROUPS = {
    "simplecnn": [("conv1", ["conv1."]), ("conv2", ["conv2."]), ("fc1", ["fc1."]), ("fc2", ["fc2."])],
    "resnet18": [("stem", ["conv1.", "bn1."]), ("layer1", ["layer1."]), ("layer2", ["layer2."]),
                 ("layer3", ["layer3."]), ("layer4", ["layer4."]), ("fc", ["fc."])],
}


def build_model(architecture: str, in_channels: int, num_classes: int) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; known: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[architecture](in_channels, num_classes)


def architecture_of(model: nn.Module) -> str:
    for name, cls in ARCHITECTURES.items():
        if isinstance(model, cls):
            return name
    raise ValueError(f"unregistered model class {type(model).__name__}")


def parameter_groups(model_or_arch: nn.Module | str) -> list[tuple[str, list[str]]]:
    """Ordered (group, prefixes) list for an architecture; every parameter key matches exactly one group."""
    arch = model_or_arch if isinstance(model_or_arch, str) else architecture_of(model_or_arch)
    if arch not in PARAMETER_GROUPS:
        raise ValueError(f"no parameter groups defined for {arch!r}")
    return [(g, list(p)) for g, p in PARAMETER_GROUPS[arch]]


def parameter_keys(model: nn.Module) -> list[str]:
    """State-dict keys that are trainable parameters (excludes BatchNorm buffers)."""
    return [k for k, _ in model.named_parameters()]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    model.load_state_dict({k: v.clone() for k, v in state.items()})
    return model
