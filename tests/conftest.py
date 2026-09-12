import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pytest, torch
from bpa_fu.datasets import TensorImageDataset

@pytest.fixture
def tiny_dataset():
    g = torch.Generator().manual_seed(0)
    imgs = torch.randint(0, 256, (600, 3, 32, 32), generator=g, dtype=torch.uint8)
    labels = torch.arange(600) % 10
    return TensorImageDataset(imgs, labels)
