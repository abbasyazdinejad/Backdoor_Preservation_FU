import sys, torchvision
from torchvision import datasets
root = sys.argv[1] if len(sys.argv)>1 else "data"
print("CIFAR10"); datasets.CIFAR10(root=root, train=True, download=True); datasets.CIFAR10(root=root, train=False, download=True)
print("MNIST"); datasets.MNIST(root=root, train=True, download=True); datasets.MNIST(root=root, train=False, download=True)
print("GTSRB"); datasets.GTSRB(root=root, split="train", download=True); datasets.GTSRB(root=root, split="test", download=True)
print("DONE")
