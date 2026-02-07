import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import ctypes
import numpy as np
import time
import os
import sys

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Load CUDA library (cross-platform support)
if sys.platform == 'win32':
    lib_path = './matmul.dll'
    if not os.path.exists(lib_path):
        lib_path = './libmatmul.so'  # Fallback
else:
    lib_path = './libmatmul.so'

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"CUDA library not found at {lib_path}. Please compile first.")

cuda_lib = ctypes.CDLL(lib_path)
cuda_lib.matmul_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
cuda_lib.matmul_forward.restype = None

class CudaMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        M, K, N = input.shape[0], input.shape[1], weight.shape[0]

        A_np = np.ascontiguousarray(input.detach().cpu().numpy(), dtype=np.float32)
        B_np = np.ascontiguousarray(weight.detach().cpu().numpy().T, dtype=np.float32)
        C_np = np.zeros((M, N), dtype=np.float32)

        cuda_lib.matmul_forward(
            A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K, N
        )

        output = torch.from_numpy(C_np)
        if bias is not None:
            output = output + bias.cpu()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = x.view(x.shape[0], -1).cpu().contiguous()
        return CudaMatmulFunction.apply(x, self.weight, self.bias)

class SimpleCNN(nn.Module):
    def __init__(self, use_custom=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = CustomLinear(64 * 7 * 7, 128) if use_custom else nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(use_custom=False, epochs=2, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = SimpleCNN(use_custom=use_custom)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*50}")
    print(f"Training with {'CUSTOM CUDA KERNEL' if use_custom else 'PyTorch'}")
    print(f"{'='*50}\n")

    for epoch in range(epochs):
        model.train()
        total_loss, start = 0, time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        model.eval()
        correct = sum(model(data).argmax(1).eq(target).sum().item() 
                     for data, target in test_loader)

        acc = 100. * correct / len(test_data)
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
              f'Accuracy={acc:.2f}%, Time={time.time()-start:.2f}s\n')
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-custom', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    set_seed(42)
    train(use_custom=args.use_custom, epochs=args.epochs)
