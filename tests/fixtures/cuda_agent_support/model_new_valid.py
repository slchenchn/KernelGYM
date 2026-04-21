import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, x):
        return x
