import math
import torch
import torch.nn as nn

class SeparatedLinear(nn.Module):
    def __init__(self, channels, dtype=torch.float32):
        super(SeparatedLinear, self).__init__()
        self.requires_grad_(True)
        self.channels = channels

        self.weights = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weights, 0, 1)
        if self.bias is not None:
            fan_in = self.channels
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return input * self.weights + self.bias
    