import torch

from modules.SeparatedLinear import SeparatedLinear

class SeparatedLinearGate(SeparatedLinear):
    def forward(self, input):
        return input * torch.sigmoid(input * self.weights + self.bias) 