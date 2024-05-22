"""Defines our P+O prediction models."""
import torch
from torch import nn

class ValueModel(nn.Module):
    """
    Defines a model that maps data features to item value for the Knapsack problem
    of the form v = a * alpha + b where a and b are input features, and alpha
    is a learnable parameter.

    Implemented from https://doi.org/10.1609/aaai.v34i02.5502 
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input[0, :] * self.alpha + input[1, :]).unsqueeze(dim=0)
