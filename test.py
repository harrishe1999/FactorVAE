import numpy
import pandas 
import torch
import torch.nn as nn

import torch

# Create a 1D tensor with shape (10,)
a = torch.randn(10)

print("Shape of a:", a.shape)

# Use unsqueeze to add a singleton dimension at the end
b = a.unsqueeze(-1)

print("Shape of b:", b.shape)
