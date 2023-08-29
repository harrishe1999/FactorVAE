import numpy
import pandas 
import torch
import torch.nn as nn

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print(A*B)