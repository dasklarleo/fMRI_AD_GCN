from torch._C import device
import model
import torch
import torch.nn.functional as F


device='cuda' if torch.cuda.is_available() else 'cpu'
    