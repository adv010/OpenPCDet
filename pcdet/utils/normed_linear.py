# From https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/models/resnext.py
# Not used currently!
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(3, 256))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x.squeeze(), dim=1).mm(F.normalize(self.weight, dim=0)) #x.squeeze() to match shapes
        return self.s * out