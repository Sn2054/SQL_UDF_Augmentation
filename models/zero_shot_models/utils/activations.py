from torch import nn
from torch.nn import LeakyReLU, ReLU, CELU, SELU
from torch.nn import functional as F

LeakyReLU
ReLU
CELU
SELU


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)
