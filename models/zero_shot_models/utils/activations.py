from torch import nn
from torch.nn import LeakyReLU, ReLU, CELU, SELU, SiLU

LeakyReLU
ReLU
CELU
SELU
SiLU


class GELU(nn.GELU):
    # Standard torch.nn.GELU (exact erf form). The only difference: FcOutModel.get_act() always
    # passes inplace=..., which nn.GELU rejects, so it's accepted here and ignored (GELU has no
    # in-place form).
    def __init__(self, inplace=False):
        super().__init__()
