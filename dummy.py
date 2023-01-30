# from whisper.transcribe import cli

# cli()

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        weight = weight.to(x.dtype)
        return super()._conv_forward(
            x, weight, None if bias is None else bias.to(x.dtype)
        )

conv = Conv1d(80, 384, kernel_size=3, padding=1)
conv = conv.cuda()

x = torch.rand(1, 80, 3000, dtype=torch.float16)
x = x.cuda()

conv(x)