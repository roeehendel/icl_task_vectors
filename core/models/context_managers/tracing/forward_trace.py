from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch


@dataclass
class ResidualStream:
    hidden: torch.Tensor
    attn: torch.Tensor
    mlp: torch.Tensor


class ForwardTrace:
    def __init__(self):
        self.residual_stream: Optional[ResidualStream] = ResidualStream(
            hidden=[],
            attn=[],
            mlp=[],
        )
        self.attentions: Optional[torch.Tensor] = None
