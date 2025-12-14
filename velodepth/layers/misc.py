import torch
import torch.nn as nn

from .layer_scale import LayerScale


class Addition(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float | torch.Tensor = 1e-5,
    ) -> None:
        super().__init__()
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + self.ls1(y)


class Concat(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float | torch.Tensor = 1e-5,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([x, y], dim=-1))
