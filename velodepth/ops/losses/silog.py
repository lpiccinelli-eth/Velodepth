import torch
import torch.nn as nn

from einops import rearrange

from velodepth.utils.constants import VERBOSE
from velodepth.utils.misc import profile_method

from .utils import (FNS, REGRESSION_DICT, masked_mean, masked_mean_var,
                    masked_quantile)


class SILog(nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str = "linear",
        output_fn: str = "sqrt",
        fn: str = "l1",
        integrated: bool = False,
        dims: bool = (-3, -2, -1),
        quantile: float = 0.0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight

        self.dims = dims
        self.input_fn = FNS[input_fn]
        self.output_fn = FNS[output_fn]
        self.fn = REGRESSION_DICT[fn]
        self.eps: float = eps
        self.integrated = integrated
        self.quantile = quantile
        self.alpha = alpha
        self.gamma = gamma

    @profile_method(verbose=VERBOSE)
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        si: torch.Tensor | None = None,
        B: int | None = None,
        T: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()
        if B is not None:
            input = rearrange(input, "(b t) c h w -> b (t c) h w", b=B, t=T)
            target = rearrange(target, "(b t) c h w -> b (t c) h w", b=B, t=T)
            mask = rearrange(mask, "(b t) c h w -> b (t c) h w", b=B, t=T)
            quality = rearrange(quality, "(b t) -> b t", b=B, t=T)
        
        B_, C, H, W = input.shape
        if si.any():
            rescale = torch.stack(
                [x[m > 0].median() for x, m in zip(target, target)]
            ) / torch.stack([x[m > 0].detach().median() for x, m in zip(input, target)])
            if rescale.isnan().any():
                print(
                    "NaN in rescale", rescale.isnan().squeeze(), mask.sum(dim=[1, 2, 3])
                )
                rescale = torch.nan_to_num(rescale, nan=1.0)
            input = (1 - si.int()).view(-1, 1, 1, 1) * input + (
                rescale * si.int()
            ).view(-1, 1, 1, 1) * input

        error = self.input_fn(input.float()) - self.input_fn(target.float())
        error_qtl = error.detach().abs()
        mask = mask & (
            error_qtl
            < masked_quantile(
                error_qtl, mask, dims=[1, 2, 3], q=1 - self.quantile
            ).view(-1, 1, 1, 1)
        )

        mean_error, var_error = masked_mean_var(
            data=error, mask=mask, dim=self.dims, keepdim=False
        )
        if var_error.ndim > 1:
            var_error = var_error.mean(dim=-1)

        if self.integrated > 0.0:
            scale_error = masked_mean(
                self.fn(error, alpha=self.alpha, gamma=self.gamma),
                mask=mask,
                dim=self.dims,
            ).reshape(-1)
            var_error = var_error + self.integrated * scale_error

        out_loss = self.output_fn(var_error)
        if out_loss.isnan().any():
            print(
                "NaN in SILog variance, input, target, mask, target>0, error",
                var_error.isnan().squeeze(),
                input[mask].isnan().any(),
                target[mask].isnan().any(),
                mask.any(dim=[1, 2, 3]),
                (target > 0.0).any(dim=[1, 2, 3]),
                error[mask].isnan().any(),
            )
        return out_loss

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            dims=config["dims"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            fn=config["fn"],
            alpha=config["alpha"],
            gamma=config["gamma"],
            integrated=config.get("integrated", False),
            quantile=config["quantile"],
        )
        return obj
