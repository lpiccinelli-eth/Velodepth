import torch
import torch.nn as nn

from velodepth.utils.constants import VERBOSE
from velodepth.utils.misc import profile_method

from .utils import FNS, REGRESSION_DICT, masked_mean, masked_quantile


class Scale(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        input_fn: str = "disp",
        fn: str = "l1",
        quantile: float = 0.0,
        gamma: float = 1.0,
        alpha: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = [-2, -1]
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.gamma = gamma
        self.alpha = alpha
        self.quantile = quantile
        self.eps = eps

    @profile_method(verbose=VERBOSE)
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        quality: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()
        input = self.input_fn(input.float())
        target = self.input_fn(target.float())

        error = self.fn(target - input, alpha=self.alpha, gamma=self.gamma)

        if self.quantile > 0.0:
            if quality is not None:
                for quality_level in [1, 2]:
                    current_quality = quality == quality_level
                    if current_quality.sum() > 0:
                        error_qtl = error[current_quality].detach().abs()
                        mask_qtl = error_qtl < masked_quantile(
                            error_qtl,
                            mask[current_quality],
                            dims=[1, 2, 3],
                            q=1 - self.quantile * quality_level,
                        ).view(-1, 1, 1, 1)
                        mask[current_quality] = mask[current_quality] & mask_qtl
            else:
                error_qtl = error.detach().abs()
                mask = mask & (
                    error_qtl
                    < masked_quantile(
                        error_qtl, mask, dims=[1, 2, 3], q=1 - self.quantile
                    ).view(-1, 1, 1, 1)
                )

        error_image = masked_mean(data=error, mask=mask, dim=self.dims).squeeze(1, 2, 3)

        error_image = self.output_fn(error_image)
        return error_image

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            input_fn=config["input_fn"],
            fn=config["fn"],
            output_fn=config["output_fn"],
            gamma=config["gamma"],
            alpha=config["alpha"],
            quantile=config.get("quantile", 0.1),
        )
        return obj
