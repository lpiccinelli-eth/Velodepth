import torch
import torch.nn as nn

from .utils import FNS, REGRESSION_DICT, masked_mean, masked_quantile


class Regression(nn.Module):
    def __init__(
        self,
        weight: float,
        gamma: float,
        fn: str,
        input_fn: str,
        output_fn: str,
        alpha: float = 1.0,
        dims: tuple[int] = (-1,),
        quantile: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.dims = dims
        self.quantile = quantile

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is not None:  # usually it is just repeated
            mask = mask[:, 0]

        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        error = self.fn(input - target, gamma=self.gamma, alpha=self.alpha).mean(dim=1)
        if self.quantile > 0.0:
            mask_quantile = error < masked_quantile(
                error, mask, dims=self.dims, q=1 - self.quantile
            ).view(-1, *((1,) * len(self.dims)))
            mask = mask & mask_quantile if mask is not None else mask_quantile
        mean_error = masked_mean(data=error, mask=mask, dim=self.dims).squeeze(
            self.dims
        )
        mean_error = self.output_fn(mean_error)
        return mean_error

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            fn=config["fn"],
            gamma=config["gamma"],
            alpha=config.get("alpha", 1.0),
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            dims=config.get("dims", (-1,)),
            quantile=config.get("quantile", 0.0),
        )
        return obj


class PolarRegression(nn.Module):
    def __init__(
        self,
        weight: float,
        gamma: float,
        fn: str,
        input_fn: str,
        output_fn: str,
        alpha: float = 1.0,
        dims: list[int] = [-1, -2],
        polar_weight: float = 1.0,
        polar_asym: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.dims = dims
        self.polar_weight = polar_weight
        self.polar_asym = polar_asym

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is not None:  # usually it is just repeated
            mask = mask.squeeze(1)

        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        input = input / torch.norm(input, dim=1, keepdim=True).clamp(min=1e-5)
        target = target / torch.norm(target, dim=1, keepdim=True).clamp(min=1e-5)

        x_target, y_target, z_target = target.unbind(dim=1)
        z_clipped = z_target.clip(min=-0.99999, max=0.99999)
        x_clipped = x_target.abs().clip(min=1e-5) * (2 * (x_target > 0).float() - 1)
        polar_target = torch.arccos(z_clipped)
        azimuth_target = torch.atan2(y_target, x_clipped)

        x_input, y_input, z_input = input.unbind(dim=1)
        z_clipped = z_input.clip(min=-0.99999, max=0.99999)
        x_clipped = x_input.abs().clip(min=1e-5) * (2 * (x_input > 0).float() - 1)
        polar_input = torch.arccos(z_clipped)
        azimuth_input = torch.atan2(y_input, x_clipped)

        polar_error = self.fn(
            polar_input - polar_target, gamma=self.gamma, alpha=self.alpha
        )
        azimuth_error = self.fn(
            azimuth_input - azimuth_target, gamma=self.gamma, alpha=self.alpha
        )

        quantile_weight = torch.ones_like(polar_input)
        quantile_weight[
            (polar_target > polar_input) & (polar_target > torch.pi / 2)
        ] = (2 * self.polar_asym)
        quantile_weight[
            (polar_target <= polar_input) & (polar_target > torch.pi / 2)
        ] = 2 * (1 - self.polar_asym)

        mean_polar_error = masked_mean(
            data=polar_error * quantile_weight, mask=mask, dim=self.dims
        ).squeeze(self.dims)
        mean_azimuth_error = masked_mean(
            data=azimuth_error, mask=mask, dim=self.dims
        ).squeeze(self.dims)
        mean_error = (self.polar_weight * mean_polar_error + mean_azimuth_error) / (
            1 + self.polar_weight
        )

        mean_error = self.output_fn(mean_error)
        return mean_error

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            fn=config["fn"],
            gamma=config["gamma"],
            alpha=config.get("alpha", 1.0),
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            dims=config.get("dims", (-1,)),
            polar_weight=config["polar_weight"],
            polar_asym=config["polar_asym"],
        )
        return obj
