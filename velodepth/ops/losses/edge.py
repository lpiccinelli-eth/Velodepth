import torch
import torch.nn as nn
import torch.nn.functional as F

from velodepth.utils.constants import VERBOSE
from velodepth.utils.geometric import dilate, erode
from velodepth.utils.misc import profile_method

from .utils import (FNS, REGRESSION_DICT, masked_mean, masked_mean_var,
                    masked_quantile)


class SpatialGradient(torch.nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str,
        output_fn: str,
        fn: str,
        scales: int = 1,
        gamma: float = 1.0,
        quantile: float = 0.0,
        laplacian: bool = False,
        canny_edge: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.gamma = gamma
        sobel_kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        laplacian_kernel = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ones = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, requires_grad=False)
        self.ones = nn.Parameter(ones, requires_grad=False)
        self.laplacian_kernel = nn.Parameter(laplacian_kernel, requires_grad=False)

        self.quantile = quantile
        self.scales = scales
        self.laplacian = laplacian
        self.canny_edge = canny_edge

    @profile_method(verbose=VERBOSE)
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input,
        target,
        mask,
        quality=None,
    ):
        B = input.shape[0]
        input = self.input_fn(input.float())
        target = self.input_fn(target.float())

        # normalize to avoid scale issue, shift is not important as we are computing gradients
        input_mean, input_var = masked_mean_var(input.detach(), mask, dim=(-3, -2, -1))
        target_mean, target_var = masked_mean_var(target, mask, dim=(-3, -2, -1))
        input = (input - input_mean) / (input_var + 1e-6) ** 0.5
        target = (target - target_mean) / (target_var + 1e-6) ** 0.5

        loss = 0.0
        norm_factor = sum([(i + 1) ** 2 for i in range(self.scales)])
        for scale in range(self.scales):
            if scale > 0:
                input = F.interpolate(
                    input,
                    size=(input.shape[-2] // 2, input.shape[-1] // 2),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                target = F.interpolate(
                    target,
                    size=(target.shape[-2] // 2, target.shape[-1] // 2),
                    mode="nearest",
                )
                mask = (
                    F.interpolate(
                        mask.float(),
                        size=(mask.shape[-2] // 2, mask.shape[-1] // 2),
                        mode="nearest",
                    )
                    > 0.9
                )
            grad_loss = self.loss(input, target, mask, quality)
            # keep per pixel same weight
            loss = loss + grad_loss * (self.scales - scale) ** 2 / norm_factor

        loss = self.output_fn(loss)
        return loss

    def loss(self, input, target, mask, quality):
        device, dtype = input.device, input.dtype
        B, C, H, W = input.shape

        # sobel
        input_edge_x = (
            F.conv2d(input, self.sobel_kernel_x.repeat(C, 1, 1, 1), groups=C) / 8
        )
        target_edge_x = (
            F.conv2d(target, self.sobel_kernel_x.repeat(C, 1, 1, 1), groups=C) / 8
        )
        input_edge_y = (
            F.conv2d(input, self.sobel_kernel_y.repeat(C, 1, 1, 1), groups=C) / 8
        )
        target_edge_y = (
            F.conv2d(target, self.sobel_kernel_y.repeat(C, 1, 1, 1), groups=C) / 8
        )
        input_edge = torch.stack([input_edge_x, input_edge_y], dim=-1)
        target_edge = torch.stack([target_edge_x, target_edge_y], dim=-1)

        mask = F.conv2d(mask.clone().to(input.dtype), self.ones) == 9
        mask = mask.squeeze(1)

        error = input_edge - target_edge
        error = error.norm(dim=-1).norm(
            dim=1
        )  # take RMSE over xy-dir (isotropic) and over channel-dir (isotropic)

        if quality is not None:
            for quality_level in [1, 2]:
                current_quality = quality == quality_level
                if current_quality.sum() > 0:
                    error_qtl = error[current_quality].detach()
                    mask_qtl = error_qtl < masked_quantile(
                        error_qtl,
                        mask[current_quality],
                        dims=[1, 2],
                        q=1 - self.quantile * quality_level,
                    ).view(-1, 1, 1)
                    mask[current_quality] = mask[current_quality] & mask_qtl
        else:
            error_qtl = error.detach()
            mask = mask & (
                error_qtl
                < masked_quantile(
                    error_qtl, mask, dims=[1, 2], q=1 - self.quantile
                ).view(-1, 1, 1)
            )

        loss = masked_mean(error, mask, dim=(-2, -1)).squeeze(dim=(-2, -1))

        if self.laplacian:
            input_laplacian = (
                F.conv2d(input, self.laplacian_kernel.repeat(C, 1, 1, 1), groups=C) / 8
            )
            target_laplacian = (
                F.conv2d(target, self.laplacian_kernel.repeat(C, 1, 1, 1), groups=C) / 8
            )
            error_laplacian = self.fn(
                input_laplacian - target_laplacian, gamma=self.gamma
            )
            error_laplacian = (torch.mean(error_laplacian**2, dim=1) + 1e-6) ** 0.5
            loss_laplacian = masked_mean(error_laplacian, mask, dim=(-2, -1)).squeeze(
                dim=(-2, -1)
            )
            loss = loss + 0.1 * loss_laplacian

        return loss

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            input_fn=config["input_fn"],
            output_fn=config["output_fn"],
            fn=config["fn"],
            gamma=config["gamma"],
            quantile=config["quantile"],
            scales=config["scales"],
            laplacian=config["laplacian"],
        )
        return obj


class SpatioTemporalGradient(torch.nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str,
        output_fn: str,
        fn: str,
        gamma: float = 1.0,
        quantile: float = 0.0,
        scales: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.gamma = gamma
        sobel_kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sobel_kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ones = torch.ones(1, 1, 2, 3, 3, dtype=torch.float32)
        self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, requires_grad=False)
        self.ones = nn.Parameter(ones, requires_grad=False)
        self.quantile = quantile
        self.scales = scales

    @profile_method(verbose=VERBOSE)
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(self, input, target, mask, B, T, weights=None):
        # shapes B*T, C, H, W -> B, T, C, H, W -> B, C, T, H, W  (stats on T-1, jump 1st)
        C, H, W = input.shape[-3:]
        input = self.input_fn(input.float())
        input = input.reshape(B, T, *input.shape[1:])
        target = self.input_fn(target.float())
        target = target.reshape(B, T, *target.shape[1:])
        mask = mask.reshape(B, T, *mask.shape[1:]).bool()

        # normalize based on time also !
        input_mean, input_var = masked_mean_var(
            input[:, 1:].detach(), mask[:, 1:], dim=(-4, -3, -2, -1)
        )
        target_mean, target_var = masked_mean_var(
            target[:, 1:], mask[:, 1:], dim=(-4, -3, -2, -1)
        )
        input = (input - input_mean) / (input_var + 1e-6) ** 0.5
        target = (target - target_mean) / (target_var + 1e-6) ** 0.5

        loss = 0.0
        norm_factor = sum([(i + 1) ** 2 for i in range(self.scales)])
        for scale in range(self.scales):
            # resize to half shape
            if scale > 0:
                input = input.reshape(B * T, *input.shape[2:])
                target = target.reshape(B * T, *target.shape[2:])
                mask = mask.reshape(B * T, *mask.shape[2:])
                input = F.interpolate(
                    input,
                    size=(input.shape[-2] // 2, input.shape[-1] // 2),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                target = F.interpolate(
                    target,
                    size=(target.shape[-2] // 2, target.shape[-1] // 2),
                    mode="nearest",
                )
                mask = (
                    F.interpolate(
                        mask.float(),
                        size=(mask.shape[-2] // 2, mask.shape[-1] // 2),
                        mode="nearest",
                    )
                    > 0.9
                )
                input = input.reshape(B, T, *input.shape[1:])
                target = target.reshape(B, T, *target.shape[1:])
                mask = mask.reshape(B, T, *mask.shape[1:])
                if weights is not None:
                    weights = F.interpolate(
                        weights.float(),
                        size=(weights.shape[-2] // 2, weights.shape[-1] // 2),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )

            grad_loss = self.loss(
                input.permute(0, 2, 1, 3, 4),
                target.permute(0, 2, 1, 3, 4),
                mask.permute(0, 2, 1, 3, 4),
                weights,
            )
            # keep per pixel same weight
            loss = loss + grad_loss * (self.scales - scale) ** 2 / norm_factor

        loss = self.output_fn(loss)

        loss = self.output_fn(loss)
        return loss

    def loss(self, input, target, mask, weights):
        device, dtype = input.device, input.dtype
        B, C, T, H, W = input.shape
        assert C == 1

        # sobel
        input_edge_x = (
            F.conv2d(
                input[:, :, 1:].reshape(B, C * (T - 1), H, W),
                self.sobel_kernel_x.repeat(C * (T - 1), 1, 1, 1),
                groups=C * (T - 1),
            )
            / 8
        )
        target_edge_x = (
            F.conv2d(
                target[:, :, 1:].reshape(B, C * (T - 1), H, W),
                self.sobel_kernel_x.repeat(C * (T - 1), 1, 1, 1),
                groups=C * (T - 1),
            )
            / 8
        )
        input_edge_y = (
            F.conv2d(
                input[:, :, 1:].reshape(B, C * (T - 1), H, W),
                self.sobel_kernel_y.repeat(C * (T - 1), 1, 1, 1),
                groups=C * (T - 1),
            )
            / 8
        )
        target_edge_y = (
            F.conv2d(
                target[:, :, 1:].reshape(B, C * (T - 1), H, W),
                self.sobel_kernel_y.repeat(C * (T - 1), 1, 1, 1),
                groups=C * (T - 1),
            )
            / 8
        )
        input_edge_t = (
            input[:, :, 1:, 1:-1, 1:-1] - input[:, :, :-1, 1:-1, 1:-1]
        ).reshape(B, C * (T - 1), H - 2, W - 2)
        target_edge_t = (
            target[:, :, 1:, 1:-1, 1:-1] - target[:, :, :-1, 1:-1, 1:-1]
        ).reshape(B, C * (T - 1), H - 2, W - 2)
        input_edge = torch.stack([input_edge_x, input_edge_y, input_edge_t], dim=-1)
        target_edge = torch.stack([target_edge_x, target_edge_y, target_edge_t], dim=-1)

        mask = F.conv3d(mask.clone().to(input.dtype), self.ones) == 18

        error = input_edge - target_edge
        error = error.norm(dim=-1).reshape(
            B * (T - 1), H - 2, W - 2
        )  # take RMSE over xy-dir (isotropic)
        mask = mask.reshape(B * (T - 1), H - 2, W - 2)

        if self.quantile > 0.0:
            error_qtl = error.detach()
            mask = mask & (
                error_qtl
                < masked_quantile(
                    error_qtl, mask, dims=[1, 2], q=1 - self.quantile
                ).view(-1, 1, 1)
            )
        if weights is not None:
            error = error * weights[..., 1:-1, 1:-1].reshape(B * (T - 1), H - 2, W - 2)
        loss = (
            masked_mean(error, mask, dim=(1, 2))
            .squeeze(dim=(1, 2))
            .reshape(B * (T - 1))
        )

        return loss

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            input_fn=config["input_fn"],
            output_fn=config["output_fn"],
            fn=config["fn"],
            gamma=config["gamma"],
            quantile=config["quantile"],
            scales=config["scales"],
            focal=config["focal"],
            canny_edge=config["canny_edge"],
            relative=config["relative"],
        )
        return obj
