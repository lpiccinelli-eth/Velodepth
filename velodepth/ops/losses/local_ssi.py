import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodepth.utils.constants import VERBOSE
from velodepth.utils.geometric import downsample, erode
from velodepth.utils.misc import profile_method

from .utils import (FNS, REGRESSION_DICT, ind2sub, masked_mean,
                    masked_quantile, ssi, ssi_nd)


def sample_strong_edges(edges_img, quantile=0.95, reshape=8):
    # flat
    edges_img = F.interpolate(
        edges_img, scale_factor=1 / reshape, mode="bilinear", align_corners=False
    )
    edges_img_flat = edges_img.flatten(1)

    # Find strong edges
    edges_mask = edges_img_flat > torch.quantile(
        edges_img_flat, quantile, dim=-1, keepdim=True
    )
    num_samples = edges_mask.sum(dim=-1)
    if (num_samples < 10).any():
        # sample random edges where num_samples < 2
        random = torch.rand_like(edges_img_flat[num_samples < 10, :]) > quantile
        edges_mask[num_samples < 10, :] = torch.logical_or(
            edges_mask[num_samples < 10, :], random
        )
        num_samples = edges_mask.sum(dim=-1)

    min_samples = num_samples.min()

    # Compute the coordinates of the strong edges as B, N, 2
    edges_coords = torch.stack(
        [torch.nonzero(x, as_tuple=False)[:min_samples].squeeze() for x in edges_mask]
    )
    edges_coords = (
        torch.stack(ind2sub(edges_coords, edges_img.shape[-1]), dim=-1) * reshape
    )
    return edges_coords


@torch.jit.script
def extract_patches(tensor, sample_coords, patch_size: tuple[int, int] = (32, 32)):
    """
    Extracts patches around specified edge coordinates, with zero padding.

    Parameters:
    - tensor: tenosr to be gatherd based on sampling (B, 1, H, W).
    - sample_coords: Batch of edge coordinates as a PyTorch tensor of shape (B, num_coords, 2).
    - patch_size: Tuple (width, height) representing the size of the patches.

    Returns:
    - Patches as a PyTorch tensor of shape (B, num_coords, patch_height, patch_width).
    """

    N, _, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    patch_width, patch_height = patch_size
    pad_width = patch_width // 2
    pad_height = patch_height // 2

    # Pad the RGB images for both sheep
    tensor_padded = F.pad(
        tensor,
        (pad_width, pad_width, pad_height, pad_height),
        mode="constant",
        value=0.0,
    )

    # Adjust edge coordinates to account for padding
    sample_coords_padded = sample_coords + torch.tensor(
        [pad_height, pad_width], dtype=dtype, device=device
    ).reshape(1, 1, 2)

    # Calculate the indices for gather operation
    x_centers = sample_coords_padded[:, :, 1].int()
    y_centers = sample_coords_padded[:, :, 0].int()

    all_patches = []
    for tensor_i, x_centers_i, y_centers_i in zip(tensor_padded, x_centers, y_centers):
        patches = []
        for x_center, y_center in zip(x_centers_i, y_centers_i):
            y_start, y_end = y_center - pad_height, y_center + pad_height + 1
            x_start, x_end = x_center - pad_width, x_center + pad_width + 1
            patches.append(tensor_i[..., y_start:y_end, x_start:x_end])
        all_patches.append(torch.stack(patches, dim=0))

    return torch.stack(all_patches, dim=0).reshape(N, -1, patch_height * patch_width)


class LocalSSI(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        patch_size: tuple[int, int] = (32, 32),
        min_samples: int = 4,
        num_levels: int = 4,
        fn: str = "l1",
        rescale_fn: str = "ssi",
        input_fn: str = "linear",
        quantile: float = 0.1,
        gamma: float = 1.0,
        alpha: float = 1.0,
        relative: bool = False,
        eps: float = 1e-5,
    ):
        super(LocalSSI, self).__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.fn = REGRESSION_DICT[fn]
        self.min_samples = min_samples
        self.eps = eps
        patch_logrange = np.linspace(
            start=np.log2(min(patch_size)),
            stop=np.log2(max(patch_size)),
            endpoint=True,
            num=num_levels + 1,
        )
        self.patch_logrange = [
            (x, y) for x, y in zip(patch_logrange[:-1], patch_logrange[1:])
        ]
        self.rescale_fn = eval(rescale_fn)
        self.quantile = quantile
        self.gamma = gamma
        self.alpha = alpha
        self.relative = relative

    @profile_method(verbose=VERBOSE)
    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        quality: torch.Tensor = None,
        down_ratio: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()

        if down_ratio > 1:
            input = downsample(input, down_ratio)
            target = downsample(target, down_ratio)
            # downsample will ignore 0s in the patch "min", if there is a 1 -> set mask to 1 there
            mask = downsample(mask.float(), down_ratio).bool()

        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        B, C, H, W = input.shape
        total_errors = []

        # save = random() < - 0.001 and is_main_process()
        for ii, patch_logrange in enumerate(self.patch_logrange):

            log_kernel = (
                np.random.uniform(*patch_logrange)
                if self.training
                else np.mean(patch_logrange)
            )
            kernel_size = int(
                (2**log_kernel) * min(input.shape[-2:])
            )  # always smaller than min_shape
            kernel_size = (kernel_size, kernel_size)
            stride = (int(kernel_size[0] * 0.9), int(kernel_size[1] * 0.9))
            # stride = kernel_size

            # unfold is always exceeding right/bottom, roll image only negative
            # to have them back in the unfolding window
            max_roll = (
                (W - kernel_size[1]) % stride[1],
                (H - kernel_size[0]) % stride[0],
            )
            roll_x, roll_y = np.random.randint(-max_roll[0], 1), np.random.randint(
                -max_roll[1], 1
            )
            input_fold = torch.roll(input, shifts=(roll_y, roll_x), dims=(2, 3))
            target_fold = torch.roll(target, shifts=(roll_y, roll_x), dims=(2, 3))
            mask_fold = torch.roll(mask.float(), shifts=(roll_y, roll_x), dims=(2, 3))

            # unfold in patches
            input_fold = F.unfold(
                input_fold, kernel_size=kernel_size, stride=stride
            ).permute(
                0, 2, 1
            )  # B N C*H_p*W_p
            target_fold = F.unfold(
                target_fold, kernel_size=kernel_size, stride=stride
            ).permute(0, 2, 1)
            mask_fold = (
                F.unfold(mask_fold, kernel_size=kernel_size, stride=stride)
                .bool()
                .permute(0, 2, 1)
            )

            # calculate error patchwise, then mean over patch, then over image based if sample size is significant
            input_fold, target_fold, _ = self.rescale_fn(
                input_fold, target_fold, mask_fold, dim=(-1,)
            )
            error = self.fn(
                input_fold - target_fold, alpha=self.alpha, gamma=self.gamma
            )

            # calculate elements more then 95 percentile and lower than 5percentile of error
            if quality is not None:
                N_patches = mask_fold.shape[1]
                for quality_level in [1, 2]:
                    current_quality = quality == quality_level
                    if current_quality.sum() > 0:
                        error_qtl = error[current_quality].detach()
                        mask_qtl = error_qtl < masked_quantile(
                            error_qtl,
                            mask_fold[current_quality],
                            dims=[2],
                            q=1 - self.quantile * quality_level,
                        ).view(-1, N_patches, 1)
                        mask_fold[current_quality] = (
                            mask_fold[current_quality] & mask_qtl
                        )
            else:
                error_qtl = error.detach()
                mask_fold = mask_fold & (
                    error_qtl
                    < masked_quantile(
                        error_qtl, mask_fold, dims=[2], q=1 - self.quantile
                    ).view(B, -1, 1)
                )

            valid_patches = mask_fold.sum(dim=-1) >= self.min_samples
            error_mean_patch = masked_mean(error, mask_fold, dim=(-1,)).squeeze(-1)
            error_mean_image = self.output_fn(error_mean_patch.clamp(min=self.eps))
            error_mean_image = masked_mean(
                error_mean_image, mask=valid_patches, dim=(-1,)
            )
            total_errors.append(error_mean_image.squeeze(-1))

        # global
        input_rescale = input.reshape(B, C, -1).clone()
        target_rescale = target.reshape(B, C, -1)
        mask = mask.reshape(B, 1, -1).clone()
        input, target, _ = self.rescale_fn(
            input_rescale,
            target_rescale,
            mask,
            dim=(-1,),
            target_info=target_rescale.norm(dim=1, keepdim=True),
            input_info=input_rescale.norm(dim=1, keepdim=True),
        )
        error = input - target
        error = error.norm(dim=1) if C > 1 else error.squeeze(1)
        if self.relative:
            error = error * torch.log(
                1.0 + 10.0 / target_rescale.norm(dim=1).clip(min=0.01)
            )

        error = self.fn(error, alpha=self.alpha, gamma=self.gamma).squeeze(1)

        mask = mask.squeeze(1)
        valid_patches = mask.sum(dim=-1) >= 3 * self.min_samples  # 30 samples per image
        if quality is not None:
            for quality_level in [1, 2]:
                current_quality = quality == quality_level
                if current_quality.sum() > 0:
                    error_qtl = error[current_quality].detach()
                    mask_qtl = error_qtl < masked_quantile(
                        error_qtl,
                        mask[current_quality],
                        dims=[1],
                        q=1 - self.quantile * quality_level,
                    ).view(-1, 1)
                    mask[current_quality] = mask[current_quality] & mask_qtl
        else:
            error_qtl = error.detach()
            mask = mask & (
                error_qtl
                < masked_quantile(error_qtl, mask, dims=[1], q=1 - self.quantile).view(
                    -1, 1
                )
            )

        error_mean_image = masked_mean(error, mask, dim=(-1,)).squeeze(-1)
        error_mean_image = (
            self.output_fn(error_mean_image.clamp(min=self.eps)) * valid_patches.float()
        )

        total_errors.append(error_mean_image)

        errors = torch.stack(total_errors).mean(dim=0)
        return errors

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            patch_size=config["patch_size"],
            output_fn=config["output_fn"],
            min_samples=config["min_samples"],
            num_levels=config["num_levels"],
            input_fn=config["input_fn"],
            quantile=config["quantile"],
            gamma=config["gamma"],
            alpha=config["alpha"],
            rescale_fn=config["rescale_fn"],
            fn=config["fn"],
            relative=config["relative"],
        )
        return obj
