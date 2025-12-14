import os
import random
from copy import deepcopy
from math import ceil, exp, log, log2, log10, tanh
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from velodepth.utils.geometric import downsample


def euler_to_rotation_matrix(angles):
    """
    Convert Euler angles to a 3x3 rotation matrix.

    Args:
        angles (torch.Tensor): Euler angles [roll, pitch, yaw].

    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    phi, theta, psi = angles
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)

    # Rotation matrices
    Rx = torch.tensor([[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]])

    Ry = torch.tensor(
        [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    )

    Rz = torch.tensor([[cos_psi, -sin_psi, 0], [sin_psi, cos_psi, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def compute_grid(H, W):
    meshgrid = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    id_coords = torch.stack(meshgrid, axis=0).to(torch.float32)
    id_coords = id_coords.reshape(2, -1)
    id_coords = torch.cat(
        [id_coords, torch.ones(1, id_coords.shape[-1])], dim=0
    )  # 3 HW
    id_coords = id_coords.unsqueeze(0)
    return id_coords


def lexsort(keys):
    sorted_indices = torch.arange(keys[0].size(0))
    for key in reversed(keys):
        _, sorted_indices = key[sorted_indices].sort()
    return sorted_indices


def masked_bilinear_interpolation(input, mask, target_size):
    B, C, H, W = input.shape
    target_H, target_W = target_size
    mask = mask.float()

    # Generate a grid of coordinates in the target space
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, target_H), torch.linspace(0, W - 1, target_W)
    )
    grid_y = grid_y.to(input.device)
    grid_x = grid_x.to(input.device)

    # Calculate the floor and ceil of the grid coordinates to get the bounding box
    x0 = torch.floor(grid_x).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(grid_y).long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # Gather depth values at the four corners
    Ia = input[..., y0, x0]
    Ib = input[..., y1, x0]
    Ic = input[..., y0, x1]
    Id = input[..., y1, x1]

    # Gather corresponding mask values
    ma = mask[..., y0, x0]
    mb = mask[..., y1, x0]
    mc = mask[..., y0, x1]
    md = mask[..., y1, x1]

    # Calculate the areas (weights) for bilinear interpolation
    wa = (x1.float() - grid_x) * (y1.float() - grid_y)
    wb = (x1.float() - grid_x) * (grid_y - y0.float())
    wc = (grid_x - x0.float()) * (y1.float() - grid_y)
    wd = (grid_x - x0.float()) * (grid_y - y0.float())

    wa = wa.reshape(1, 1, target_H, target_W).repeat(B, C, 1, 1)
    wb = wb.reshape(1, 1, target_H, target_W).repeat(B, C, 1, 1)
    wc = wc.reshape(1, 1, target_H, target_W).repeat(B, C, 1, 1)
    wd = wd.reshape(1, 1, target_H, target_W).repeat(B, C, 1, 1)

    # Only consider valid points for interpolation
    weights_sum = (wa * ma) + (wb * mb) + (wc * mc) + (wd * md)
    weights_sum = torch.clamp(weights_sum, min=1e-5)

    # Perform the interpolation
    interpolated_depth = (
        wa * Ia * ma + wb * Ib * mb + wc * Ic * mc + wd * Id * md
    ) / weights_sum

    return interpolated_depth, (ma + mb + mc + md) > 0


def masked_nearest_interpolation(input, mask, target_size):
    B, C, H, W = input.shape
    target_H, target_W = target_size
    mask = mask.float()

    # Generate a grid of coordinates in the target space
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, target_H),
        torch.linspace(0, W - 1, target_W),
        indexing="ij",
    )
    grid_y = grid_y.to(input.device)
    grid_x = grid_x.to(input.device)

    # Calculate the floor and ceil of the grid coordinates to get the bounding box
    x0 = torch.floor(grid_x).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(grid_y).long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # Gather depth values at the four corners
    Ia = input[..., y0, x0]
    Ib = input[..., y1, x0]
    Ic = input[..., y0, x1]
    Id = input[..., y1, x1]

    # Gather corresponding mask values
    ma = mask[..., y0, x0]
    mb = mask[..., y1, x0]
    mc = mask[..., y0, x1]
    md = mask[..., y1, x1]

    # Calculate distances to each neighbor
    # The distances are calculated from the center (grid_x, grid_y) to each corner
    dist_a = (grid_x - x0.float()) ** 2 + (grid_y - y0.float()) ** 2  # Top-left
    dist_b = (grid_x - x0.float()) ** 2 + (grid_y - y1.float()) ** 2  # Bottom-left
    dist_c = (grid_x - x1.float()) ** 2 + (grid_y - y0.float()) ** 2  # Top-right
    dist_d = (grid_x - x1.float()) ** 2 + (grid_y - y1.float()) ** 2  # Bottom-right

    # Stack the neighbors, their masks, and distances
    stacked_values = torch.stack(
        [Ia, Ib, Ic, Id], dim=-1
    )  # Shape: (B, C, target_H, target_W, 4)
    stacked_masks = torch.stack(
        [ma, mb, mc, md], dim=-1
    )  # Shape: (B, 1, target_H, target_W, 4)
    stacked_distances = torch.stack(
        [dist_a, dist_b, dist_c, dist_d], dim=-1
    )  # Shape: (target_H, target_W, 4)
    stacked_distances = (
        stacked_distances.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1, 1)
    )  # Shape: (B, 1, target_H, target_W, 4)

    # Set distances to infinity for invalid neighbors (so that invalid neighbors are never chosen)
    stacked_distances[stacked_masks == 0] = float("inf")

    # Find the index of the nearest valid neighbor (the one with the smallest distance)
    nearest_indices = stacked_distances.argmin(dim=-1, keepdim=True)[
        ..., :1
    ]  # Shape: (B, 1, target_H, target_W, 1)

    # Select the corresponding depth value using the nearest valid neighbor index
    interpolated_depth = torch.gather(
        stacked_values, dim=-1, index=nearest_indices.repeat(1, C, 1, 1, 1)
    ).squeeze(-1)

    # Set depth to zero where no valid neighbors were found
    interpolated_depth = interpolated_depth * stacked_masks.sum(dim=-1).clip(
        min=0.0, max=1.0
    )

    return interpolated_depth


def masked_nxn_interpolation(input, mask, target_size, N=2):
    B, C, H, W = input.shape
    target_H, target_W = target_size

    # Generate a grid of coordinates in the target space
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, target_H),
        torch.linspace(0, W - 1, target_W),
        indexing="ij",
    )
    grid_y = grid_y.to(input.device)
    grid_x = grid_x.to(input.device)

    # Calculate the top-left corner of the NxN grid
    half_N = (N - 1) // 2
    y0 = torch.floor(grid_y - half_N).long().clamp(0, H - 1)
    x0 = torch.floor(grid_x - half_N).long().clamp(0, W - 1)

    # Prepare to gather NxN neighborhoods
    input_patches = []
    mask_patches = []
    weights = []

    for i in range(N):
        for j in range(N):
            yi = (y0 + i).clamp(0, H - 1)
            xi = (x0 + j).clamp(0, W - 1)

            # Gather depth and mask values
            input_patches.append(input[..., yi, xi])
            mask_patches.append(mask[..., yi, xi])

            # Compute bilinear weights
            weight_y = 1 - torch.abs(grid_y - yi.float()) / N
            weight_x = 1 - torch.abs(grid_x - xi.float()) / N
            weight = (
                (weight_y * weight_x)
                .reshape(1, 1, target_H, target_W)
                .repeat(B, C, 1, 1)
            )
            weights.append(weight)

    input_patches = torch.stack(input_patches)
    mask_patches = torch.stack(mask_patches)
    weights = torch.stack(weights)

    # Calculate weighted sum and normalize by the sum of weights
    weighted_sum = (input_patches * mask_patches * weights).sum(dim=0)
    weights_sum = (mask_patches * weights).sum(dim=0)
    interpolated_tensor = weighted_sum / torch.clamp(weights_sum, min=1e-8)

    if N != 2:
        interpolated_tensor_2x2, mask_sum_2x2 = masked_bilinear_interpolation(
            input, mask, target_size
        )
        interpolated_tensor = torch.where(
            mask_sum_2x2, interpolated_tensor_2x2, interpolated_tensor
        )

    return interpolated_tensor


class PanoCrop:
    def __init__(self, crop_v=0.15):
        self.crop_v = crop_v

    def _crop_data(self, results, crop_size):
        offset_w, offset_h = crop_size
        left, top, right, bottom = offset_w[0], offset_h[0], offset_w[1], offset_h[1]
        H, W = results["image"].shape[-2:]
        for key in results.get("image_fields", ["image"]):
            img = results[key][..., top : H - bottom, left : W - right]
            results[key] = img
            results["image_shape"] = tuple(img.shape)

        for key in results.get("gt_fields", []):
            results[key] = results[key][..., top : H - bottom, left : W - right]

        for key in results.get("mask_fields", []):
            results[key] = results[key][..., top : H - bottom, left : W - right]

        results["camera"] = results["camera"].crop(left, top, right, bottom)
        return results

    def __call__(self, results):
        H, W = results["image"].shape[-2:]
        crop_w = (0, 0)
        crop_h = (int(H * self.crop_v), int(H * self.crop_v))
        results = self._crop_data(results, (crop_w, crop_h))
        return results


class PanoRoll:
    def __init__(self, test_mode, roll=[-0.5, 0.5]):
        self.roll = roll
        self.test_mode = test_mode

    def __call__(self, results):
        if self.test_mode:
            return results
        W = results["image"].shape[-1]
        roll = random.randint(int(W * self.roll[0]), int(W * self.roll[1]))
        for key in results.get("image_fields", ["image"]):
            img = results[key]
            img = torch.roll(img, roll, dims=-1)
            results[key] = img
        for key in results.get("gt_fields", []):
            results[key] = torch.roll(results[key], roll, dims=-1)
        for key in results.get("mask_fields", []):
            results[key] = torch.roll(results[key], roll, dims=-1)
        return results


class RandomFlip:
    def __init__(self, direction="horizontal", prob=0.5, consistent=False, **kwargs):
        self.flip_ratio = prob
        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError("direction must be either str or list of str")
        self.direction = direction
        self.consistent = consistent

    def __call__(self, results):
        if "flip" not in results:
            # None means non-flip
            if isinstance(self.direction, list):
                direction_list = self.direction + [None]
            else:
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [
                    non_flip_ratio
                ]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results["flip"] = cur_dir is not None

        if "flip_direction" not in results:
            results["flip_direction"] = cur_dir

        if results["flip"]:
            # flip image
            if results["flip_direction"] != "vertical":
                for key in results.get("image_fields", ["image"]):
                    results[key] = TF.hflip(results[key])
                for key in results.get("mask_fields", []):
                    results[key] = TF.hflip(results[key])
                for key in results.get("gt_fields", []):
                    results[key] = TF.hflip(results[key])
                    if "flow" in key:  # flip u direction
                        results[key][:, 0] = -results[key][:, 0]

                H, W = results["image"].shape[-2:]
                results["camera"] = results["camera"].flip(
                    H=H, W=W, direction="horizontal"
                )
                flip_transform = torch.tensor(
                    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=torch.float32,
                ).unsqueeze(0)
                repeats = (results["cam2w"].shape[0],) + (1,) * (
                    results["cam2w"].ndim - 1
                )
                results["cam2w"] = flip_transform.repeat(*repeats) @ results["cam2w"]

            if results["flip_direction"] != "horizontal":
                for key in results.get("image_fields", ["image"]):
                    results[key] = TF.vflip(results[key])
                for key in results.get("mask_fields", []):
                    results[key] = TF.vflip(results[key])
                for key in results.get("gt_fields", []):
                    results[key] = TF.vflip(results[key])
                    results["K"][..., 1, 2] = (
                        results["image"].shape[-2] - results["K"][..., 1, 2]
                    )
        results["flip"] = [results["flip"]] * len(results["image"])
        return results


class Crop:
    def __init__(
        self,
        crop_size,
        crop_type="absolute",
        crop_offset=(0, 0),
    ):
        if crop_type not in [
            "relative_range",
            "relative",
            "absolute",
            "absolute_range",
        ]:
            raise ValueError(f"Invalid crop_type {crop_type}.")
        if crop_type in ["absolute", "absolute_range"]:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.offset_h, self.offset_w = (
            crop_offset[: len(crop_offset) // 2],
            crop_offset[len(crop_offset) // 2 :],
        )

    def _get_crop_size(self, image_shape):
        h, w = image_shape
        if self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1
            )
            crop_w = np.random.randint(
                min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1
            )
            return crop_h, crop_w
        elif self.crop_type == "relative":
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def _crop_data(self, results, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get("image_fields", ["image"]):
            img = results[key]
            img = TF.crop(
                img, self.offset_h[0], self.offset_w[0], crop_size[0], crop_size[1]
            )
            results[key] = img
            results["image_shape"] = tuple(img.shape)

        for key in results.get("gt_fields", []):
            gt = results[key]
            results[key] = TF.crop(
                gt, self.offset_h[0], self.offset_w[0], crop_size[0], crop_size[1]
            )

        # crop semantic seg
        for key in results.get("mask_fields", []):
            mask = results[key]
            results[key] = TF.crop(
                mask, self.offset_h[0], self.offset_w[0], crop_size[0], crop_size[1]
            )

        results["K"][..., 0, 2] = results["K"][..., 0, 2] - self.offset_w[0]
        results["K"][..., 1, 2] = results["K"][..., 1, 2] - self.offset_h[0]
        return results

    def __call__(self, results):
        image_shape = results["image"].shape[-2:]
        crop_size = self._get_crop_size(image_shape)
        results = self._crop_data(results, crop_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size={self.crop_size}, "
        repr_str += f"crop_type={self.crop_type}, "
        return repr_str


class KittiCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def _crop_data(self, results, crop_size):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'image_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get("image_fields", ["image"]):
            img = results[key]
            h, w = img.shape[-2:]
            offset_h, offset_w = int(h - self.crop_size[0]), int(
                (w - self.crop_size[1]) / 2
            )

            # crop the image
            img = TF.crop(img, offset_h, offset_w, crop_size[0], crop_size[1])
            results[key] = img
            results["image_shape"] = tuple(img.shape)

        for key in results.get("gt_fields", []):
            gt = results[key]
            results[key] = TF.crop(gt, offset_h, offset_w, crop_size[0], crop_size[1])

        # crop semantic seg
        for key in results.get("mask_fields", []):
            mask = results[key]
            results[key] = TF.crop(mask, offset_h, offset_w, crop_size[0], crop_size[1])

        results["camera"] = results["camera"].crop(offset_w, offset_h)
        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'image_shape' key in result dict is
                updated according to crop size.
        """
        results = self._crop_data(results, self.crop_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size={self.crop_size}, "
        return repr_str


class RandomMasking:
    def __init__(
        self,
        mask_ratio,
        mask_patch=16,
        prob=0.5,
        warmup_steps=50000,
        sampling="random",
        curriculum=False,
    ):
        self.mask_patch = mask_patch
        self.prob = prob
        self.mask_ratio = mask_ratio
        self.warmup_steps = max(1, warmup_steps)
        self.hard_bound = 1
        self.idx = 0
        self.curriculum = curriculum
        self.sampling = sampling
        self.low_bound = 0.0
        self.up_bound = 0.0

    def __call__(self, results):
        B, _, H, W = results["image"].shape
        device = results["image"].device
        down_size = H // self.mask_patch, W // self.mask_patch
        if np.random.random() > self.prob:  # fill with dummy
            return self._nop(results, down_size, device)

        validity_mask = results["validity_mask"].float().reshape(B, -1, H, W)
        validity_mask = F.interpolate(validity_mask, size=down_size).bool()
        validity_mask = validity_mask.reshape(B, 1, *down_size)
        is_random = self.is_warmup or results.get("guidance") is None

        if not is_random:
            guidance = F.interpolate(results["guidance"], size=(H, W), mode="bilinear")
            results["guidance"] = -F.max_pool2d(
                -guidance, kernel_size=self.mask_patch, stride=self.mask_patch
            )

        if is_random and self.sampling == "inverse":
            sampling = self.inverse_sampling
        elif is_random and self.sampling == "random":
            sampling = self.random_sampling
        else:
            sampling = self.guided_sampling
        mask_ratio = np.random.uniform(self.low_bound, self.up_bound)
        for key in results.get("image_fields", ["image"]):
            mask = sampling(results, mask_ratio, down_size, validity_mask, device)
            results[key + "_mask"] = mask
        return results

    def _nop(self, results, down_size, device):
        B = results["image"].shape[0]
        for key in results.get("image_fields", ["image"]):
            mask_blocks = torch.zeros(size=(B, 1, *down_size), device=device)
            results[key + "_mask"] = mask_blocks
        return results

    def random_sampling(self, results, mask_ratio, down_size, validity_mask, device):
        B = results["image"].shape[0]
        prob_blocks = torch.rand(size=(B, 1, *down_size), device=device)
        mask_blocks = torch.logical_and(prob_blocks < mask_ratio, validity_mask)
        return mask_blocks

    def inverse_sampling(self, results, mask_ratio, down_size, validity_mask, device):
        # from PIL import Image
        # from velodepth.utils import colorize
        def area_sample(depth, fx, fy):
            dtype = depth.dtype
            B = depth.shape[0]
            H, W = down_size
            depth = downsample(depth, depth.shape[-2] // H)
            depth[depth > 200] = 50  # set sky as if depth 50 meters
            pixel_area3d = depth / torch.sqrt(fx * fy)

            # Set invalid as -1 (no div problem) -> then clip to 0.0
            pixel_area3d[depth == 0.0] = -1
            prob_density = (1 / pixel_area3d).clamp(min=0.0).square()
            prob_density = prob_density / prob_density.sum(
                dim=(-1, -2), keepdim=True
            ).clamp(min=1e-5)

            # Sample locations based on prob_density
            prob_density_flat = prob_density.view(B, -1)

            # Get the avgerage valid locations, of those we mask self.mask_ratio
            valid_locations = (prob_density_flat > 0).to(dtype).sum(dim=1)

            masks = []
            for i in range(B):
                num_samples = int(valid_locations[i] * mask_ratio)
                mask = torch.zeros_like(prob_density_flat[i])
                # Sample indices
                if num_samples > 0:
                    sampled_indices_flat = torch.multinomial(
                        prob_density_flat[i], num_samples, replacement=False
                    )
                    mask.scatter_(0, sampled_indices_flat, 1)
                masks.append(mask)
            return torch.stack(masks).bool().view(B, 1, H, W)

        def random_sample(validity_mask):
            prob_blocks = torch.rand(
                size=(validity_mask.shape[0], 1, *down_size), device=device
            )
            mask = torch.logical_and(prob_blocks < mask_ratio, validity_mask)
            return mask

        fx = results["K"][..., 0, 0].view(-1, 1, 1, 1) / self.mask_patch
        fy = results["K"][..., 1, 1].view(-1, 1, 1, 1) / self.mask_patch

        valid = ~results["ssi"] & ~results["si"] & results["valid_camera"]
        mask_blocks = torch.zeros_like(validity_mask)
        if valid.any():
            out = area_sample(results["depth"][valid], fx[valid], fy[valid])
            mask_blocks[valid] = out
        if (~valid).any():
            mask_blocks[~valid] = random_sample(validity_mask[~valid])

        return mask_blocks

    def guided_sampling(self, results, mask_ratio, down_size, validity_mask, device):
        # get the lowest (based on guidance) "mask_ratio" quantile of the patches that are in validity mask
        B = results["image"].shape[0]
        guidance = results["guidance"]
        mask_blocks = torch.zeros(size=(B, 1, *down_size), device=device)
        for b in range(B):
            low_bound = torch.quantile(
                guidance[b][validity_mask[b]], max(0.0, self.hard_bound - mask_ratio)
            )
            up_bound = torch.quantile(
                guidance[b][validity_mask[b]], min(1.0, self.hard_bound)
            )
            mask_blocks[b] = torch.logical_and(
                guidance[b] < up_bound, guidance[b] > low_bound
            )
        mask_blocks = torch.logical_and(mask_blocks, validity_mask)
        return mask_blocks

    def step(self):
        self.idx += 1
        # schedule hard from 1.0 to self.mask_ratio
        if self.curriculum:
            step = max(0, self.idx / self.warmup_steps / 2 - 0.5)
            self.hard_bound = 1 - (1 - self.mask_ratio) * tanh(step)
            self.up_bound = self.mask_ratio * tanh(step)
            self.low_bound = 0.1 * tanh(step)

    @property
    def is_warmup(self):
        return self.idx < self.warmup_steps


class Resize:
    def __init__(self, image_scale=None, image_shape=None, keep_original=False):
        assert (image_scale is None) ^ (image_shape is None)
        if isinstance(image_scale, (float, int)):
            image_scale = (image_scale, image_scale)
        if isinstance(image_shape, (float, int)):
            image_shape = (int(image_shape), int(image_shape))
        self.image_scale = image_scale
        self.image_shape = image_shape
        self.keep_original = keep_original

    def _resize_img(self, results):
        for key in results.get("image_fields", ["image"]):
            img = TF.resize(
                results[key],
                results["resized_shape"],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            results[key] = img

    def _resize_masks(self, results):
        for key in results.get("mask_fields", []):
            mask = TF.resize(
                results[key],
                results["resized_shape"],
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                antialias=True,
            )
            results[key] = mask

    def _resize_gt(self, results):
        for key in results.get("gt_fields", []):
            gt = TF.resize(
                results[key],
                results["resized_shape"],
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                antialias=True,
            )
            results[key] = gt

    def __call__(self, results):
        h, w = results["image"].shape[-2:]
        results["K_original"] = results["K"].clone()
        if self.image_scale:
            image_shape = (
                int(h * self.image_scale[0] + 0.5),
                int(w * self.image_scale[1] + 0.5),
            )
            image_scale = self.image_scale
        elif self.image_shape:
            image_scale = (self.image_shape[0] / h, self.image_shape[1] / w)
            image_shape = self.image_shape
        else:
            raise ValueError(
                f"In {self.__class__.__name__}: image_scale of image_shape must be set"
            )

        results["resized_shape"] = tuple(image_shape)
        results["resize_factor"] = tuple(image_scale)
        results["K"][..., 0, 2] = (results["K"][..., 0, 2] - 0.5) * image_scale[1] + 0.5
        results["K"][..., 1, 2] = (results["K"][..., 1, 2] - 0.5) * image_scale[0] + 0.5
        results["K"][..., 0, 0] = results["K"][..., 0, 0] * image_scale[1]
        results["K"][..., 1, 1] = results["K"][..., 1, 1] * image_scale[0]

        self._resize_img(results)
        if not self.keep_original:
            self._resize_masks(results)
            self._resize_gt(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class Rotate:
    def __init__(
        self, angle, center=None, img_fill_val=(123.68, 116.28, 103.53), prob=0.5
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, (
                "image_fill_val as tuple must "
                f"have 3 elements. got {len(img_fill_val)}."
            )
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError("image_fill_val must be float or tuple with 3 elements.")
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), f"all elements of img_fill_val should between range [0,255] got {img_fill_val}."
        assert 0 <= prob <= 1.0, f"The probability should be in range [0,1]bgot {prob}."
        self.center = center
        self.img_fill_val = img_fill_val
        self.prob = prob
        self.random = not isinstance(angle, (float, int))
        self.angle = angle

    def _rotate(self, results, angle, center=None, fill_val=0.0):
        for key in results.get("image_fields", ["image"]):
            img = results[key]
            img_rotated = TF.rotate(
                img,
                angle,
                center=center,
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=self.img_fill_val,
            )
            results[key] = img_rotated.to(img.dtype)
            results["image_shape"] = results[key].shape

        for key in results.get("mask_fields", []):
            results[key] = TF.rotate(
                results[key],
                angle,
                center=center,
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=fill_val,
            )

        for key in results.get("gt_fields", []):
            results[key] = TF.rotate(
                results[key],
                angle,
                center=center,
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=fill_val,
            )

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        if np.random.random() > self.prob:
            return results

        angle = (
            (self.angle[1] - self.angle[0]) * np.random.rand() + self.angle[0]
            if self.random
            else np.random.choice([-1, 1], size=1) * self.angle
        )
        self._rotate(results, angle, None, fill_val=0.0)
        results["rotation"] = angle
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(angle={self.angle}, "
        repr_str += f"center={self.center}, "
        repr_str += f"image_fill_val={self.img_fill_val}, "
        repr_str += f"prob={self.prob}, "
        return repr_str


class RandomColor:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.adjust_hue(results[key], factor)  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else self.level
        )
        self._adjust_color_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomSaturation:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_saturation_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get("image_fields", ["image"]):
            # NOTE defaultly the image should be BGR format
            results[key] = TF.adjust_saturation(results[key], factor)  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            2 ** ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else 2**self.level
        )
        self._adjust_saturation_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomSharpness:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_sharpeness_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get("image_fields", ["image"]):
            # NOTE defaultly the image should be BGR format
            results[key] = TF.adjust_sharpness(results[key], factor)  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            2 ** ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else 2**self.level
        )
        self._adjust_sharpeness_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomSolarize:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_solarize_img(self, results, factor=255.0):
        """Apply Color transformation to image."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.solarize(results[key], factor)  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else self.level
        )
        self._adjust_solarize_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomPosterize:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _posterize_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.posterize(results[key], int(factor))  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else self.level
        )
        self._posterize_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomEqualize:
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]."
        self.prob = prob

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.equalize(results[key])  # .to(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.random() > self.prob:
            return results
        self._imequalize(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob})"


class RandomBrightness:
    """Apply Brightness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Brightness transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_brightness_img(self, results, factor=1.0):
        """Adjust the brightness of image."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.adjust_brightness(results[key], factor)  # .to(img.dtype)

    def __call__(self, results, level=None):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            2 ** ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else 2**self.level
        )
        self._adjust_brightness_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomContrast:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def _adjust_contrast_img(self, results, factor=1.0):
        """Adjust the image contrast."""
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.adjust_contrast(results[key], factor)  # .to(img.dtype)

    def __call__(self, results, level=None):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.random() > self.prob:
            return results
        factor = (
            2 ** ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else 2**self.level
        )
        self._adjust_contrast_img(results, factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class RandomGamma:
    def __init__(self, level, prob=0.5):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob

    def __call__(self, results, level=None):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.random() > self.prob:
            return results
        factor = (self.level[1] - self.level[0]) * np.random.rand() + self.level[0]
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                results[key] = TF.adjust_gamma(results[key], 1 + factor)
        return results


class RandomInvert:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                results[key] = TF.invert(results[key])  # .to(img.dtype)
        return results


class RandomAutoContrast:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _autocontrast_img(self, results):
        for key in results.get("image_fields", ["image"]):
            img = results[key]
            results[key] = TF.autocontrast(img)  # .to(img.dtype)

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        self._autocontrast_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class Dilation:
    def __init__(self, origin, kernel, border_value=-1.0, iterations=1) -> None:
        self.structured_element = torch.ones(size=kernel)
        self.origin = origin
        self.border_value = border_value
        self.iterations = iterations

    def dilate(self, image):
        image_pad = F.pad(
            image,
            [
                self.origin[0],
                self.structured_element.shape[0] - self.origin[0] - 1,
                self.origin[1],
                self.structured_element.shape[1] - self.origin[1] - 1,
            ],
            mode="constant",
            value=self.border_value,
        )
        if image_pad.ndim < 4:
            image_pad = image_pad.unsqueeze(0)
        # Unfold the image to be able to perform operation on neighborhoods
        image_unfold = F.unfold(image_pad, kernel_size=self.structured_element.shape)
        # Flatten the structural element since its two dimensions have been flatten when unfolding
        # structured_element_flatten = torch.flatten(self.structured_element).unsqueeze(0).unsqueeze(-1)
        # Perform the greyscale operation; sum would be replaced by rest if you want erosion
        # sums = image_unfold + structured_element_flatten
        # Take maximum over the neighborhood
        # since we use depth, we need to take the cloest point (perspectivity)
        # thus the min. But min is for "unknown" (0), so put it to a large number
        # than take min

        mask = image_unfold < 1e-3  # if == 0, some pixels are not involved, why?

        # Replace the zero elements with a large value, so they don't affect the minimum operation
        image_unfold = image_unfold.masked_fill(mask, 1000.0)

        # Calculate the minimum along the neighborhood axis
        dilate_image = torch.min(image_unfold, dim=1).values

        # Fill the masked values with 0 to propagate zero if all pixels are zero
        dilate_image[mask.all(dim=1)] = 0
        return torch.reshape(dilate_image, image.shape)

    def __call__(self, results):
        for key in results.get("gt_fields", []):
            gt = results[key]
            for _ in range(self.iterations):
                gt[gt < 1e-4] = self.dilate(gt)[gt < 1e-4]
            results[key] = gt

        return results


class RandomShear(object):
    def __init__(
        self,
        level,
        prob=0.5,
        direction="horizontal",
    ):
        self.random = not isinstance(level, (float, int))
        self.level = level
        self.prob = prob
        self.direction = direction

    def _shear_img(self, results, magnitude):
        for key in results.get("image_fields", ["image"]):
            img_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=[0.0, 0.0],
                scale=1.0,
                shear=magnitude,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0,
            )
            results[key] = img_sheared

    def _shear_masks(self, results, magnitude):
        for key in results.get("mask_fields", []):
            mask_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=[0.0, 0.0],
                scale=1.0,
                shear=magnitude,
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=0.0,
            )
            results[key] = mask_sheared

    def _shear_gt(
        self,
        results,
        magnitude,
    ):
        for key in results.get("gt_fields", []):
            mask_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=[0.0, 0.0],
                scale=1.0,
                shear=magnitude,
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=0.0,
            )
            results[key] = mask_sheared

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        magnitude = (
            ((self.level[1] - self.level[0]) * np.random.rand() + self.level[0])
            if self.random
            else np.random.choice([-1, 1], size=1) * self.level
        )
        if self.direction == "horizontal":
            magnitude = [magnitude, 0.0]
        else:
            magnitude = [0.0, magnitude]
        self._shear_img(results, magnitude)
        self._shear_masks(results, magnitude)
        self._shear_gt(results, magnitude)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(level={self.level}, "
        repr_str += f"img_fill_val={self.img_fill_val}, "
        repr_str += f"seg_ignore_label={self.seg_ignore_label}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        repr_str += f"max_shear_magnitude={self.max_shear_magnitude}, "
        repr_str += f"random_negative_prob={self.random_negative_prob}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


class RandomTranslate(object):
    def __init__(
        self,
        range,
        prob=0.5,
        direction="horizontal",
    ):
        self.range = range
        self.prob = prob
        self.direction = direction

    def _translate_img(self, results, magnitude):
        for key in results.get("image_fields", ["image"]):
            img_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=magnitude,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=(123.68, 116.28, 103.53),
            )
            results[key] = img_sheared

    def _translate_mask(self, results, magnitude):
        for key in results.get("mask_fields", []):
            mask_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=magnitude,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=0.0,
            )
            results[key] = mask_sheared

    def _translate_gt(
        self,
        results,
        magnitude,
    ):
        for key in results.get("gt_fields", []):
            mask_sheared = TF.affine(
                results[key],
                angle=0.0,
                translate=magnitude,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST_EXACT,
                fill=0.0,
            )
            results[key] = mask_sheared

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        magnitude = (self.range[1] - self.range[0]) * np.random.rand() + self.range[0]
        if self.direction == "horizontal":
            magnitude = [magnitude * results["image"].shape[1], 0]
        else:
            magnitude = [0, magnitude * results["image"].shape[0]]
        self._translate_img(results, magnitude)
        self._translate_mask(results, magnitude)
        self._translate_gt(results, magnitude)
        results["K"][..., 0, 2] = results["K"][..., 0, 2] + magnitude[0]
        results["K"][..., 1, 2] = results["K"][..., 1, 2] + magnitude[1]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(range={self.range}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"direction={self.direction}, "
        return repr_str


class RandomCut(object):
    def __init__(self, prob=0.5, direction="all"):
        self.direction = direction
        self.prob = prob

    def _cut_img(self, results, coord, dim):
        for key in results.get("image_fields", ["image"]):
            img_sheared = torch.roll(
                results[key], int(coord * results[key].shape[dim]), dims=dim
            )
            results[key] = img_sheared

    def _cut_mask(self, results, coord, dim):
        for key in results.get("mask_fields", []):
            mask_sheared = torch.roll(
                results[key], int(coord * results[key].shape[dim]), dims=dim
            )
            results[key] = mask_sheared

    def _cut_gt(self, results, coord, dim):
        for key in results.get("gt_fields", []):
            gt_sheared = torch.roll(
                results[key], int(coord * results[key].shape[dim]), dims=dim
            )
            results[key] = gt_sheared

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        coord = 0.8 * random.random() + 0.1
        if self.direction == "horizontal":
            dim = -1
        elif self.direction == "vertical":
            dim = -2
        else:
            dim = -1 if random.random() < 0.5 else -2

        self._cut_img(results, coord, dim)
        self._cut_mask(results, coord, dim)
        self._cut_gt(results, coord, dim)
        return results


class DownsamplerGT(object):
    def __init__(self, downsample_factor: int, min_depth: float = 0.01):
        assert downsample_factor == round(
            downsample_factor, 0
        ), f"Downsample factor needs to be an integer, got {downsample_factor}"
        self.downsample_factor = downsample_factor
        self.min_depth = min_depth

    def _downsample_gt(self, results):
        for key in deepcopy(results.get("gt_fields", [])):
            gt = results[key]
            N, H, W = gt.shape
            gt = gt.view(
                N,
                H // self.downsample_factor,
                self.downsample_factor,
                W // self.downsample_factor,
                self.downsample_factor,
                1,
            )
            gt = gt.permute(0, 1, 3, 5, 2, 4)
            gt = gt.view(-1, self.downsample_factor * self.downsample_factor)
            gt_tmp = torch.where(gt == 0.0, 1e5 * torch.ones_like(gt), gt)
            gt = torch.min(gt_tmp, dim=-1).values
            gt = gt.view(N, H // self.downsample_factor, W // self.downsample_factor)
            gt = torch.where(gt > 1000, torch.zeros_like(gt), gt)
            results[f"{key}_downsample"] = gt
            results["gt_fields"].append(f"{key}_downsample")
        results["downsampled"] = True
        return results

    def __call__(self, results):
        results = self._downsample_gt(results)
        return results


class RandomColorJitter:
    def __init__(self, level, prob=0.9):
        self.level = level
        self.prob = prob
        self.list_transform = [
            self._adjust_brightness_img,
            # self._adjust_sharpness_img,
            self._adjust_contrast_img,
            self._adjust_saturation_img,
            self._adjust_color_img,
        ]

    def _adjust_contrast_img(self, results, factor=1.0):
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                img = results[key]
                results[key] = TF.adjust_contrast(img, factor)

    def _adjust_sharpness_img(self, results, factor=1.0):
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                img = results[key]
                results[key] = TF.adjust_sharpness(img, factor)

    def _adjust_brightness_img(self, results, factor=1.0):
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                img = results[key]
                results[key] = TF.adjust_brightness(img, factor)

    def _adjust_saturation_img(self, results, factor=1.0):
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                img = results[key]
                results[key] = TF.adjust_saturation(img, factor / 2.0)

    def _adjust_color_img(self, results, factor=1.0):
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                img = results[key]
                results[key] = TF.adjust_hue(img, (factor - 1.0) / 4.0)

    def __call__(self, results):
        random.shuffle(self.list_transform)
        for op in self.list_transform:
            if np.random.random() < self.prob:
                factor = 1.0 + (
                    (self.level[1] - self.level[0]) * np.random.random() + self.level[0]
                )
                op(results, factor)
        return results


class RandomGrayscale:
    def __init__(self, prob=0.1, num_output_channels=3):
        super().__init__()
        self.prob = prob
        self.num_output_channels = num_output_channels

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results

        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                results[key] = TF.rgb_to_grayscale(
                    results[key], num_output_channels=self.num_output_channels
                )
        return results


class ContextCrop(Resize):
    def __init__(
        self,
        image_shape,
        keep_original=False,
        test_min_ctx=1.0,
        train_ctx_range=[0.5, 1.5],
        shape_constraints={},
    ):
        super().__init__(image_shape=image_shape, keep_original=keep_original)
        self.test_min_ctx = test_min_ctx
        self.train_ctx_range = train_ctx_range

        self.shape_mult = shape_constraints["shape_mult"]
        self.sample = shape_constraints["sample"]
        self.ratio_bounds = shape_constraints["ratio_bounds"]
        pixels_min = shape_constraints["pixels_min"] / (
            self.shape_mult * self.shape_mult
        )
        pixels_max = shape_constraints["pixels_max"] / (
            self.shape_mult * self.shape_mult
        )
        self.pixels_bounds = (pixels_min, pixels_max)
        self.keepGT = int(os.environ.get("keepGT", 0))
        self.ctx = None

    def _transform_img(self, results, shapes):
        for key in results.get("image_fields", ["image"]):
            img = self.crop(results[key], **shapes)
            img = TF.resize(
                img,
                results["resized_shape"],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            results[key] = img

    def _transform_masks(self, results, shapes):
        for key in results.get("mask_fields", []):
            mask = self.crop(results[key].float(), **shapes).byte()
            if "flow" in key:  # take pad/crop into flow resize
                mask = TF.resize(
                    mask,
                    results["resized_shape"],
                    interpolation=TF.InterpolationMode.NEAREST_EXACT,
                    antialias=False,
                )
            else:
                mask = masked_nearest_interpolation(
                    mask, mask > 0, results["resized_shape"]
                )
            results[key] = mask

    def _transform_gt(self, results, shapes):
        for key in results.get("gt_fields", []):
            gt = self.crop(results[key], **shapes)
            if not self.keepGT:
                if "flow" in key:  # take pad/crop into flow resize
                    gt = self._rescale_flow(gt, results)
                    gt = TF.resize(
                        gt,
                        results["resized_shape"],
                        interpolation=TF.InterpolationMode.NEAREST_EXACT,
                        antialias=False,
                    )
                else:
                    gt = masked_nearest_interpolation(
                        gt, gt > 0, results["resized_shape"]
                    )

            results[key] = gt

    def _rescale_flow(self, gt, results):
        h_new, w_new = gt.shape[-2:]
        h_old, w_old = results["image_ori_shape"]
        gt[:, 0] = gt[:, 0] * (w_old - 1) / (w_new - 1)
        gt[:, 1] = gt[:, 1] * (h_old - 1) / (h_new - 1)
        return gt

    @staticmethod
    def crop(img, height, width, top, left) -> torch.Tensor:
        h, w = img.shape[-2:]
        right = left + width
        bottom = top + height
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(w, left), 0),
            max(bottom - max(h, top), 0),
        ]
        image_cropped = img[..., max(top, 0) : bottom, max(left, 0) : right]
        return TF.pad(image_cropped, padding_ltrb)

    def test_closest_shape(self, image_shape):
        h, w = image_shape
        input_ratio = w / h
        if self.sample:
            input_pixels = int(ceil(h / self.shape_mult * w / self.shape_mult))
            pixels = max(
                min(input_pixels, self.pixels_bounds[1]), self.pixels_bounds[0]
            )
            ratio = min(max(input_ratio, self.ratio_bounds[0]), self.ratio_bounds[1])
            h = round((pixels / ratio) ** 0.5)
            w = h * ratio
            self.image_shape[0] = int(h) * self.shape_mult
            self.image_shape[1] = int(w) * self.shape_mult

    def _get_crop_shapes(self, image_shape, ctx=None):
        h, w = image_shape
        input_ratio = w / h
        if self.keep_original:
            self.test_closest_shape(image_shape)
            ctx = 1.0
        elif ctx is None:
            ctx = float(
                torch.empty(1)
                .uniform_(self.train_ctx_range[0], self.train_ctx_range[1])
                .item()
            )
        output_ratio = self.image_shape[1] / self.image_shape[0]

        if output_ratio <= input_ratio:  # out like 4:3 in like kitti
            if (
                ctx >= 1
            ):  # fully in -> use just max_length with sqrt(ctx), here max is width
                new_w = w * ctx**0.5
            # sporge un po in una sola dim
            # we know that in_width will stick out before in_height, partial overshoot (sporge)
            # new_h > old_h via area -> new_h ** 2 * ratio_new = old_h ** 2 * ratio_old * ctx
            elif output_ratio / input_ratio * ctx > 1:
                new_w = w * ctx
            else:  # fully contained -> use area
                new_w = w * (ctx * output_ratio / input_ratio) ** 0.5
            new_h = new_w / output_ratio
        else:
            if ctx >= 1:
                new_h = h * ctx**0.5
            elif input_ratio / output_ratio * ctx > 1:
                new_h = h * ctx
            else:
                new_h = h * (ctx * input_ratio / output_ratio) ** 0.5
            new_w = new_h * output_ratio
        return (int(ceil(new_h - 0.5)), int(ceil(new_w - 0.5))), ctx

    def __call__(self, results):
        h, w = results["image"].shape[-2:]
        results["image_ori_shape"] = (h, w)
        results["camera_fields"].add("camera_original")
        results["camera_original"] = results["camera"].clone()

        results.get("mask_fields", set()).add("validity_mask")
        if "validity_mask" not in results:
            results["validity_mask"] = torch.ones(
                (results["image"].shape[0], 1, h, w),
                dtype=torch.uint8,
                device=results["image"].device,
            )

        n_iter = 1 if self.keep_original or not self.sample else 100

        min_valid_area = 0.5
        max_hfov, max_vfov = results["camera"].max_fov[0]  # it is a 1-dim list
        ctx = None
        for ii in range(n_iter):

            (height, width), ctx = self._get_crop_shapes((h, w), ctx=self.ctx or ctx)
            margin_h = h - height
            margin_w = w - width

            # keep it centered in y direction
            top = margin_h // 2
            left = margin_w // 2
            if not self.keep_original:
                left = left + np.random.randint(
                    -self.shape_mult // 2, self.shape_mult // 2 + 1
                )
                top = top + np.random.randint(
                    -self.shape_mult // 2, self.shape_mult // 2 + 1
                )

            right = left + width
            bottom = top + height
            x_zoom = self.image_shape[0] / height
            paddings = [
                max(-left + min(0, right), 0),
                max(bottom - max(h, top), 0),
                max(right - max(w, left), 0),
                max(-top + min(0, bottom), 0),
            ]

            valid_area = (
                h
                * w
                / (h + paddings[1] + paddings[3])
                / (w + paddings[0] + paddings[2])
            )
            new_hfov, new_vfov = results["camera_original"].get_new_fov(
                new_shape=(height, width), original_shape=(h, w)
            )[0]
            # if valid_area >= min_valid_area or getattr(self, "ctx", None) is not None:
            # break
            if (
                valid_area >= min_valid_area
                and new_hfov < max_hfov
                and new_vfov < max_vfov
            ):
                break
            ctx = (
                ctx * 0.96
            )  # if not enough valid area, try again with less ctx (more zoom)

        # save ctx for next iteration of sequences?
        self.ctx = ctx

        results["resized_shape"] = self.image_shape
        results["paddings"] = paddings  # left ,top ,right, bottom
        results["image_rescale"] = x_zoom
        results["scale_factor"] = results.get("scale_factor", 1.0) * x_zoom
        results["camera"] = results["camera"].crop(
            left, top, right=w - right, bottom=h - bottom
        )
        results["camera"] = results["camera"].resize(x_zoom)

        # print("XAM", results["camera"].params.squeeze(), results["camera"][0].params.squeeze(), results["camera_original"].params.squeeze(), results["camera_original"][0].params.squeeze())

        shapes = dict(height=height, width=width, top=top, left=left)
        self._transform_img(results, shapes)
        if not self.keep_original:
            self._transform_gt(results, shapes)
            self._transform_masks(results, shapes)
        else:
            # only validity_mask (rgb's masks follows rgb transform) #FIXME
            mask = results["validity_mask"].float()
            mask = self.crop(mask, **shapes).byte()
            mask = TF.resize(
                mask,
                results["resized_shape"],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            results["validity_mask"] = mask

        # keep original images before photo-augment
        results["image_original"] = results["image"].clone()
        results["image_fields"].add(
            *[
                field.replace("image", "image_original")
                for field in results["image_fields"]
            ]
        )

        # repeat for batch resized shape and paddings
        results["paddings"] = [results["paddings"]] * results["image"].shape[0]
        results["resized_shape"] = [results["resized_shape"]] * results["image"].shape[
            0
        ]
        return results


class RandomFiller:
    def __init__(self, test_mode, *args, **kwargs):
        super().__init__()
        self.test_mode = test_mode

    def _transform(self, results):
        def fill_noise(size, device):
            return torch.normal(0, 2.0, size=size, device=device)

        def fill_black(size, device):
            return -4 * torch.ones(size, device=device, dtype=torch.float32)

        def fill_white(size, device):
            return 4 * torch.ones(size, device=device, dtype=torch.float32)

        def fill_zero(size, device):
            return torch.zeros(size, device=device, dtype=torch.float32)

        B, C = results["image"].shape[:2]
        mismatch = B // results["validity_mask"].shape[0]
        if mismatch:
            results["validity_mask"] = results["validity_mask"].repeat(
                mismatch, 1, 1, 1
            )
        validity_mask = results["validity_mask"].repeat(1, C, 1, 1).bool()
        filler_fn = np.random.choice([fill_noise, fill_black, fill_white, fill_zero])
        if self.test_mode:
            filler_fn = fill_zero
        for key in results.get("image_fields", ["image"]):
            results[key][~validity_mask] = filler_fn(
                size=results[key][~validity_mask].shape, device=results[key].device
            )

    def __call__(self, results):
        # generate mask for filler
        if "validity_mask" not in results:
            paddings = results.get("padding_size", [0] * 4)
            height, width = results["image"].shape[-2:]
            results.get("mask_fields", set()).add("validity_mask")
            results["validity_mask"] = torch.zeros_like(results["image"][:, :1])
            results["validity_mask"][
                ...,
                paddings[1] : height - paddings[3],
                paddings[0] : width - paddings[2],
            ] = 1.0
        self._transform(results)
        return results


class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0), prob=0.9):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob
        self.padding = kernel_size // 2

    def apply(self, x, kernel):
        # Pad the input tensor
        x = F.pad(
            x, (self.padding, self.padding, self.padding, self.padding), mode="reflect"
        )
        # Apply the convolution with the Gaussian kernel
        return F.conv2d(x, kernel, stride=1, padding=0, groups=x.size(1))

    def _create_kernel(self, sigma):
        # Create a 1D Gaussian kernel
        kernel_1d = torch.exp(
            -torch.arange(-self.padding, self.padding + 1) ** 2 / (2 * sigma**2)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Expand the kernel to 2D and match size of the input
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.view(1, 1, self.kernel_size, self.kernel_size).expand(
            3, 1, -1, -1
        )
        return kernel_2d

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results
        sigma = (self.sigma[1] - self.sigma[0]) * np.random.rand() + self.sigma[0]
        kernel = self._create_kernel(sigma)
        for key in results.get("image_fields", ["image"]):
            if "original" not in key:
                results[key] = self.apply(results[key], kernel)
        return results


class MotionBlur:
    def __init__(self, kernel_size=(9, 9), angles=(-180, 180), prob=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.angles = angles
        self.prob = prob
        self.padding = kernel_size // 2

    def _create_kernel(self, angle):
        # Generate a 2D grid of coordinates
        grid = torch.meshgrid(
            torch.arange(self.kernel_size), torch.arange(self.kernel_size)
        )
        grid = torch.stack(grid).float()  # Shape: (2, kernel_size, kernel_size)

        # Calculate relative coordinates from the center
        center = (self.kernel_size - 1) / 2.0
        x_offset = grid[1] - center
        y_offset = grid[0] - center

        # Compute motion blur kernel
        cos_theta = torch.cos(angle * torch.pi / 180.0)
        sin_theta = torch.sin(angle * torch.pi / 180.0)
        kernel = (1.0 / self.kernel_size) * (
            1.0 - torch.abs(x_offset * cos_theta + y_offset * sin_theta)
        )

        # Expand kernel dimensions to match input image channels
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        return kernel

    def apply(self, image, kernel):
        x = F.pad(
            x, (self.padding, self.padding, self.padding, self.padding), mode="reflect"
        )
        # Apply convolution with the motion blur kernel
        blurred_image = F.conv2d(image, kernel, stride=1, padding=0, groups=x.size(1))
        return blurred_image

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results

        angle = np.random.uniform(self.angles[0], self.angles[1])
        kernel = self._create_kernel(angle)
        for key in results.get("image_fields", ["image"]):
            if "original" in key:
                continue
            results[key] = self.apply(results[key], kernel)

        return results


class JPEGCompression:
    def __init__(self, level=(10, 70), prob=0.1):
        super().__init__()
        self.level = level
        self.prob = prob

    def __call__(self, results):
        if np.random.random() > self.prob:
            return results

        level = np.random.uniform(self.level[0], self.level[1])
        for key in results.get("image_fields", ["image"]):
            if "original" in key:
                continue
            results[key] = TF.jpeg(results[key], level)

        return results


class RandomSequenceFlip:
    def __init__(self, prob=0.25, **kwargs):
        self.flip_ratio = prob

    def __call__(self, results):
        if np.random.random() < self.flip_ratio:
            for i in range(len(results["sequence_fields"]) // 2):
                (
                    results[(i, 0)],
                    results[(len(results["sequence_fields"]) - 1 - i, 0)],
                ) = (
                    results[(len(results["sequence_fields"]) - 1 - i, 0)],
                    results[(i, 0)],
                )
        return results


class Compose:
    def __init__(self, transforms):
        self.transforms = deepcopy(transforms)

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        for t in self.transforms:
            setattr(t, name, value)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class DummyCrop(Resize):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # dummy image shape, not really used
        super().__init__(image_shape=(512, 512))

    def __call__(self, results):
        h, w = results["image"].shape[-2:]
        results["image_ori_shape"] = (h, w)
        results["camera_fields"].add("camera_original")
        results["camera_original"] = results["camera"].clone()
        results.get("mask_fields", set()).add("validity_mask")
        if "validity_mask" not in results:
            results["validity_mask"] = torch.ones(
                (results["image"].shape[0], 1, h, w),
                dtype=torch.uint8,
                device=results["image"].device,
            )

        self.ctx = 1.0

        results["resized_shape"] = self.image_shape
        results["paddings"] = [0, 0, 0, 0]
        results["image_rescale"] = 1.0
        results["scale_factor"] = results.get("scale_factor", 1.0) * 1.0
        results["camera"] = results["camera"].crop(0, 0, right=w, bottom=h)
        results["camera"] = results["camera"].resize(1)

        # keep original images before photo-augment
        results["image_original"] = results["image"].clone()
        results["image_fields"].add(
            *[
                field.replace("image", "image_original")
                for field in results["image_fields"]
            ]
        )

        # repeat for batch resized shape and paddings
        results["paddings"] = [results["paddings"]] * results["image"].shape[0]
        results["resized_shape"] = [results["resized_shape"]] * results["image"].shape[
            0
        ]
        return results
