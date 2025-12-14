import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from velodepth.utils.geometric import dilate, downsample, erode

from .utils import FNS, masked_mean, masked_quantile


class LocalNormal(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        min_samples: int = 4,
        quantile: float = 0.2,
        eps: float = 1e-5,
    ):
        super(LocalNormal, self).__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.min_samples = min_samples
        self.eps = eps
        self.patch_weight = torch.ones(1, 1, 3, 3, device="cuda")
        self.quantile = quantile

    def bilateral_filter(self, rgb, surf, mask, patch_size=(9, 9)):
        B, _, H, W = rgb.shape
        sigma_surf = 0.4
        sigma_color = 0.3
        sigma_loc = 0.3 * max(H, W)

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid = torch.stack([grid_x, grid_y], dim=0).to(rgb.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        paddings = [patch_size[0] // 2, patch_size[1] // 2]
        rgbd = torch.cat([rgb, grid.float(), surf], dim=1)

        # format to B,H*W,C,H_p*W_p format
        rgbd_neigh = F.pad(rgbd, 2 * paddings, mode="constant")
        rgbd_neigh = F.unfold(rgbd_neigh, kernel_size=patch_size)
        rgbd_neigh = rgbd_neigh.permute(0, 2, 1).reshape(
            B, H * W, 8, -1
        )  # B N 8 H_p*W_p
        mask_neigh = F.pad(mask.float(), 2 * paddings, mode="constant")
        mask_neigh = F.unfold(mask_neigh, kernel_size=patch_size)
        mask_neigh = mask_neigh.permute(0, 2, 1).reshape(B, H * W, -1)
        rgbd = rgbd.permute(0, 2, 3, 1).reshape(B, H * W, 8, 1)  # B H*W 8 1
        rgb_neigh = rgbd_neigh[:, :, :3, :]
        grid_neigh = rgbd_neigh[:, :, 3:5, :]
        surf_neigh = rgbd_neigh[:, :, 5:, :]
        rgb = rgbd[:, :, :3, :]
        grid = rgbd[:, :, 3:5, :]
        surf = rgbd[:, :, 5:, :]

        # calc distance
        rgb_dist = torch.norm(rgb - rgb_neigh, dim=-2, p=2) ** 2
        grid_dist = torch.norm(grid - grid_neigh, dim=-2, p=2) ** 2
        surf_dist = torch.norm(surf - surf_neigh, dim=-2, p=2) ** 2
        rgb_sim = torch.exp(-rgb_dist / 2 / sigma_color**2)
        grid_sim = torch.exp(-grid_dist / 2 / sigma_loc**2)
        surf_sim = torch.exp(-surf_dist / 2 / sigma_surf**2)

        weight = mask_neigh * rgb_sim * grid_sim * surf_sim  # B H*W H_p*W_p
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        z = (surf_neigh * weight.unsqueeze(-2)).sum(dim=-1)
        return z.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def get_surface_normal(self, xyz: torch.Tensor, mask: torch.Tensor):
        P0 = xyz
        mask = mask.float()
        normals, masks_valid_triangle = [], []
        combinations = list(itertools.combinations_with_replacement([-2, -1, 1, 2], 2))
        combinations += [c[::-1] for c in combinations]
        # combinations = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        for shift_0, shift_1 in set(combinations):
            P1 = torch.roll(xyz, shifts=(0, shift_0), dims=(-1, -2))
            P2 = torch.roll(xyz, shifts=(shift_1, 0), dims=(-1, -2))
            if (shift_0 > 0) ^ (shift_1 > 0):
                P1, P2 = P2, P1
            vec1, vec2 = P1 - P0, P2 - P0
            normal = torch.cross(vec1, vec2, dim=1)
            vec1_norm = torch.norm(vec1, dim=1, keepdim=True).clip(min=1e-8)
            vec2_norm = torch.norm(vec2, dim=1, keepdim=True).clip(min=1e-8)
            normal_norm = torch.norm(normal, dim=1, keepdim=True).clip(min=1e-8)
            normals.append(normal / normal_norm)
            is_valid = (
                torch.roll(mask, shifts=(0, shift_0), dims=(-1, -2))
                + torch.roll(mask, shifts=(shift_1, 0), dims=(-1, -2))
                + mask
                == 3
            )
            is_valid = (
                (normal_norm > 1e-6)
                & (vec1_norm > 1e-6)
                & (vec2_norm > 1e-6)
                & is_valid
            )
            masks_valid_triangle.append(is_valid)

        normals = torch.stack(normals, dim=-1)
        mask_valid_triangle = torch.stack(masks_valid_triangle, dim=-1).float()
        mask_valid = mask_valid_triangle.sum(dim=-1)
        normals = (normals * mask_valid_triangle).sum(dim=-1) / mask_valid.clamp(
            min=1.0
        )
        normals_norm = torch.norm(normals, dim=1, keepdim=True).clip(min=1e-8)
        normals = normals / normals_norm
        mask_valid = (
            (mask_valid > 0.001)
            & (~normals.sum(dim=1, keepdim=True).isnan())
            & (normals_norm > 1e-6)
        )
        return normals, mask_valid  # B 3 H W, B 1 H W

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask, valid):
        if not valid.any():
            return 0.0 * input.mean(dim=(1, 2, 3))

        input = input.float()
        target = target.float()

        mask = erode(mask, kernel_size=3)
        target_normal, mask_target = self.get_surface_normal(target[valid], mask[valid])
        input_normal, mask_input = self.get_surface_normal(
            input[valid], torch.ones_like(mask[valid])
        )

        gt_similarity = F.cosine_similarity(input_normal, target_normal, dim=1)  # B H W
        mask_target = (
            mask_target.squeeze(1) & (gt_similarity < 0.999) & (gt_similarity > -0.999)
        )
        error = F.relu((1 - gt_similarity) / 2 - 0.01)

        error_full = torch.ones_like(mask.squeeze(1).float())
        error_full[valid] = error
        mask_full = torch.ones_like(mask.squeeze(1))
        mask_full[valid] = mask_target

        error_qtl = error_full.detach()
        mask_full = mask_full & (
            error_qtl
            < masked_quantile(
                error_qtl, mask_full, dims=[1, 2], q=1 - self.quantile
            ).view(-1, 1, 1)
        )

        loss = masked_mean(error_full, mask=mask_full, dim=(-2, -1)).squeeze(
            dim=(-2, -1)
        )  # B
        loss = self.output_fn(loss)
        return loss

    def von_mises(self, input, target, mask, kappa):
        score = torch.cosine_similarity(input, target, dim=1).unsqueeze(1)
        mask_cosine = torch.logical_and(
            mask, torch.logical_and(score.detach() < 0.999, score.detach() > -0.999)
        )
        nll = masked_mean(
            kappa * (1 - score), mask=mask_cosine, dim=(-1, -2, -3)
        ).squeeze()
        return nll

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            quantile=config.get("quantile", 0.2),
        )
        return obj
