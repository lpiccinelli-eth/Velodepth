"""
Author: Luigi Piccinelli
Licensed under the CC BY-NC-SA 4.0 license (http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import importlib
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from velodepth.models.decoder_velo import flow2grid_sample
from velodepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                       IMAGENET_DATASET_STD)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long
    )
    select = select.to(device=img.device, non_blocking=True)
    select = select[:, i]
    if select.ndim > 3:
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)


def denormalize(x, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD):
    mean = torch.tensor(mean, device=x.device)
    std = torch.tensor(std, device=x.device)
    return x * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)


def rgb2luma(rgb):
    return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]


class SmallEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        mod = importlib.import_module("velodepth.models.encoder")
        residual_encoder_factory = getattr(
            mod, config["model"]["residual_encoder"]["name"]
        )
        residual_encoder_config = {
            **config["training"],
            **config["model"]["residual_encoder"],
            **config["data"],
        }
        residual_encoder = residual_encoder_factory(residual_encoder_config)

        config["model"]["residual_encoder"]["embed_dim"] = getattr(
            residual_encoder, "embed_dim"
        )
        rgb_dims = residual_encoder.embed_dims
        config["model"]["residual_encoder"]["embed_dims"] = rgb_dims
        num_levels = config["model"]["residual_encoder"]["num_levels"]
        self.num_levels = num_levels
        self.encoder_rgb = deepcopy(residual_encoder.clean(4))
        self.encoder_depth = deepcopy(residual_encoder.clean(num_levels))

        # separate_flow
        flow_encoder_factory = getattr(mod, config["model"]["flow_encoder"]["name"])
        flow_encoder_config = {
            **config["training"],
            **config["model"]["flow_encoder"],
            **config["data"],
        }
        self.flow_levels = config["model"]["flow_encoder"]["num_levels"]
        self.encoder_flow = flow_encoder_factory(flow_encoder_config).clean(
            self.flow_levels
        )
        flow_dims = self.encoder_flow.embed_dims[: self.flow_levels]
        config["model"]["flow_encoder"]["embed_dims"] = flow_dims
        dim = flow_dims[self.flow_levels - 1]

        self.use_hsv = config["model"]["residual_encoder"].get("use_hsv", False)
        self.decode_flow_features = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64))
        self.decode_flow_refined = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        self.flow_proj = nn.ModuleList([])
        assert self.num_levels <= self.flow_levels
        for i in range(self.num_levels):
            self.flow_proj.append(
                nn.Sequential(
                    nn.LayerNorm(flow_dims[i]), nn.Linear(flow_dims[i], rgb_dims[i])
                )
            )

    def get_flow(self, flow):
        return flow[:, :2] / 100.0, flow[:, 2:]

    def forward(
        self, rgb_t, depth_t, flow_t, raft_t=None, info_t=None, residuals_t=None
    ):
        B, _, H, W = rgb_t.shape
        flow_hsv = torch.ones_like(rgb_t)  # Set saturation as image luma
        rgb_01 = denormalize(residuals_t.abs())
        luma_01 = rgb2luma(rgb_01)
        if self.use_hsv:
            flow_hsv[:, 0] = (
                torch.atan2(flow_t[:, 1], flow_t[:, 0]) / torch.pi / 2 + 0.5
            )  # [-pi, pi] -> [0,1]
            flow_hsv[:, 2] = torch.norm(flow_t, dim=1).clip(max=0.1) * 10.0
            flow_hsv[:, 1] = luma_01
            flow_rgb = (_hsv_to_rgb(flow_hsv) - 0.5) * 4.0 # make stronger, flow is quite small
        else:
            flow_hsv[:, 0] = flow_t[:, 0] * 10.0
            flow_hsv[:, 1] = flow_t[:, 1] * 10.0
            flow_hsv[:, 2] = (luma_01 - 0.5) * 4.0
            flow_rgb = flow_hsv

        rgb_feats = self.encoder_rgb.forward_part(rgb_t, range(0, self.num_levels))[0]
        flow_feats = self.encoder_flow.forward_part(
            flow_rgb, range(0, self.flow_levels)
        )[0]

        flow_features = self.decode_flow_features(flow_feats[-1]).permute(0, 3, 1, 2)
        flow_refined = self.decode_flow_refined(flow_features)
        flow_refined, flow_gate = self.get_flow(flow_refined)

        flow_gate = F.interpolate(flow_gate, size=(H, W), mode="bilinear")
        flow_refined = (
            F.interpolate(flow_refined, size=(H, W), mode="bilinear") + flow_t
        )

        depth_t = F.grid_sample(
            depth_t,
            flow2grid_sample(flow_refined.clone()),
            mode="bilinear",
            align_corners=False,
        )
        depth_t = (torch.log(depth_t.clamp(min=1e-2)) - 2) / 3.0
        depth_feats = self.encoder_depth.forward_part(
            depth_t.repeat(1, 3, 1, 1), range(0, self.num_levels)
        )[0]

        # gate rgb, depth with flow
        for i in range(len(rgb_feats)):
            flow_gate_ = F.interpolate(
                flow_gate, size=rgb_feats[i].shape[1:3], mode="bilinear"
            ).permute(0, 2, 3, 1)
            rgb_feats[i] = (
                rgb_feats[i]
                + depth_feats[i] * torch.sigmoid(flow_gate_)
                + self.flow_proj[i](flow_feats[i])
            )

        # run the fused for last features
        rgb_feats_2 = self.encoder_rgb.forward_part(
            rgb_feats[-1].permute(0, 3, 1, 2), range(self.num_levels, 4)
        )[0]
        out_features = [x.permute(0, 3, 1, 2) for x in [*rgb_feats, *rgb_feats_2]]

        return out_features, flow_refined, flow_features
