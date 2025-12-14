"""
Author: Luigi Piccinelli
Licensed under the CC BY-NC-SA 4.0 license (http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

from velodepth.layers import (MLP, AttentionBlock, AttentionLayer, FusionBlock,
                              PositionEmbeddingSine, ResUpsampleBil)
from velodepth.utils.coordinate import coords_grid
from velodepth.utils.geometric import flat_interpolate
from velodepth.utils.misc import get_params
from velodepth.utils.positional_embedding import generate_fourier_features
from velodepth.utils.sht import rsh_cart_3


def flow2grid_sample(flow, mask=True):
    B, C, H, W = flow.shape
    flow[:, 0] = flow[:, 0] * (W - 1)
    flow[:, 1] = flow[:, 1] * (H - 1)
    coords = coords_grid(B, H, W, device=flow.device)
    grid = flow + coords
    grid[:, 0] = (grid[:, 0] - 0.5) * 2 / (W - 1) - 1
    grid[:, 1] = (grid[:, 1] - 0.5) * 2 / (H - 1) - 1
    if mask:
        mask_grid_sample = (
            (grid[:, 0] >= -1)
            & (grid[:, 0] <= 1)
            & (grid[:, 1] >= -1)
            & (grid[:, 1] <= 1)
        )
        mask_grid_sample = mask_grid_sample.unsqueeze(1).repeat(1, 2, 1, 1).bool()
        grid[~mask_grid_sample] = coords[~mask_grid_sample]
    grid = grid.permute(0, 2, 3, 1)
    return grid


def orthonormal_init(num_tokens, dims):
    pe = torch.randn(num_tokens, dims)
    for i in range(num_tokens):
        for j in range(i):
            pe[i] -= torch.dot(pe[i], pe[j]) * pe[j]
        pe[i] = F.normalize(pe[i], p=2, dim=0)
    return pe


class ListAdapter(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        for input_dim in input_dims:
            self.input_adapters.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = [self.input_adapters[i](x) for i, x in enumerate(xs)]
        return outs


class AngularModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.pin_params = 3
        self.deg1_params = 3
        self.deg2_params = 5
        self.deg3_params = 7
        self.num_params = (
            self.pin_params + self.deg1_params + self.deg2_params + self.deg3_params
        )

        self.aggregate1 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, self.num_params, hidden_dim), requires_grad=True
        )

        self.project_pin = nn.Linear(
            hidden_dim, self.pin_params * hidden_dim, bias=False
        )
        self.project_deg1 = nn.Linear(
            hidden_dim, self.deg1_params * hidden_dim, bias=False
        )
        self.project_deg2 = nn.Linear(
            hidden_dim, self.deg2_params * hidden_dim, bias=False
        )
        self.project_deg3 = nn.Linear(
            hidden_dim, self.deg3_params * hidden_dim, bias=False
        )

        self.out_pinhole = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=1)
        self.out_deg1 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg2 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg3 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)

    def fill_intrinsics(self, x):
        hfov, cx, cy = x.unbind(dim=-1)
        hfov = torch.sigmoid(hfov - 1.1)  # 1.1 magic number s.t hfov = pi/2 for x=0
        ratio = self.shapes[0] / self.shapes[1]
        vfov = hfov * ratio
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        correction_tensor = torch.tensor(
            [2 * torch.pi, 2 * torch.pi, self.shapes[1], self.shapes[0]],
            device=x.device,
            dtype=x.dtype,
        )

        intrinsics = torch.stack([hfov, vfov, cx, cy], dim=1)
        intrinsics = correction_tensor.unsqueeze(0) * intrinsics
        return intrinsics

    def forward(self, cls_tokens) -> torch.Tensor:
        pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens = cls_tokens.chunk(4, dim=1)
        pin_tokens = rearrange(
            self.project_pin(pin_tokens), "b n (h c) -> b (n h) c", h=self.pin_params
        )
        deg1_tokens = rearrange(
            self.project_deg1(deg1_tokens), "b n (h c) -> b (n h) c", h=self.deg1_params
        )
        deg2_tokens = rearrange(
            self.project_deg2(deg2_tokens), "b n (h c) -> b (n h) c", h=self.deg2_params
        )
        deg3_tokens = rearrange(
            self.project_deg3(deg3_tokens), "b n (h c) -> b (n h) c", h=self.deg3_params
        )
        tokens = torch.cat([pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens], dim=1)

        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        tokens = self.aggregate1(tokens, pos_embed=latents_pos)
        tokens = self.aggregate2(tokens, pos_embed=latents_pos)

        tokens_pinhole, tokens_deg1, tokens_deg2, tokens_deg3 = torch.split(
            tokens,
            [self.pin_params, self.deg1_params, self.deg2_params, self.deg3_params],
            dim=1,
        )
        x = self.out_pinhole(tokens_pinhole).squeeze(-1)
        d1 = self.out_deg1(tokens_deg1)
        d2 = self.out_deg2(tokens_deg2)
        d3 = self.out_deg3(tokens_deg3)

        camera_intrinsics = self.fill_intrinsics(x)
        return camera_intrinsics, torch.cat([d1, d2, d3], dim=1)

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes


class RadialModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths: int | list[int] = 4,
        camera_dim: int = 256,
        dropout: float = 0.0,
        kernel_size: int = 7,
        layer_scale: float = 1.0,
        out_dim: int = 1,
        num_prompt_blocks: int = 1,
        residual_dims=[256, 128, 64, 32],
        num_fusion_block: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.camera_dim = camera_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.ups = nn.ModuleList([])
        self.depth_mlp = nn.ModuleList([])
        self.process_features = nn.ModuleList([])
        self.project_features = nn.ModuleList([])
        self.prompt_camera = nn.ModuleList([])
        self.fusion = nn.ModuleList([])
        mult = 2
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        for _ in range(4):
            self.prompt_camera.append(
                AttentionLayer(
                    num_blocks=num_prompt_blocks,
                    dim=hidden_dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    layer_scale=-1.0,
                    context_dim=hidden_dim,
                )
            )

        self.fusion.append(
            FusionBlock(
                residual_dims[0],
                hidden_dim,
                previous_dim=residual_dims[0],
                num_blocks=num_fusion_block,
                gate_dim=64,
                use_norm=True,
            )
        )
        p_dim = hidden_dim
        for i, depth in enumerate(depths):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=max(1, 2 * i),
                    stride=max(1, 2 * i),
                    padding=0,
                )
            )

            fusion_dim = hidden_dim
            self.fusion.append(
                FusionBlock(
                    residual_dims[i + 1],
                    fusion_dim,
                    previous_dim=p_dim,
                    num_blocks=num_fusion_block,
                    gate_dim=64,
                    use_norm=True,
                )
            )
            p_dim = fusion_dim

            self.ups.append(
                ResUpsampleBil(
                    current_dim,
                    output_dim=output_dim,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    kernel_size=kernel_size,
                    num_layers=depth,
                    use_norm=False,
                )
            )
            depth_mlp = (
                nn.Sequential(nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim))
                if i == len(depths) - 1
                else nn.Identity()
            )
            self.depth_mlp.append(depth_mlp)

        self.confidence_mlp = nn.Sequential(
            nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim)
        )

        self.to_depth_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_confidence_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.to_confidence_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_rays(self, rays):
        rays_embedding = flat_interpolate(
            rays, old=self.original_shapes, new=self.shapes, antialias=True
        )
        rays_embedding = rays_embedding / torch.norm(
            rays_embedding, dim=-1, keepdim=True
        ).clip(min=1e-4)
        x, y, z = rays_embedding[..., 0], rays_embedding[..., 1], rays_embedding[..., 2]
        polar = torch.acos(z)
        x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
        azimuth = torch.atan2(y, x_clipped)
        rays_embedding = torch.stack([polar, azimuth], dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.hidden_dim,
            max_freq=max(self.shapes) // 2,
            use_log=True,
            cat_orig=False,
        )
        return rays_embedding

    def condition(self, feat, rays_embeddings):
        conditioned_features = [
            prompter(rearrange(feature, "b h w c -> b (h w) c"), rays_embeddings)
            for prompter, feature in zip(self.prompt_camera, feat)
        ]
        return conditioned_features

    def process(self, features_list, rays_embeddings):
        conditioned_features = self.condition(features_list, rays_embeddings)
        init_latents = self.to_latents(conditioned_features[0])
        init_latents = rearrange(
            init_latents, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]
        ).contiguous()
        conditioned_features = [
            rearrange(
                x, "b (h w) c -> b c h w", h=self.shapes[0], w=self.shapes[1]
            ).contiguous()
            for x in conditioned_features
        ]
        latents = init_latents
        lateral_features = [conditioned_features[0]] + [
            self.process_features[i](conditioned_features[i + 1])
            for i in range(len(self.ups))
        ]

        out_features = []
        for i, up in enumerate(self.ups):
            latents = latents + lateral_features[i + 1]
            latents = up(latents)
            out_features.append(latents)

        return out_features, conditioned_features

    def depth_proj(self, out_features):
        h_out, w_out = out_features[-1].shape[-2:]
        # aggregate output and project to depth
        for i, (layer, features) in enumerate(zip(self.depth_mlp, out_features)):
            out_depth_features = layer(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out_depth_features = F.interpolate(
            out_depth_features, size=(h_out, w_out), mode="bilinear", align_corners=True
        )
        logdepth = self.to_depth_lr(out_depth_features)
        logdepth = F.interpolate(
            logdepth, size=self.original_shapes, mode="bilinear", align_corners=True
        )
        logdepth = self.to_depth_hr(logdepth)
        return logdepth

    def confidence_proj(self, out_features):
        highres_features = out_features[-1].permute(0, 2, 3, 1)
        confidence = self.confidence_mlp(highres_features).permute(0, 3, 1, 2)
        confidence = self.to_confidence_lr(confidence)
        confidence = F.interpolate(
            confidence, size=self.original_shapes, mode="bilinear", align_corners=True
        )
        confidence = self.to_confidence_hr(confidence)
        return confidence

    def decode(self, out_features):
        logdepth = self.depth_proj(out_features)
        confidence = self.confidence_proj(out_features)
        return logdepth, confidence

    def forward_keyframe(self, features, rays_hr):
        rays_embeddings = self.embed_rays(rays_hr)
        features_teacher, cond_features = self.process(features, rays_embeddings)
        logdepth_teacher, logconf_teacher = self.decode(features_teacher)
        return logdepth_teacher, logconf_teacher, cond_features

    def forward_fast(
        self, lateral_features_teacher, features_residual, flow_bwd, flow_features
    ):
        features_propagated = []
        for i, (init_, residual) in enumerate(
            zip(lateral_features_teacher, features_residual)
        ):
            flow_bwd = F.interpolate(
                flow_bwd, size=init_.shape[-2:], mode="bilinear", align_corners=True
            )
            warp_grid = flow2grid_sample(flow_bwd.clone())
            init_warped = F.grid_sample(
                init_, warp_grid, mode="bilinear", align_corners=True
            )
            corrected_features = self.fusion[i](
                init_warped,
                residual,
                original_features=init_,
                previous=residual if i == 0 else features_propagated[-1],
                flow_features=flow_features,
            )
            features_propagated.append(corrected_features)

        lateral_features = [features_propagated[0]] + [
            self.process_features[i](features_propagated[i + 1])
            for i in range(len(self.ups))
        ]

        # run decoder
        latents = lateral_features[0]
        features_student = []
        for i, up in enumerate(self.ups):
            latents = latents + lateral_features[i + 1]
            latents = up(latents)
            features_student.append(latents)
        logdepth_student = self.depth_proj(features_student)
        logconf_student = logdepth_student

        return logdepth_student, logconf_student, features_propagated

    def forward(
        self,
        features: list[torch.Tensor],
        features_residual: list[torch.Tensor],
        rays_hr: torch.Tensor,
        flow: torch.Tensor,
    ):
        logdepth_teacher, logconf_teacher, lateral_features_teacher = (
            self.forward_keyframe(features, rays_hr)
        )
        logdepth_student, logconf_student, lateral_features_student = self.forward_fast(
            lateral_features_teacher, features_residual, flow, rays_hr
        )
        return (
            logdepth_teacher,
            logconf_teacher,
            lateral_features_teacher,
            logdepth_student,
            logconf_student,
            lateral_features_student,
        )

    def train(self, mode):
        super().train(mode)
        for mod in [
            self.ups,
            self.depth_mlp,
            self.confidence_mlp,
            self.prompt_camera,
            self.project_features,
            self.process_features,
            self.to_depth_hr,
            self.to_depth_lr,
            self.to_confidence_hr,
            self.to_confidence_lr,
            self.to_latents,
        ]:
            mod.train(False)
            for p in mod.parameters():
                p.requires_grad = False
        self.fusion.train(mode)


class Decoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def run_camera(self, cls_tokens, original_shapes, rays_gt):
        H, W = original_shapes

        # camera layer
        intrinsics, sh_coeffs = self.angular_module(cls_tokens=cls_tokens)
        B, N = intrinsics.shape
        device = intrinsics.device
        dtype = intrinsics.dtype

        id_coords = coords_grid(B, H, W, device=sh_coeffs.device)

        # This is fov based
        longitude = (
            (id_coords[:, 0] - intrinsics[:, 2].view(-1, 1, 1))
            / W
            * intrinsics[:, 0].view(-1, 1, 1)
        )
        latitude = (
            (id_coords[:, 1] - intrinsics[:, 3].view(-1, 1, 1))
            / H
            * intrinsics[:, 1].view(-1, 1, 1)
        )
        x = torch.cos(latitude) * torch.sin(longitude)
        z = torch.cos(latitude) * torch.cos(longitude)
        y = -torch.sin(latitude)
        unit_sphere = torch.stack([x, y, z], dim=-1)
        unit_sphere = unit_sphere / torch.norm(unit_sphere, dim=-1, keepdim=True).clip(
            min=1e-5
        )

        harmonics = rsh_cart_3(unit_sphere)[..., 1:]  # remove constant-value harmonic
        rays_pred = torch.einsum("bhwc,bcd->bhwd", harmonics, sh_coeffs)
        rays_pred = rays_pred / torch.norm(rays_pred, dim=-1, keepdim=True).clip(
            min=1e-5
        )
        rays_pred = rays_pred.permute(0, 3, 1, 2)

        ### LEGACY CODE for training
        # if self.training:
        #     prob = 1 - tanh(self.steps / 100000)
        #     where_use_gt_rays = torch.rand(B, 1, 1, device=device, dtype=dtype) < prob
        #     where_use_gt_rays = where_use_gt_rays.int()
        #     rays = rays_gt * where_use_gt_rays + rays_pred * (1 - where_use_gt_rays)

        # should clean also nans
        if self.training:
            rays = rays_pred
        elif self.camera_gt:
            rays = rays_gt if rays_gt is not None else rays_pred
        else:
            rays = rays_pred
        rays = rearrange(rays, "b c h w -> b (h w) c")

        return intrinsics, rays

    def forward_keyframe(self, inputs, image_metas):
        B, C, H, W = inputs["image"].shape
        device = inputs["image"].device

        rays_gt = inputs.get("rays", None)

        common_shape = self.common_shape

        features = self.input_adapter(inputs["features"])

        # positional embeddings, spatial and level
        level_embed = self.level_embeds.repeat(
            B, common_shape[0] * common_shape[1], 1, 1
        )
        level_embed = rearrange(level_embed, "b n l d -> b (n l) d")
        dummy_tensor = torch.zeros(
            B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False
        )
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(1, 4, 1)

        # get cls tokens projections
        camera_tokens = inputs["tokens"]
        camera_tokens = self.camera_token_adapter(camera_tokens)
        self.angular_module.set_shapes((H, W))
        intrinsics, rays = self.run_camera(
            torch.cat(camera_tokens, dim=1),
            original_shapes=(H, W),
            rays_gt=rays_gt,
        )
        lograd_teacher, logconf_teacher, lateral_features_teacher = (
            self.radial_module.forward_keyframe(features, rays)
        )
        rad_teacher = torch.exp(lograd_teacher.clip(min=-8.0, max=8.0) + 2.0)
        confidence_teacher = torch.exp(logconf_teacher.clip(min=-8.0, max=10.0))
        return rays, rad_teacher, confidence_teacher, lateral_features_teacher

    def forward_fast(self, inputs, image_metas):
        B, T = image_metas[0]["B"], image_metas[0]["T"]
        features_teacher = inputs["features_teacher"]
        features_teacher = [
            x.reshape(B, -1, *x.shape[1:])[:, :1].reshape(-1, *x.shape[1:])
            for x in features_teacher
        ]
        # at T > 1, this come from the previous prop
        features_residual = inputs["features_residual"][::-1]
        lograd_student, logconf_student, lateral_features_student = (
            self.radial_module.forward_fast(
                features_teacher,
                features_residual,
                flow_bwd=inputs["flow_bwd_refined"],
                flow_features=inputs["flow_features"],
            )
        )
        rad_student = torch.exp(lograd_student.clip(min=-8.0, max=8.0) + 2.0)
        confidence_student = torch.exp(logconf_student.clip(min=-8.0, max=10.0))
        return rad_student, confidence_student, lateral_features_student

    def set_shapes(self, inputs, image_metas) -> torch.Tensor:
        H, W = inputs["image"].shape[-2:]
        self.common_shape = inputs["features"][0].shape[1:3]
        self.radial_module.set_shapes(self.common_shape)
        self.radial_module.set_original_shapes((H, W))
        self.angular_module.set_shapes((H, W))

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def train(self, mode=True):
        super().train(mode)
        self.radial_module.train(mode)
        self.angular_module.train(False)
        for p in self.angular_module.parameters():
            p.requires_grad = False

        self.input_adapter.train(False)
        for p in self.input_adapter.parameters():
            p.requires_grad = False

        self.camera_token_adapter.train(False)
        for p in self.camera_token_adapter.parameters():
            p.requires_grad = False

    def get_params(self, lr, wd):
        angles_p = get_params(self.angular_module, lr, wd)[0]
        radius_p = get_params(self.radial_module, lr, wd)[0]
        tokens_p = get_params(self.camera_token_adapter, lr, wd)[0]
        input_p = get_params(self.input_adapter, lr, wd)[0]
        return [*tokens_p, *angles_p, *input_p, *radius_p]

    def build(self, config):
        model_cfg = config["model"]
        decoder_cfg = model_cfg["pixel_decoder"]
        encoder_cfg = model_cfg["pixel_encoder"]
        residual_cfg = model_cfg["residual_encoder"]

        input_dims = encoder_cfg["embed_dims"]
        residual_input_dims = residual_cfg["embed_dims"]
        hidden_dim = decoder_cfg["hidden_dim"]
        expansion = model_cfg["expansion"]
        num_heads = model_cfg["num_heads"]
        dropout = decoder_cfg["dropout"]
        layer_scale = model_cfg["layer_scale"]
        depth = decoder_cfg["depths"]
        depths_encoder = encoder_cfg["depths"]
        out_dim = decoder_cfg["out_dim"]
        kernel_size = decoder_cfg.get("kernel_size", 3)
        self.slices_encoder = list(zip([d - 1 for d in depths_encoder], depths_encoder))
        input_dims = [input_dims[d - 1] for d in depths_encoder]
        self.num_resolutions = len(depths_encoder)
        self.steps = 0

        camera_dims = input_dims
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.camera_token_adapter = ListAdapter(camera_dims, hidden_dim)
        self.angular_module = AngularModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.radial_module = RadialModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=96,
            layer_scale=layer_scale,
            out_dim=out_dim,
            kernel_size=kernel_size,
            num_prompt_blocks=decoder_cfg["num_prompt_blocks"],
            residual_dims=residual_input_dims[::-1],
            num_fusion_block=decoder_cfg["num_fusion_block"],
        )
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            orthonormal_init(len(input_dims), hidden_dim).reshape(
                1, 1, len(input_dims), hidden_dim
            ),
            requires_grad=False,
        )
        self.camera_gt = True
