"""
Author: Luigi Piccinelli
Licensed under the CC BY-NC-SA 4.0 license (http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import importlib
import warnings
from copy import deepcopy
from contextlib import nullcontext
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from velodepth.models.decoder_velo import Decoder, flow2grid_sample
from velodepth.models.encoder_velo import SmallEncoder, denormalize, rgb2luma
from velodepth.utils.camera import BatchCamera, Camera
from velodepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
from velodepth.utils.distributed import is_main_process
from velodepth.utils.flow import forward_backward_consistency_check
from velodepth.utils.misc import get_params, last_stack, match_gt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLED = torch.cuda.is_available()

FLOW_THR = 0.8
FLOW_ALPHA = 0.1
FLOW_BETA = 0.15
LUMA_ALPHA = 0.0
LUMA_BETA = 0.1
MAX_TIME = 10

@torch.no_grad()
def run_raft(model, pair, mask):
    """Run RAFT forward/backward on a pair of frames (pair[0], pair[1])."""
    pair = 2 * denormalize(pair) - 1
    flow_bwd = model(pair[1:], pair[:1])["flow"]
    flow_fwd = model(pair[:1], pair[1:])["flow"]
    flow_bwd[~mask] = 0.0
    flow_fwd[~mask] = 0.0
    return flow_bwd, flow_fwd


def transform_tensor(src_batch, dst_batch, pad_src, pad_dst, mode: str = "nearest-exact"):
    """
    Resize/crop/pad each tensor in `src_batch` to match the spatial extent of
    the corresponding tensor in `dst_batch`, given per-sample paddings.
    """
    out = []
    src_dtype = src_batch[0].dtype
    tgt_dtype = dst_batch[0].dtype
    for i, (src, dst) in enumerate(zip(src_batch, dst_batch)):
        p1 = pad_src[i] if pad_src is not None else (0, 0, 0, 0)
        p2 = pad_dst[i] if pad_dst is not None else (0, 0, 0, 0)
        if isinstance(p1, torch.Tensor): p1 = p1.tolist()
        if isinstance(p2, torch.Tensor): p2 = p2.tolist()
        # lrtb
        l1, r1, t1, b1 = p1
        l2, r2, t2, b2 = p2

        trimmed = src[:, t1: src.shape[1]-b1, l1: src.shape[2]-r1]
        target_h = dst.shape[1] - t2 - b2
        target_w = dst.shape[2] - l2 - r2
        resized = F.interpolate(trimmed.unsqueeze(0).to(tgt_dtype),
                                size=(target_h, target_w),
                                mode=mode)
        out.append(F.pad(resized, (l2, r2, t2, b2)))
    return torch.cat(out).to(src_dtype)


def find_bounding_box(masks):
    """Compute LTRB limits and LRTB paddings (per-sample) of valid regions."""
    limits, paddings = [], []
    for mask in masks:
        m = mask.squeeze()
        if m.all():
            h, w = m.shape
            limits.append((0, 0, w - 1, h - 1))
            paddings.append((0, 0, 0, 0))
            continue
        rows = torch.any(m, dim=1)
        cols = torch.any(m, dim=0)
        y_idx = torch.nonzero(rows, as_tuple=False).flatten()
        x_idx = torch.nonzero(cols, as_tuple=False).flatten()
        y_min, y_max = y_idx.min().item(), y_idx.max().item()
        x_min, x_max = x_idx.min().item(), x_idx.max().item()
        ltrb = (x_min, y_min, x_max, y_max)
        lrtb = (x_min, m.shape[1] - x_max - 1, y_min, m.shape[0] - y_max - 1)
        limits.append(ltrb)
        paddings.append(lrtb)
    return limits, paddings


def get_paddings(original_shape, aspect_ratio_range):
    # Original dimensions
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    # Determine the closest aspect ratio within the range
    min_ratio, max_ratio = aspect_ratio_range
    target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape, pixels_range, shape_multiplier=14):
    # Original dimensions
    H_ori, W_ori = original_shape
    n_pixels_ori = W_ori * H_ori

    # Determine the closest number of pixels within the range
    min_pixels, max_pixels = pixels_range
    target_pixels = min(max_pixels, max(min_pixels, n_pixels_ori))

    # Calculate the resize factor
    resize_factor = (target_pixels / n_pixels_ori) ** 0.5
    new_width = int(W_ori * resize_factor)
    new_height = int(H_ori * resize_factor)
    new_height = ceil(new_height / shape_multiplier) * shape_multiplier
    new_width = ceil(new_width / shape_multiplier) * shape_multiplier

    return resize_factor, (new_height, new_width)


def _postprocess(tensor, shapes, paddings, interpolation_mode="bilinear"):

    # interpolate to original size
    tensor = F.interpolate(
        tensor, size=shapes, mode=interpolation_mode, align_corners=False
    )

    # remove paddings
    pad1_l, pad1_r, pad1_t, pad1_b = paddings
    tensor = tensor[..., pad1_t : shapes[0] - pad1_b, pad1_l : shapes[1] - pad1_r]
    return tensor

class VeloDepth(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="VeloDepth",
    repo_url="https://github.com/lpiccinelli-eth/VeloDepth",
    tags=["video-depth-propagation", "metric-4D-estimation"],
):
# class VeloDepth(nn.Module):
    def __init__(self, config, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.build(config)

        # Sensible defaults for KF heuristics (can be overridden from config)
        self.alpha_flow = config.get("model", {}).get("alpha_flow", FLOW_ALPHA)
        self.beta_flow  = config.get("model", {}).get("beta_flow",  FLOW_BETA)
        self.alpha_luma = config.get("model", {}).get("alpha_luma", FLOW_ALPHA)
        self.beta_luma  = config.get("model", {}).get("beta_luma",  FLOW_BETA)

    @staticmethod
    def _flat(x, B, T):
        return x.reshape(B * T, *x.shape[2:]) if isinstance(x, torch.Tensor) else x.reshape(-1)

    @staticmethod
    def _bt(x, B, T):
        return x.reshape(B, T, *x.shape[1:]) if isinstance(x, torch.Tensor) else x.reshape(B, T)

    def generate_camera(self, inputs, image_metas):
        B, T, _, H, W = inputs["image"].shape
        device = inputs["image"].device
        if "camera" in inputs:
            rays = inputs["camera"].get_rays(shapes=(B * T, H, W), noisy=self.training)
            inputs["rays"] = rays.reshape(B, T, 3, H, W)
        return inputs

    def generate_depth_infos(self, inputs, image_metas):
        B, T, _, H, W = inputs["image"].shape
        device = inputs["image"].device

        def to_tensor_list(key):
            return torch.tensor([m[key] for m in image_metas], device=device)

        # inputs["ssi"] = to_tensor_list("ssi")
        # inputs["si"] = to_tensor_list("si")
        inputs["paddings"] = torch.tensor([m["paddings"] for m in image_metas], device=device)[..., [0, 2, 1, 3]]
        if "depth_paddings" in image_metas[0]:
            inputs["depth_paddings"] = torch.tensor([m["depth_paddings"] for m in image_metas], device=device)
            if self.training:
                # At inference we don't carry image paddings on top of depth ones.
                inputs["depth_paddings"] = inputs["depth_paddings"] + inputs["paddings"]
        return inputs

    def pack_sequence(self, inputs: dict[str, torch.Tensor]):
        B, T = inputs["image"].shape[:2]
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = self._flat(v, B, T)
            elif isinstance(v, BatchCamera):
                inputs[k] = v.reshape(-1)
        return inputs

    def unpack_sequence(self, outputs: dict[str, torch.Tensor], B: int, T: int):
        for k, v in list(outputs.items()):
            if isinstance(v, torch.Tensor):
                outputs[k] = self._bt(v, B, T)
            elif isinstance(v, BatchCamera):
                outputs[k] = v.reshape(B, T)
        return outputs

    def forward(self, inputs, image_metas):
        if self.training:
            return self.forward_train(inputs, image_metas)
        return self.forward_test(inputs, image_metas)

    def forward_train(self, inputs, image_metas):
        image = inputs["image"]
        B, T, C, H, W = image.shape
        device, dtype = image.device, image.dtype

        # Keep at most MAX_TIME+1 frames (KF included)
        start_idx = max(0, T - (MAX_TIME + 1))
        grad_T    = min(T, MAX_TIME + 1)

        inputs = self.generate_camera(inputs, image_metas)
        inputs = self.generate_depth_infos(inputs, image_metas)
        inputs = self.pack_sequence(inputs)
        image_metas[0]["B"], image_metas[0]["T"] = B, T

        _, outputs = self.encode_decode(inputs, image_metas)

        validity_mask  = inputs["validity_mask"]
        original_image = inputs["image_original"]
        depth_mask     = inputs["depth_mask"].bool()
        depth_gt       = inputs["depth"]

        quality, dense, si = inputs["quality"], inputs["dense"], inputs["si"]

        radius_teacher    = outputs["radius_teacher"]
        radius_student    = outputs["radius_student"]
        confidence_student = outputs["confidence_student"]
        features_teacher  = outputs["features_teacher"]
        features_pred     = outputs["features_student"]

        # Align refined flow to (H, W) and time-strip to grad_T
        flow_bwd_refined = outputs["flow_bwd_refined"].reshape(B, T - 1, *outputs["flow_bwd_refined"].shape[1:])
        flow_bwd_refined = torch.cat([flow_bwd_refined, torch.zeros_like(flow_bwd_refined[:, :1])], dim=1)
        flow_bwd_refined = F.interpolate(flow_bwd_refined.reshape(-1, *flow_bwd_refined.shape[2:]),
                                         size=(H, W), mode="bilinear").reshape(B, T, -1, H, W)
        outputs["flow_bwd_refined"] = flow_bwd_refined[:, start_idx:].reshape(-1, *flow_bwd_refined.shape[2:])

        # Slice RAFT data to grad_T
        for k in ["flow_bwd_raft", "flow_fwd_raft", "flow_mask_raft"]:
            inputs[k] = self._flat(self._bt(inputs[k], B, T)[:, start_idx:], B, T - start_idx)

        # Slice student to grad_T-1
        radius_pred_sup = self._flat(self._bt(radius_student, B, T - 1)[:, start_idx:], B, T - 1 - start_idx)
        confidence_pred_sup = self._flat(self._bt(confidence_student, B, T - 1)[:, start_idx:], B, T - 1 - start_idx)

        # 3D predictions for supervision packaging
        pts_3d_teacher = outputs["predictions_3d_teacher"]
        pts_3d_student = self._bt(outputs["predictions_3d_student"], B, T - 1)[:, start_idx:]
        pts_3d_teacher_0 = self._bt(pts_3d_teacher, B, T)[:, start_idx:start_idx + 1]
        pts_3d_pred = torch.cat([pts_3d_teacher_0, pts_3d_student], dim=1).reshape(-1, *pts_3d_student.shape[2:])

        # Radius packaging over grad_T
        radius_teacher_0 = self._bt(radius_teacher, B, T)[:, start_idx:start_idx + 1]
        radius_pred = torch.cat([radius_teacher_0,
                                 self._bt(radius_pred_sup, B, grad_T - 1)], dim=1).reshape(-1, *radius_pred_sup.shape[1:])
        radius_teacher_sup = self._flat(self._bt(radius_teacher, B, T)[:, start_idx + 1:], B, grad_T - 1)
        radius_teacher_all = self._flat(self._bt(radius_teacher, B, T)[:, start_idx:], B, grad_T)

        # Feature lists sliced to grad_T-1
        features_teacher = [self._flat(self._bt(x, B, T)[:, start_idx + 1:], B, grad_T - 1)
                            for x in features_teacher]
        features_pred    = [self._flat(self._bt(x, B, T - 1)[:, start_idx:], B, grad_T - 1)
                            for x in features_pred]

        # GT radius/masks (project 3D)
        pts_gt   = inputs["camera"].reconstruct(depth_gt) * validity_mask.float()
        pts_gt   = torch.where(pts_gt.isnan().any(dim=1, keepdim=True), 0.0, pts_gt)
        radius_gt = pts_gt.norm(dim=1, keepdim=True)
        depth_mask = depth_mask & (~pts_gt.isnan().any(dim=1, keepdim=True)) & validity_mask.bool()

        depth_mask_sup = self._flat(self._bt(depth_mask, B, T)[:, start_idx + 1:], B, grad_T - 1)
        radius_gt_sup  = self._flat(self._bt(radius_gt, B, T)[:, start_idx + 1:],  B, grad_T - 1)
        radius_gt_all  = self._flat(self._bt(radius_gt, B, T)[:, start_idx:],      B, grad_T)
        depth_mask_all = self._flat(self._bt(depth_mask, B, T)[:, start_idx:],      B, grad_T)

        validity_mask_sup = self._flat(self._bt(validity_mask, B, T)[:, start_idx + 1:], B, grad_T - 1).bool()
        quality_sup = self._bt(quality, B, T)[:, start_idx + 1:].reshape(-1)
        dense_sup   = self._bt(dense,   B, T)[:, 0]
        si_sup      = self._bt(si,      B, T)[:, start_idx + 1:].reshape(-1)

        # Temporal weights
        weights = torch.tensor([i + 1 for i in range(grad_T - 1)], device=device, dtype=dtype).unsqueeze(0).repeat(B, 1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        has_gt  = (self._bt(depth_mask, B, grad_T)[:, 0].sum(dim=(-1, -2, -3)) > 0)

        # Residual-based spatial focus
        original_image_all = self._flat(self._bt(original_image, B, T)[:, start_idx:], B, grad_T).bool()
        validity_mask_all  = self._flat(self._bt(validity_mask, B, T)[:, start_idx:],  B, grad_T).bool()
        _luma = rgb2luma(denormalize(original_image_all)).unsqueeze(1)
        _flow = outputs["flow_bwd_refined"].detach()
        residuals = self.run_residual(_luma, _flow, B, grad_T, grad_T)
        weights_focus = (residuals.abs() > 0.1).float() * 2 + 1
        weights_focus = weights_focus * validity_mask_sup.sum() / weights_focus[validity_mask_sup].sum()

        # inputs payload
        inputs.update(dict(
            B=B, grad_T=grad_T, weights=weights, has_gt=has_gt,
            quality_sup=quality_sup, si_sup=si_sup,
            depth_mask_sup=depth_mask_sup, validity_mask_sup=validity_mask_sup,
            depth_mask=depth_mask_all, validity_mask=validity_mask_all,
            dense_sup=dense_sup, weights_focus=weights_focus,
        ))

        # outputs payload
        outputs.update(dict(
            radius_pred_sup=radius_pred_sup, radius_gt_sup=radius_gt_sup, radius_teacher_sup=radius_teacher_sup,
            radius_pred=radius_pred, radius_gt=radius_gt_all, radius_teacher=radius_teacher_all,
            features_teacher_list=features_teacher, features_pred_list=features_pred,
            pts_3d_pred=pts_3d_pred,
            flow_bwd_raft=inputs["flow_bwd_raft"], flow_fwd_raft=inputs["flow_fwd_raft"],
            flow_mask_raft=inputs["flow_mask_raft"],
        ))

        losses = self.compute_losses(inputs, outputs, image_metas)
        # keep opt graph connected in case some schedulers inspect it
        losses["opt"]["dummy_mask"] = 0.0 * inputs.get("mask_token", torch.tensor(0.0, device=device)).mean()

        out = {"depth": radius_pred.detach()}
        out = self.unpack_sequence(out, B, grad_T)
        return out, losses

    def forward_test(self, inputs, image_metas):
        losses, infos = {"opt": {}, "stat": {}}, {}
        image = inputs["image"]
        B, T, C, H, W = image.shape

        # copy + enrich + pack
        inputs = {k: v.clone() for k, v in inputs.items()}
        inputs = self.generate_camera(inputs, image_metas)
        inputs = self.generate_depth_infos(inputs, image_metas)
        inputs = self.pack_sequence(inputs)
        image_metas[0]["B"], image_metas[0]["T"] = B, T

        _, outputs = self.encode_decode_test(inputs, image_metas)

        # Assume B=1 at test. Map preds to GT canvas.
        depth_mask = inputs["depth_mask"].bool()
        depth_gt = inputs["depth"]
        points_pred = outputs["points"].squeeze(0)

        points_pred2gt = match_gt(
            points_pred, depth_gt, padding1=inputs["paddings"], padding2=None
        )
        depth_pred2gt = points_pred2gt[:, -1:]

        losses["stat"]["MSE_meter"] = F.mse_loss(depth_pred2gt[depth_mask], depth_gt[depth_mask]).sqrt()

        out = {"depth": depth_pred2gt.clamp(min=0.01), "points": points_pred2gt, **infos}
        out = self.unpack_sequence(out, B, T)
        return ({"depth": out["depth"], "points": out["points"], "infos": {}}, losses)

    def _set_image_paddings(self, inputs):
        """Compute image paddings from validity mask and store tensors."""
        limits, paddings = find_bounding_box(inputs["validity_mask"])
        device = inputs["image"].device
        inputs["image_limits"]   = limits
        inputs["image_paddings"] = torch.tensor(paddings, device=device)

    def _normalize_and_grid_from_flow(self, flow_px, rgbs, image_paddings, depth_paddings):
        """
        If needed (test), transform flow from GT canvas to image canvas, then
        normalize to [-1,1] grid coordinates and return both flow and grid.
        """
        H, W = rgbs.shape[-2:]
        rescale = torch.tensor([W - 1, H - 1], device=rgbs.device).view(1, 2, 1, 1)

        # In training flows are already aligned to image, in test we adapt.
        if not self.training:
            H_gt, W_gt = flow_px.shape[-2:]
            (l1, r1, t1, b1) = depth_paddings[0] if depth_paddings[0] is not None else (0, 0, 0, 0)
            (l2, r2, t2, b2) = image_paddings[0] if image_paddings[0] is not None else (0, 0, 0, 0)
            H_resize = (H - t1 - b1) / max(1, (H_gt - t2 - b2))
            W_resize = (W - l1 - r1) / max(1, (W_gt - l2 - r2))

            flow_px = transform_tensor(flow_px, rgbs,
                                       depth_paddings,
                                       image_paddings)
            flow_px[:, 0] = flow_px[:, 0] * W_resize
            flow_px[:, 1] = flow_px[:, 1] * H_resize

        flow = flow_px / rescale
        return flow, flow2grid_sample(flow.clone())

    def _run_raft_block(self, inputs, rgbs_bt, B, T, rescale):
        """Compute RAFT fwd/bwd and the consistency mask on packed time (B*T)."""
        flow_bwd_raft, flow_fwd_raft = [], []
        rgbs_original = rgbs_bt.reshape(B, T, *rgbs_bt.shape[1:])
        for b in range(B):
            flow_bwd_clip, flow_fwd_clip = [], []
            validity_mask = inputs["validity_mask"].reshape(B, T, 1, *rgbs_bt.shape[2:])[b, :1].repeat(1, 2, 1, 1).bool()
            for t in range(0, T - 1):
                fb, ff = run_raft(self.raft, rgbs_original[b, t:t + 2], mask=validity_mask)
                flow_bwd_clip.append(fb)
                flow_fwd_clip.append(ff)
            flow_bwd_raft.append(torch.cat([*flow_bwd_clip, torch.zeros_like(flow_bwd_clip[0])], dim=0))
            flow_fwd_raft.append(torch.cat([*flow_fwd_clip, torch.zeros_like(flow_fwd_clip[0])], dim=0))

        flow_bwd_px = torch.stack(flow_bwd_raft, dim=0).reshape(B * T, *flow_bwd_raft[0].shape[1:])
        flow_fwd_px = torch.stack(flow_fwd_raft, dim=0).reshape(B * T, *flow_fwd_raft[0].shape[1:])
        flow_mag_px = (flow_bwd_px.norm(dim=1, keepdim=True) + flow_fwd_px.norm(dim=1, keepdim=True)) / 2
        fb_cons = forward_backward_consistency_check(flow_fwd_px, flow_bwd_px)[-1]  # backward mask
        flow_mask = torch.tanh(fb_cons / (FLOW_ALPHA * flow_mag_px + FLOW_BETA))

        return flow_fwd_px / rescale, flow_bwd_px / rescale, flow_mask
    
    def run_residual(self, rgbs, flow, B, T, T_flow):
        rgbs = rgbs.clone()
        grid = flow2grid_sample(flow.clone()).reshape(B, T_flow, *flow.shape[1:])
        rgbs_bt = rgbs.reshape(B, T, *rgbs.shape[1:])
        if T < 2:
            return torch.zeros(0, rgbs.shape[1], *rgbs.shape[-2:], device=rgbs.device)
        residuals = []
        for t in range(T - 1):
            warped = F.grid_sample(rgbs_bt[:, t], grid[:, t], mode="bilinear")
            residuals.append(rgbs_bt[:, t + 1] - warped)
        return torch.stack(residuals, dim=1).reshape(B * (T - 1), *rgbs.shape[1:])

    @torch.no_grad()
    def encode_decode_test(self, inputs, image_metas):
        rgbs = inputs["image"].clone()
        device = rgbs.device
        H, W = rgbs.shape[-2:]
        B, T = image_metas[0]["B"], image_metas[0]["T"]

        self._set_image_paddings(inputs)
        flow_bwd = inputs["flow_bwd"]
        flow_bwd, grid_sample = self._normalize_and_grid_from_flow(
            flow_bwd, rgbs, inputs["image_paddings"], inputs["depth_paddings"]
        )

        inputs["grid_sample"], inputs["flow_bwd"] = grid_sample, flow_bwd

        if "rays" in inputs:
            self.pixel_decoder.test_gt_camera = True

        image_bt = inputs["image"]

        current_t = 0
        fast_tail_radius, fast_tail_confidence = None, None
        overlap = 1
        schedule = np.linspace(0, 1, max(1, overlap) + 1, endpoint=True)
        radius_list, rays_list, confidence_list = [], [], []
        _first = True
        forget = 0.5
        while current_t < T:
            total_luma_change, flow_magnitude, distance_kf = 0.0, 0.0, 1

            # --- Keyframe pass ---
            base_inputs, base_metas = {}, [{}]
            base_inputs["image"] = self._bt(image_bt, B, T)[:, current_t]
            base_inputs["image_original"] = self._bt(image_bt, B, T)[:, current_t]
            if "rays" in inputs:
                base_inputs["rays"] = self._bt(inputs["rays"], B, T)[:, current_t]
            base_inputs["validity_mask"] = self._bt(inputs["validity_mask"], B, T)[:, current_t]
            base_metas[0]["T"], base_metas[0]["B"] = 1, B

            base_features, base_tokens = self.pixel_encoder(base_inputs["image"])
            base_features = [self.stacking_fn(base_features[i:j]).contiguous()
                             for i, j in self.slices_encoder_range]
            base_tokens  = [self.stacking_fn(base_tokens[i:j]).contiguous()
                            for i, j in self.slices_encoder_range]
            base_inputs["features"], base_inputs["tokens"] = base_features, base_tokens

            self.pixel_decoder.set_shapes(base_inputs, base_metas)
            rays_base, radius_teacher, confidence_teacher, features_teacher = self.pixel_decoder.forward_keyframe(base_inputs, base_metas)

            if not _first and overlap > 0:
                radius_teacher = schedule[0] * radius_teacher + (1 - schedule[0]) * fast_tail_radius
                confidence_teacher = schedule[0] * confidence_teacher + (1 - schedule[0]) * fast_tail_confidence
                fast_tail_features = features_teacher

            radius_list.append(radius_teacher)
            rays_list.append(rearrange(rays_base, "b (h w) c -> b c h w", h=H, w=W))
            confidence_list.append(confidence_teacher)

            current_t += 1

            # --- Fast propagation until next KF ---
            while current_t < T:
                fast_in, fast_metas = {}, [{}]
                curr_img = self._bt(image_bt, B, T)[:, current_t]
                prev_img = self._bt(image_bt, B, T)[:, current_t - 1]

                fast_in["grid_sample"] = self._bt(inputs["grid_sample"], B, T)[:, current_t - 1]
                fast_in["image_residual"] = rgb2luma(curr_img).unsqueeze(1) - F.grid_sample(
                    rgb2luma(prev_img).unsqueeze(1), fast_in["grid_sample"], mode="bilinear"
                )
                fast_in["image"] = curr_img
                fast_in["image_original"] = self._bt(image_bt, B, T)[:, current_t]
                if "rays" in inputs:
                    fast_in["rays"] = self._bt(inputs["rays"], B, T)[:, current_t]
                fast_in["validity_mask"] = self._bt(inputs["validity_mask"], B, T)[:, current_t]
                fast_in["flow_bwd"] = self._bt(inputs["flow_bwd"], B, T)[:, current_t - 1]
                fast_in["features_teacher"] = features_teacher
                fast_metas[0]["T"], fast_metas[0]["B"] = 1, B

                flow_t_tm1 = fast_in["flow_bwd"].clone()
                residuals_t_tm1 = fast_in["image_residual"].clone()

                features_residual, flow_refined, flow_features = self.small_encoder(
                    rgb_t=curr_img, depth_t=radius_teacher, flow_t=flow_t_tm1, raft_t=None, residuals_t=residuals_t_tm1
                )
                if len(features_residual) > 4:  # ViT case: keep last stage of each block
                    features_residual = [x.permute(0, 3, 1, 2) for i, x in enumerate(features_residual) if i in [2, 5, 8, 11]]

                fast_in["features_residual"] = features_residual
                fast_in["flow_features"] = flow_features
                fast_in["flow_bwd_refined"] = flow_refined
                radius_fast, confidence_fast, feature_fast = self.pixel_decoder.forward_fast(fast_in, fast_metas)

                if distance_kf - 1 < overlap and not _first:
                    fast_in["features_teacher"] = fast_tail_features
                    radius_fast_overlap, confidence_fast_overlap, _ = self.pixel_decoder.forward_fast(fast_in, fast_metas)
                    radius_fast = schedule[distance_kf] * radius_fast + (1 - schedule[distance_kf]) * radius_fast_overlap
                    confidence_fast = schedule[distance_kf] * confidence_teacher + (1 - schedule[distance_kf]) * confidence_fast_overlap

                # Prepare for next fast step
                features_teacher = feature_fast
                radius_teacher   = radius_fast
                fast_tail_radius = radius_fast
                confidence_teacher = confidence_fast
                fast_tail_confidence = confidence_fast
                fast_tail_features = features_teacher

                # Heuristics for KF restart
                flow_magnitude = flow_magnitude * forget + flow_t_tm1.norm(dim=1).quantile(0.95).item()
                flow_heur = flow_magnitude > (self.alpha_flow + self.beta_flow * (0.9 ** (distance_kf - 1)))

                mask_current = (F.grid_sample(torch.ones_like(fast_in["validity_mask"]).float(),
                                              flow2grid_sample(fast_in["flow_bwd_refined"].clone(), mask=False),
                                              mode="nearest") < 0.5)
                current_flowmask_ = mask_current & fast_in["validity_mask"].bool()
                total_luma_change = total_luma_change * forget + current_flowmask_.float().mean().item()
                luma_heur = total_luma_change > (self.alpha_luma + self.beta_luma * (0.9 ** (distance_kf - 1)))

                if flow_heur or luma_heur:
                    _first = False
                    break  # new KF

                radius_list.append(radius_teacher)
                rays_list.append(rearrange(rays_base, "b (h w) c -> b c h w", h=H, w=W))
                confidence_list.append(confidence_teacher)
                distance_kf += 1
                current_t += 1

        radius = torch.cat(radius_list, dim=0)
        rays = torch.cat(rays_list, dim=0)
        confidence = torch.cat(confidence_list, dim=0)
        out = {"points": radius * rays, "rays": rays, "radius": radius, "confidence": confidence}
        self.pixel_decoder.test_gt_camera = False
        return inputs, out

    def encode_decode(self, inputs, image_metas):
        rgbs = inputs["image"].clone()
        H, W = rgbs.shape[-2:]
        B, T = image_metas[0]["B"], image_metas[0]["T"]

        self._set_image_paddings(inputs)

        # Flow normalization + grid
        flow_bwd = inputs["flow_bwd"]
        # flows already on image canvas
        rescale = torch.tensor([W - 1, H - 1], device=rgbs.device).view(1, 2, 1, 1)
        flow_bwd = flow_bwd / rescale
        grid_sample = flow2grid_sample(flow_bwd.clone())

        inputs["grid_sample"] = grid_sample
        inputs["flow_bwd"] = flow_bwd

        if "rays" in inputs:
            self.pixel_decoder.test_gt_camera = True

        # Optional RAFT supervision (compute on original images)
        if self.raft is not None:
            rescale = torch.tensor([W - 1, H - 1], device=rgbs.device).view(1, 2, 1, 1)
            flow_fwd_raft, flow_bwd_raft, flow_mask_raft = self._run_raft_block(inputs, inputs["image_original"], B, T, rescale)
            inputs["flow_fwd_raft"] = flow_fwd_raft
            inputs["flow_bwd_raft"] = flow_bwd_raft
            inputs["flow_mask_raft"] = flow_mask_raft

        # Storage for all time
        outputs = {}
        interval = 64
        chunk_memsave_kf = B

        # Mask token (handle encoders without it)
        inputs["mask_token"] = getattr(self.pixel_encoder, "mask_token", torch.tensor(0.0, device=inputs["image"].device))
        image_bt = inputs["image"]
        image_orig_bt = inputs["image_original"]

        for t0_kf in range(0, T, interval):
            t1_kf = min(t0_kf + interval, T)
            chunk_kf = t1_kf - t0_kf
            sl = slice(t0_kf, t1_kf)

            # Build current slice inputs
            cur_in, cur_meta = {}, [{}]
            cur_meta[0]["T"], cur_meta[0]["B"] = chunk_kf, B

            def bt_slice(x): return self._bt(x, B, T)[:, sl].reshape(B * chunk_kf, *x.shape[2:])

            cur_in["image"]          = bt_slice(image_bt)
            cur_in["image_original"] = bt_slice(image_orig_bt)
            cur_in["rays"]           = bt_slice(inputs["rays"])
            cur_in["grid_sample"]    = bt_slice(inputs["grid_sample"])
            cur_in["validity_mask"]  = bt_slice(inputs["validity_mask"])
            cur_in["flow_bwd"]       = bt_slice(inputs["flow_bwd"])
            cur_in["flow_bwd_raft"]  = bt_slice(inputs["flow_bwd_raft"]) if "flow_bwd_raft" in inputs else None
            flow_mask_raft = inputs.get("flow_mask_raft", None)
            if flow_mask_raft is not None:
                cur_in["flow_mask_raft"] = self._bt(flow_mask_raft, B, T)[:, sl].reshape(B * chunk_kf, *flow_mask_raft.shape[1:])
            cur_in["image_residual"] = self.run_residual(cur_in["image_original"], cur_in["flow_bwd"], B, chunk_kf, chunk_kf)

            # ---- Teacher (KF) pass in micro-batches to save memory ----
            with torch.no_grad():
                all_rays, all_radius_t, all_conf_t = [], [], []
                all_feat_t = [[] for _ in range(4)]
                batch_size = cur_in["image_original"].shape[0]

                for st in range(0, batch_size, chunk_memsave_kf):
                    ed = min(st + chunk_memsave_kf, batch_size)
                    img_chunk = cur_in["image_original"][st:ed]

                    stud_features, stud_tokens = self.pixel_encoder(img_chunk)
                    stud_features = [self.stacking_fn(x).contiguous() for x in stud_features]
                    stud_features = [self.stacking_fn(stud_features[i:j]).contiguous() for i, j in self.slices_encoder_range]
                    stud_tokens  = [self.stacking_fn(stud_tokens[i:j]).contiguous() for i, j in self.slices_encoder_range]

                    chunked = dict(
                        image=img_chunk,
                        rays=cur_in["rays"][st:ed],
                        features=stud_features,
                        tokens=stud_tokens,
                        camera_tokens=stud_tokens,
                    )
                    self.pixel_decoder.set_shapes(chunked, cur_meta)
                    rays_t, rad_t, conf_t, lat_feat_t = self.pixel_decoder.forward_keyframe(chunked, cur_meta)

                    all_rays.append(rearrange(rays_t, "b (h w) c -> b c h w", h=H, w=W))
                    all_radius_t.append(rad_t)
                    all_conf_t.append(conf_t)
                    for i, f in enumerate(lat_feat_t):
                        all_feat_t[i].append(f)

                features_teacher = [torch.cat(all_feat_t[i], dim=0) for i in range(4)]
                radius_teacher   = torch.cat(all_radius_t, dim=0)
                outputs_teacher  = dict(
                    radius_teacher=radius_teacher,
                    confidence_teacher=torch.cat(all_conf_t, dim=0),
                    features_teacher=features_teacher,
                    rays_teacher=torch.cat(all_rays, dim=0),
                )
                # reshape to B,T,...
                for k, v in list(outputs_teacher.items()):
                    if isinstance(v, torch.Tensor):
                        outputs_teacher[k] = v.reshape(B, batch_size // B, *v.shape[1:])
                    else:
                        outputs_teacher[k] = [x.reshape(B, batch_size // B, *x.shape[1:]) for x in v]

            cur_in["features_teacher"] = features_teacher

            # ---- Student propagation within this KF window ----
            rad_student_list, conf_student_list, flow_refined_list = [], [], []
            student_feat_list = [[] for _ in range(4)]

            features_teacher_bt = [x.reshape(B, -1, *x.shape[1:])[:, 0] for x in features_teacher]
            radius_teacher_bt   = radius_teacher.reshape(B, -1, *radius_teacher.shape[1:])[:, 0]

            prop_img      = cur_in["image"].reshape(B, chunk_kf, *cur_in["image"].shape[1:])
            flow_bwd_all  = cur_in["flow_bwd"].reshape(B, chunk_kf, *cur_in["flow_bwd"].shape[1:])[:, :-1]
            flow_bwd_raft = (cur_in["flow_bwd_raft"].reshape(B, chunk_kf, *cur_in["flow_bwd_raft"].shape[1:])[:, :-1]
                             if cur_in.get("flow_bwd_raft", None) is not None else None)
            image_residual = cur_in["image_residual"].reshape(B, chunk_kf - 1, *cur_in["image_residual"].shape[1:])
            rays_t = outputs_teacher["rays_teacher"][:, 0].permute(0, 2, 3, 1).reshape(B, H * W, 3)

            for t0 in range(0, chunk_kf - 1):  # skip KF
                cur_meta[0]["T"] = 1
                rgb_t = prop_img[:, t0 + 1]
                flow_t_tm1 = flow_bwd_all[:, t0].clone()
                raft_t_tm1 = flow_bwd_raft[:, t0].clone() if flow_bwd_raft is not None else None
                residuals_t_tm1 = image_residual[:, t0].clone()

                with (torch.no_grad() if (t0 < chunk_kf - 1 - MAX_TIME and t0 > 0) else nullcontext()):
                    feat_res, flow_ref, flow_feat = self.small_encoder(
                        rgb_t=rgb_t, depth_t=radius_teacher_bt, flow_t=flow_t_tm1, raft_t=raft_t_tm1,
                        residuals_t=residuals_t_tm1
                    )
                    if len(feat_res) > 4:
                        feat_res = [x.permute(0, 3, 1, 2) for i, x in enumerate(feat_res) if i in [2, 5, 8, 11]]

                    chunked = dict(features_teacher=features_teacher_bt, features_residual=feat_res,
                                   flow_features=flow_feat, flow_bwd=flow_ref, rays=rays_t)
                    rad_s, conf_s, lat_feat_s = self.pixel_decoder.forward_fast(chunked, cur_meta)

                rad_student_list.append(rad_s)
                conf_student_list.append(conf_s)
                flow_refined_list.append(flow_ref)
                for i, f in enumerate(lat_feat_s): student_feat_list[i].append(f)

                features_teacher_bt = lat_feat_s
                radius_teacher_bt   = rad_s

            if chunk_kf > 1:
                student_features = [torch.cat([y.reshape(B, -1, *y.shape[1:]) for y in x], dim=1) for x in student_feat_list]
                rad_student      = torch.cat([y.reshape(B, -1, *y.shape[1:]) for y in rad_student_list], dim=1)
                conf_student     = torch.cat([y.reshape(B, -1, *y.shape[1:]) for y in conf_student_list], dim=1)
                flow_refined     = (torch.cat([y.reshape(B, -1, *y.shape[1:]) for y in flow_refined_list], dim=1)
                                    if flow_refined_list[0] is not None else None)
            else:
                student_features, rad_student, conf_student, flow_refined = None, None, None, None  # no propagation

            out_kf = dict(
                **outputs_teacher,
                features_student=student_features,
                radius_student=rad_student,
                confidence_student=conf_student,
                predictions_3d_teacher=outputs_teacher["rays_teacher"] * outputs_teacher["radius_teacher"],
                predictions_3d_student=outputs_teacher["rays_teacher"][:, :1] * (rad_student.clone() if rad_student is not None else outputs_teacher["radius_teacher"][:, :1]),
            )
            if flow_refined is not None:
                out_kf["flow_bwd_refined"] = flow_refined

            # Aggregate on T
            for k, v in out_kf.items():
                if k not in outputs:
                    outputs[k] = v
                else:
                    if isinstance(v, list):
                        outputs[k] = [torch.cat([a, b], dim=1) for a, b in zip(outputs[k], v)]
                    else:
                        outputs[k] = torch.cat([outputs[k], v], dim=1)

        # Flatten B,T to B*T for all tensor outputs
        outputs = {
            k: (v.reshape(-1, *v.shape[2:]) if isinstance(v, torch.Tensor)
                else [x.reshape(-1, *x.shape[2:]) for x in v])
            for k, v in outputs.items()
        }

        self.pixel_decoder.test_gt_camera = False
        return inputs, outputs

    @torch.no_grad()
    @torch.autocast(device_type=DEVICE, enabled=ENABLED, dtype=torch.float16)
    def infer(
        self,
        rgbs: torch.Tensor,                       # T C H W  or  B T C H W (uint8 or float in [0,255])
        camera: torch.Tensor | Camera | None = None,
        rays: torch.Tensor | None = None,
        normalize: bool = True,
    ):
        """
        Streamlined video inference:
        - pad to ratio + resize to pixel budget
        - optional camera/rays adjusted identically
        - DIS optical flow (backward t->t-1) on the network canvas (dy,dx)
        - encode_decode_test
        - _postprocess back to original HxW
        Returns dict of B T C H W (B=1 if input was TCHW).
        """
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("Please install OpenCV: `pip install opencv-python-headless`") from e
        ratio_bounds = self.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            self.shape_constraints["pixels_min"],
            self.shape_constraints["pixels_max"],
        ]
        if hasattr(self, "resolution_level"):
            assert (
                self.resolution_level >= 0 and self.resolution_level < 10
            ), "resolution_level should be in [0, 10)"
            pixels_range = pixels_bounds[1] - pixels_bounds[0]
            interval = pixels_range / 10
            new_lowbound = self.resolution_level * interval + pixels_bounds[0]
            new_upbound = (self.resolution_level + 1) * interval + pixels_bounds[0]
            pixels_bounds = (new_lowbound, new_upbound)
        else:
            warnings.warn("!! self.resolution_level not set, using default bounds !!")

        assert rgbs.ndim in (4, 5), "rgbs must be TCHW or BTCHW"
        if rgbs.ndim == 4:
            rgbs = rgbs.unsqueeze(0)  # 1 T C H W
        if camera is not None: # assume same camera
            camera = BatchCamera.from_camera(camera)
            camera = camera.to(self.device)

        B, T, C, H, W = rgbs.shape
        device = self.device

        # preprocess
        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor(
            (padded_H, padded_W), pixels_bounds
        )

        rgbs = rgbs.reshape(B * T, C, H, W).to(device)
        if normalize:
            rgbs = TF.normalize(
                rgbs.float() / 255.0,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )
        rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgbs = F.interpolate(
            rgbs, size=(new_H, new_W), mode="bilinear", align_corners=False
        )

        # -> camera preprocess
        if camera is not None:
            camera = camera.crop(
                left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom
            )
            camera = camera.resize(resize_factor)

        if camera is not None:
            cam = BatchCamera.from_camera(camera).to(device)
            cam = cam.crop(left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom)
            cam = cam.resize(resize_factor)

        inputs = {"image": rgbs}
        if camera is not None:
            inputs["camera"] = camera
            rays = camera.get_rays(shapes=(B*T, new_H, new_W), noisy=False).reshape(
                B * T, 3, new_H, new_W
            )
            inputs["rays"] = rays

        if rays is not None:
            rays = rays.to(self.device)
            if rays.ndim == 4:
                rays  =rays.unsqueeze(0).repeat(B, 1, 1, 1, 1).reshape(B * T, *rays.shape[2:])
            rays = F.pad(rays, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            rays = F.interpolate(
                rays, size=(new_H, new_W), mode="bilinear", align_corners=False
            )

        mean = torch.tensor(IMAGENET_DATASET_MEAN, device=rgbs.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DATASET_STD, device=rgbs.device).view(1, 3, 1, 1)
        frames_u8 = ((rgbs * std + mean).clamp(0, 1) * 255.0).byte().reshape(B, T, C, new_H, new_W).cpu()
        flows = []
        for b in range(B):
            if T == 1:
                flows.append(torch.zeros(1, 2, new_H, new_W, device=device))
                continue
            for t in range(T - 1):
                prev_rgb = frames_u8[b, t    ].permute(1, 2, 0).numpy()
                curr_rgb = frames_u8[b, t + 1].permute(1, 2, 0).numpy()
                f = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(
                    cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY),
                    None,
                )  # (H,W,2) = (dx,dy)
                f = torch.from_numpy(f).to(device=device).permute(2, 0, 1)
                flows.append(f.unsqueeze(0))
            flows.append(torch.zeros_like(flows[-1]))  # pad last
        flow_bwd = torch.cat(flows, dim=0)

        inputs.update({
            "flow_bwd": flow_bwd,
            "validity_mask": torch.ones(B * T, 1, new_H, new_W, device=device, dtype=torch.bool),
            "depth_paddings": torch.zeros(B * T, 4, device=device, dtype=torch.long),
        })
        
        # run model
        _, model_outputs = self.encode_decode_test(inputs, image_metas=[{"B": B, "T": T}])


        # collect outputs
        out = {}
        if "confidence" in model_outputs:
            out["confidence"] = _postprocess(
                model_outputs["confidence"],
                (padded_H, padded_W),
                paddings=paddings,
                interpolation_mode=self.interpolation_mode,
        )
        points = _postprocess(
            model_outputs["points"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        rays = _postprocess(
            model_outputs["rays"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        out["distance"] = points.norm(dim=1, keepdim=True)
        out["depth"] = points[:, -1:]
        out["points"] = points
        out["rays"] = rays / torch.norm(rays, dim=1, keepdim=True).clip(min=1e-5)
        out = {k: v.reshape(B, T, *v.shape[1:]).squeeze(0) for k, v in out.items()}
        return out

    def compute_losses(self, inputs, outputs, image_metas):
        losses = {"opt": {}, "stat": {}}
        to_compute = list(self.losses.keys())

        B, grad_T = inputs["B"], inputs["grad_T"]
        weights   = inputs["weights"]
        has_gt    = inputs["has_gt"].int()

        si = torch.tensor(
            [x.get("si", False) for x in image_metas], device=self.device
        ).reshape(B)

        # teacher defaults for teacher branches
        si_teacher = torch.zeros_like(si)

        # --- depth (GT + distill) ---
        depth_mod = self.losses["depth"]
        depth_gt = depth_mod(outputs["radius_pred_sup"], target=outputs["radius_gt_sup"],
                             mask=inputs["depth_mask_sup"].clone(), si=si, B=B, T=grad_T - 1)
        depth_t  = depth_mod(outputs["radius_pred_sup"], target=outputs["radius_teacher_sup"],
                             mask=inputs["validity_mask_sup"].clone(), si=si_teacher, B=B, T=grad_T - 1)
        depth_all = depth_gt * has_gt + depth_t * (1 - has_gt) * 0.2
        depth_val = depth_all.mean()
        losses["opt"][depth_mod.name] = depth_mod.weight * depth_val
        if depth_mod.weight <= 0.0: losses["stat"][depth_mod.name] = depth_val
        to_compute.remove("depth")

        # --- edge (over grad_T, GT or teacher depending on dens.) ---
        edge_mod = self.losses["edge"]
        edge_gt = edge_mod(outputs["radius_pred"], outputs["radius_gt"],
                           mask=inputs["depth_mask"].clone(), si=si,
                           B=B, T=grad_T, weights=inputs["weights_focus"]).reshape(B, grad_T - 1)
        edge_t = edge_mod(outputs["radius_pred"], outputs["radius_teacher"],
                          mask=inputs["validity_mask"].clone(),
                          B=B, T=grad_T, weights=inputs["weights_focus"]).reshape(B, grad_T - 1)
        edge_all = (weights * edge_gt).sum(dim=-1) * has_gt + (weights * edge_t).sum(dim=-1) * (1 - has_gt) * 0.2
        edge_val = edge_all.mean()
        losses["opt"][edge_mod.name] = edge_mod.weight * edge_val
        if edge_mod.weight <= 0.0: losses["stat"][edge_mod.name] = edge_val
        to_compute.remove("edge")

        # --- features (multi-scale) ---
        feat_mod = self.losses["features"]
        for i, (feat_gt, feat_pred) in enumerate(zip(outputs["features_teacher_list"], outputs["features_pred_list"])):
            mask_ = (F.interpolate(inputs["validity_mask_sup"].float(),
                                   size=feat_gt.shape[-2:], mode="nearest-exact") > 0.9)
            if (feat_pred.shape[-1] * feat_pred.shape[-2]) != (feat_gt.shape[-1] * feat_gt.shape[-2]):
                feat_pred = F.interpolate(feat_pred, size=feat_gt.shape[-2:], mode="bilinear", align_corners=False)
            feat_losses = feat_mod(feat_pred, feat_gt, mask=mask_)
            feat_losses = (weights * feat_losses.reshape(B, grad_T - 1)).sum(dim=-1)
            val = feat_losses.mean()
            losses["opt"][f"{feat_mod.name}_{i}"] = feat_mod.weight * val
            if feat_mod.weight <= 0.0: losses["stat"][f"{feat_mod.name}_{i}"] = val
        to_compute.remove("features")

        # --- flow supervision (RAFT) + self consistency ---
        flow_mod = self.losses["flow"]
        self_mod = self.losses["self"]

        def unwrap(x): return self._flat(self._bt(x, B, grad_T)[:, :-1], B, grad_T - 1)
        if outputs.get("flow_bwd_refined") is not None and getattr(self, "raft") is not None:
            # unwrap time (drop KF frames)
            
            flow_bwd_raft = unwrap(outputs["flow_bwd_raft"])
            flow_fwd_raft = unwrap(outputs["flow_fwd_raft"])
            flow_mask_raft = unwrap(outputs["flow_mask_raft"])
            flow_bwd_ref  = unwrap(outputs["flow_bwd_refined"])

            mask_raft = inputs["validity_mask_sup"] & (flow_mask_raft < FLOW_THR)
            flow_losses = flow_mod(flow_bwd_ref, flow_bwd_raft, mask=mask_raft).reshape(B, grad_T - 1)
            flow_losses = (weights * flow_losses).sum(dim=-1)
            flow_val = flow_losses.mean()
            losses["opt"][flow_mod.name] = 1.4142 * flow_mod.weight * flow_val
            if flow_mod.weight <= 0.0: losses["stat"][flow_mod.name] = flow_val
        to_compute.remove("flow")

        # Self-supervision via forward/backward consistency on 3D radius
        if outputs.get("flow_bwd_refined") is not None and getattr(self, "raft") is not None:
            # backward (t -> t+1)
            grid_bwd = flow2grid_sample(flow_bwd_raft.clone())
            pts_pred_bwd = unwrap(outputs["pts_3d_pred"])
            pts_gt_bwd = unwrap(outputs["pts_3d_pred"].detach())
            pts_pred_bwd_warp = F.grid_sample(pts_pred_bwd, grid_bwd, mode="bilinear", align_corners=True, padding_mode="border")
            mask_bwd = (inputs["validity_mask_sup"] & F.grid_sample(inputs["validity_mask_sup"].float(), grid_bwd, mode="bilinear", align_corners=True).bool() & (flow_mask_raft < FLOW_THR))
            mask_bwd_corr = (mask_bwd & (pts_pred_bwd_warp.norm(dim=1, keepdim=True) < 50.0) & (pts_gt_bwd.norm(dim=1, keepdim=True) < 50.0))
            transl_corr_bwd = torch.stack([
                (x[:, m.squeeze(0)].median(dim=-1)[0] - y[:, m.squeeze(0)].median(dim=-1)[0]) if m.any()
                else torch.zeros(3, device=x.device)
                for x, y, m in zip(pts_pred_bwd_warp, pts_gt_bwd, mask_bwd_corr)
            ])
            pts_pred_bwd_warp -= transl_corr_bwd.detach().view(B * (grad_T - 1), 3, 1, 1)
            rad_pred_bwd = pts_pred_bwd_warp.norm(dim=1, keepdim=True)
            rad_gt_bwd = pts_gt_bwd.norm(dim=1, keepdim=True)
            self_bwd = self_mod(rad_pred_bwd, target=rad_gt_bwd, mask=mask_bwd).reshape(B, grad_T - 1)
            self_bwd = (weights * self_bwd).sum(dim=-1)
            losses["opt"]["self_bwd"] = 0.5 * self_mod.weight * self_bwd.mean()
            if self_mod.weight <= 0.0: losses["stat"]["self_bwd"] = self_bwd.mean()

            # forward (t+1 -> t)
            grid_fwd = flow2grid_sample(flow_fwd_raft.clone())
            pts_pred_fwd = self._flat(self._bt(outputs["pts_3d_pred"], B, grad_T)[:, 1:], B, grad_T - 1)
            pts_gt_fwd   = self._flat(self._bt(outputs["pts_3d_pred"].detach(), B, grad_T)[:, :-1], B, grad_T - 1)
            pts_pred_fwd_warp = F.grid_sample(pts_pred_fwd, grid_fwd, mode="bilinear",
                                              align_corners=True, padding_mode="border")
            mask_fwd = (inputs["validity_mask_sup"] &
                        F.grid_sample(inputs["validity_mask_sup"].float(), grid_fwd,
                                      mode="bilinear", align_corners=True).bool() &
                        (flow_mask_raft < FLOW_THR))
            mask_fwd_corr = (mask_fwd &
                             (pts_pred_fwd_warp.norm(dim=1, keepdim=True) < 50.0) &
                             (pts_gt_fwd.norm(dim=1, keepdim=True) < 50.0))
            transl_corr_fwd = torch.stack([
                (x[:, m.squeeze(0)].median(dim=-1)[0] - y[:, m.squeeze(0)].median(dim=-1)[0]) if m.any()
                else torch.zeros(3, device=x.device)
                for x, y, m in zip(pts_pred_fwd_warp, pts_gt_fwd, mask_fwd_corr)
            ])
            pts_pred_fwd_warp -= transl_corr_fwd.detach().view(B * (grad_T - 1), 3, 1, 1)
            rad_pred_fwd = pts_pred_fwd_warp.norm(dim=1, keepdim=True)
            rad_gt_fwd   = pts_gt_fwd.norm(dim=1, keepdim=True)
            self_fwd = self_mod(rad_pred_fwd, target=rad_gt_fwd, mask=mask_fwd).reshape(B, grad_T - 1)
            self_fwd = (weights * self_fwd).sum(dim=-1)
            losses["opt"]["self_fwd"] = 0.5 * self_mod.weight * self_fwd.mean()
            if self_mod.weight <= 0.0: losses["stat"]["self_fwd"] = self_fwd.mean()

        to_compute.remove("self")
        assert not to_compute, f"Losses {to_compute} not computed, revise `compute_losses`."
        return losses

    def load_pretrained(self, model_file):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_model = torch.load(model_file, map_location=device, weights_only=False)
        if "model" in dict_model:
            dict_model = dict_model["model"]

        info = self.load_state_dict(deepcopy(dict_model), strict=False)
        if is_main_process():
            print(f"Loaded from {model_file} for {self.__class__.__name__}:\n", info)

    def build(self, config):
        # Pixel encoder
        enc_mod = importlib.import_module("velodepth.models.encoder")
        enc_factory = getattr(enc_mod, config["model"]["pixel_encoder"]["name"])
        enc_cfg = {**config["training"], **config["model"]["pixel_encoder"], **config["data"]}
        pixel_encoder = enc_factory(enc_cfg)

        config["model"]["pixel_encoder"]["patch_size"] = (14 if "dino" in config["model"]["pixel_encoder"]["name"] else 16)
        enc_embed_dims = (pixel_encoder.embed_dims if hasattr(pixel_encoder, "embed_dims")
                          else [getattr(pixel_encoder, "embed_dim") * 2 ** i for i in range(4)])
        config["model"]["pixel_encoder"]["embed_dim"]  = getattr(pixel_encoder, "embed_dim")
        config["model"]["pixel_encoder"]["embed_dims"] = enc_embed_dims
        config["model"]["pixel_encoder"]["depths"]     = pixel_encoder.depths
        config["model"]["pixel_encoder"]["cls_token_embed_dims"] = getattr(
            pixel_encoder, "cls_token_embed_dims", enc_embed_dims
        )

        self.small_encoder = SmallEncoder(config)
        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = Decoder(config)
        self.image_shape = config["data"]["image_shape"]
        self.slices_encoder_range = list(zip([0, *self.pixel_encoder.depths[:-1]], self.pixel_encoder.depths))
        self.stacking_fn = last_stack
        self.shape_constraints = config["data"]["shape_constraints"]
        self.interpolation_mode = "bilinear"

        # RAFT
        self.raft = None
        if config["model"].get("raft_weights", None) is not None:
            from argparse import Namespace
            from velodepth.models.additionals import RAFT

            args = Namespace(
                name="Tartan-C-T-TSKH432x960-M", dataset="TSKH", gpus=[0],
                use_var=True, var_min=0, var_max=10, pretrain="resnet34",
                initial_dim=64, block_dims=[64, 128, 256], radius=4, dim=128,
                num_blocks=2, iters=8, image_size=[432, 960], scale=0, batch_size=32,
                epsilon=1e-08, lr=0.0004, wdecay=1e-05, dropout=0, clip=1.0,
                gamma=0.85, num_steps=120000, restore_ckpt=None, coarse_config=None,
            )
            self.raft = RAFT(args)
            weight = torch.load(config["model"]["raft_weights"], map_location=torch.device("cpu"), weights_only=False)
            info = self.raft.load_state_dict(weight, strict=False)
            print("Loaded RAFT with following weights:", info)

        # Losses
        self.build_losses(config)

    def build_losses(self, config):
        self.losses = {}
        for name, cfg in config["training"]["losses"].items():
            mod = importlib.import_module("velodepth.ops.losses")
            factory = getattr(mod, cfg["name"])
            self.losses[name] = factory.build(cfg)

    def get_params(self, config):
        encoder_lr = config["model"]["pixel_encoder"]["lr"]
        encoder_wd = config["model"]["pixel_encoder"].get("wd", config["training"]["wd"])
        residual_lr = config["model"]["residual_encoder"]["lr"]
        wd = config["training"]["wd"]
        lr = config["training"]["lr"]
        ld = config["training"].get("ld", 1.0)

        if hasattr(self.pixel_encoder, "get_params"):
            enc_p = self.pixel_encoder.get_params(encoder_lr, encoder_wd, ld)[0]
        else:
            enc_p = get_params(self.pixel_encoder, encoder_lr, encoder_wd)[0]

        if hasattr(self.small_encoder, "get_params"):
            small_p = self.small_encoder.get_params(residual_lr, wd, ld)[0]
        else:
            small_p = get_params(self.small_encoder, residual_lr, wd)[0]

        dec_p, _ = get_params(self.pixel_decoder, lr, wd)
        return [*enc_p, *small_p, *dec_p]

    def train(self, mode: bool = True):
        super().train(mode)
        # freeze pixel encoder
        self.pixel_encoder.train(False)
        for p in self.pixel_encoder.parameters(): p.requires_grad = False
        # train decoder + small encoder
        self.pixel_decoder.train(mode)
        self.small_encoder.train(mode)
        # freeze RAFT
        if self.raft is not None:
            for p in self.raft.parameters(): p.requires_grad = False
            self.raft.train(False)

    def step(self):
        self.pixel_decoder.steps += 1

    def parameters_grad(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    @property
    def device(self):
        return next(self.parameters()).device
