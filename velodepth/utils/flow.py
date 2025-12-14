import torch
import torch.nn.functional as F

from .camera import invert_pinhole, project_pinhole, unproject_pinhole
from .coordinate import coords_grid, normalize_coords
from .pose import apply_pose_transformation


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32)
def bilinear_sample(img, sample_coords, mode="bilinear", padding_mode="border"):
    if sample_coords.shape[1] == 2:
        sample_coords = sample_coords.permute(0, 2, 3, 1)
    img = F.grid_sample(
        img,
        sample_coords,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False if mode == "bilinear" else None,
    )
    mask = torch.logical_or(sample_coords >= -1, sample_coords <= 1).permute(
        0, 3, 1, 2
    )  # [B, 2, H, W]
    return img, mask[:, :1] & mask[:, 1:]


def flow_warp(feature, flow, mode="bilinear", padding_mode="zeros"):
    dtype = feature.dtype
    feature_shape = feature.shape
    flow_shape = flow.shape
    leading_ndim = max(0, (4 - len(feature_shape)))
    feature = feature.reshape((1,) * leading_ndim + feature_shape)
    flow = flow.reshape((1,) * leading_ndim + flow_shape)
    b, c, h, w = feature.shape
    assert flow.shape[1] == 2
    grid = coords_grid(b, h, w).to(flow.device)
    grid = grid - 0.5  # compensate for grid as central point of pixel?
    grid = normalize_coords(grid, h, w) + flow  # [B, 2, H, W]
    warped_feature, mask = bilinear_sample(
        feature.float(), grid, mode=mode, padding_mode=padding_mode
    )
    return (
        warped_feature.reshape(feature_shape).to(dtype),
        mask,
    )  # FIXME, should be featue dim with 1 set to 1


def forward_backward_consistency_check(fwd_flow_px, bwd_flow_px, alpha=0.01, beta=0.0):
    assert fwd_flow_px.ndim == 4 and bwd_flow_px.ndim == 4
    assert fwd_flow_px.shape[1] == 2 and bwd_flow_px.shape[1] == 2

    # here we consider flow in pixel, but flow_warp expects in [-1,1]
    H, W = fwd_flow_px.shape[-2:]
    correction_factor = torch.tensor([W - 1, H - 1], device=fwd_flow_px.device).view(
        1, 2, 1, 1
    )
    bwd_flow = bwd_flow_px / correction_factor
    fwd_flow = fwd_flow_px / correction_factor

    warped_bwd_flow_px, _ = flow_warp(bwd_flow_px, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow_px, _ = flow_warp(fwd_flow_px, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(
        fwd_flow_px + warped_bwd_flow_px, dim=1, keepdim=True
    )  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow_px + warped_fwd_flow_px, dim=1, keepdim=True)

    return diff_fwd, diff_bwd


def flow_from_depth(depth, intrinsics, pose_21):
    # depth: [B, 1, H, W], intrinsics: [B, 3, 3], extrinsics: [B, 4, 4]
    assert depth.ndim == 4 and intrinsics.ndim == 3 and pose_21.ndim == 3
    assert depth.shape[1] == 1 and intrinsics.shape[1] == 3 and intrinsics.shape[2] == 3
    assert pose_21.shape[1] == 4 and pose_21.shape[2] == 4
    b, _, h, w = depth.shape
    device = depth.device
    grid = coords_grid(b, h, w, homogeneous=True, device=device)  # [B, 3, H, W]
    K_inv = invert_pinhole(intrinsics)
    pcd = unproject_pinhole(depth, K_inv)  # [B, 3, H, W]
    # R_21, t_21 = pose_to_Rt(pose_21)
    pcd_21 = apply_pose_transformation(pcd, pose_21)  # [B, 3, H, W]
    # depth_21 = pcd_21[:, 2:]  # [B, H, W]
    projected_coords = project_pinhole(pcd_21, intrinsics)  # [B, 2, H, W]
    flow = projected_coords - grid[:, :2]  # [B, 2, H, W]
    flow = flow / torch.tensor([w - 1, h - 1], device=device).view(1, 2, 1, 1)
    return flow
