from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image

from velodepth.utils.chamfer_distance import ChamferDistance
from velodepth.utils.constants import DEPTH_BINS

chamfer_cls = ChamferDistance()


def kl_div(gt, pred, eps: float = 1e-6):
    depth_bins = DEPTH_BINS.to(gt.device)
    gt, pred = torch.bucketize(
        gt, boundaries=depth_bins, out_int32=True
    ), torch.bucketize(pred, boundaries=depth_bins, out_int32=True)
    gt = torch.bincount(gt, minlength=len(depth_bins) + 1)
    pred = torch.bincount(pred, minlength=len(depth_bins) + 1)
    gt = gt / gt.sum()
    pred = pred / pred.sum()
    return torch.sum(gt * (torch.log(gt + eps) - torch.log(pred + eps)))


def chamfer_dist(tensor1, tensor2):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    return (torch.sqrt(dist1) + torch.sqrt(dist2)) / 2


def auc(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    auc_value = torch.trapz(
        torch.tensor(precisions, device=tensor1.device),
        torch.tensor(recalls, device=tensor1.device),
    )
    return auc_value


def delta(tensor1, tensor2, exponent):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25**exponent).to(torch.float32).mean()


def rho(tensor1, tensor2):
    min_deg = 0.5
    tensor1_norm = tensor1 / torch.norm(tensor1, dim=-1, p=2, keepdim=True).clip(
        min=1e-6
    )
    tensor2_norm = tensor2 / torch.norm(tensor2, dim=-1, p=2, keepdim=True).clip(
        min=1e-6
    )
    max_polar_angle = torch.arccos(tensor1_norm[..., 2]).max() * 180.0 / torch.pi

    if max_polar_angle < 100.0:
        threshold = 15.0
    elif max_polar_angle < 190.0:
        threshold = 20.0
    else:
        threshold = 30.0

    acos_clip = 1 - 1e-6
    # inner prod of norm vector -> cosine
    angular_error = (
        torch.arccos(
            (tensor1_norm * tensor2_norm)
            .sum(dim=-1)
            .clip(min=-acos_clip, max=acos_clip)
        )
        * 180.0
        / torch.pi
    )
    thresholds = torch.linspace(min_deg, threshold, steps=100, device=tensor1.device)
    y_values = [
        (angular_error.abs() <= th).to(torch.float32).mean() for th in thresholds
    ]
    auc_value = torch.trapz(
        torch.tensor(y_values, device=tensor1.device), thresholds
    ) / (threshold - min_deg)
    return auc_value


def tau(tensor1, tensor2, perc):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < (1.0 + perc)).to(torch.float32).mean()


@torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
def ssi(tensor1, tensor2, qtl=0.05):
    stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
    error = (tensor1 - tensor2).abs()
    mask = error < torch.quantile(error, 1 - qtl)
    tensor1_mask = tensor1.to(torch.float32)[mask]
    tensor2_mask = tensor2.to(torch.float32)[mask]
    stability_mat = 1e-4 * torch.eye(2, device=tensor1.device)
    tensor2_one = torch.stack([tensor2_mask, torch.ones_like(tensor2_mask)], dim=1)
    A = torch.matmul(tensor2_one.T, tensor2_one) + stability_mat
    det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    A_inv = (1.0 / det_A) * torch.tensor(
        [[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], device=tensor1.device
    )
    b = tensor2_one.T @ tensor1_mask.unsqueeze(1)
    scale_shift = A_inv @ b
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return tensor2 * scale + shift


def si(tensor1, tensor2):
    return tensor2 * torch.median(tensor1) / torch.median(tensor2)


def arel(tensor1, tensor2):
    tensor2 = tensor2 * torch.median(tensor1) / torch.median(tensor2)
    return (torch.abs(tensor1 - tensor2) / tensor1).mean()


def d_auc(tensor1, tensor2):
    exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
    deltas = [delta(tensor1, tensor2, exponent) for exponent in exponents]
    return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0


def f1_score(tensor1, tensor2, thresholds):
    x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
    y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
    dist1, dist2, idx1, idx2 = chamfer_cls(
        tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
    )
    # compute precision recall
    precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
    recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
    precisions = torch.tensor(precisions, device=tensor1.device)
    recalls = torch.tensor(recalls, device=tensor1.device)
    f1_thresholds = 2 * precisions * recalls / (precisions + recalls)
    f1_thresholds = torch.where(
        torch.isnan(f1_thresholds), torch.zeros_like(f1_thresholds), f1_thresholds
    )
    f1_value = torch.trapz(f1_thresholds) / len(thresholds)
    return f1_value


def f1_score_si(tensor1, tensor2, thresholds):
    tensor2 = (
        tensor2
        * torch.median(tensor1.norm(dim=-1))
        / torch.median(tensor2.norm(dim=-1))
    )
    f1_value = f1_score(tensor1, tensor2, thresholds)
    return f1_value


DICT_METRICS = {
    "d1": partial(delta, exponent=1.0),
    "d2": partial(delta, exponent=2.0),
    "d3": partial(delta, exponent=3.0),
    "rmse": lambda gt, pred: torch.sqrt(((gt - pred) ** 2).mean()),
    "rmselog": lambda gt, pred: torch.sqrt(
        ((torch.log(gt) - torch.log(pred)) ** 2).mean()
    ),
    "arel": lambda gt, pred: (torch.abs(gt - pred) / gt).mean(),
    "sqrel": lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10": lambda gt, pred: torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
    "silog": lambda gt, pred: 100 * torch.std(torch.log(pred) - torch.log(gt)).mean(),
    "medianlog": lambda gt, pred: 100
    * (torch.log(pred) - torch.log(gt)).median().abs(),
    "d_auc": d_auc,
    "tau": partial(tau, perc=0.03),
}


DICT_METRICS_3D = {
    "MSE_3d": lambda gt, pred, thresholds: torch.norm(gt - pred, dim=0, p=2),
    "arel_3d": lambda gt, pred, thresholds: torch.norm(gt - pred, dim=0, p=2)
    / torch.norm(gt, dim=0, p=2),
    "tau_3d": lambda gt, pred, thresholds: (
        (torch.norm(pred, dim=0, p=2) / torch.norm(gt, dim=0, p=2)).log().abs().exp()
        < 1.25
    )
    .float()
    .mean(),
    "chamfer": lambda gt, pred, thresholds: chamfer_dist(
        gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)
    ),
    "F1": lambda gt, pred, thresholds: f1_score(
        gt.unsqueeze(0).permute(0, 2, 1),
        pred.unsqueeze(0).permute(0, 2, 1),
        thresholds=thresholds,
    ),
    "F1_si": lambda gt, pred, thresholds: f1_score_si(
        gt.unsqueeze(0).permute(0, 2, 1),
        pred.unsqueeze(0).permute(0, 2, 1),
        thresholds=thresholds,
    ),
    "rays": lambda gt, pred, thresholds: rho(
        gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)
    ),
}


DICT_METRICS_FLOW = {
    "epe": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)),
    "epe1": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 1,
    "epe3": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 3,
    "epe5": lambda gt, pred: torch.sqrt(torch.square(gt - pred).sum(dim=0)) < 5,
}


DICT_METRICS_D = {
    "a1": lambda gt, pred: (torch.maximum((gt / pred), (pred / gt)) > 1.25**1.0).to(
        torch.float32
    ),
    "abs_rel": lambda gt, pred: (torch.abs(gt - pred) / gt),
}


def eval_depth(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, max_depth=None
):
    summary_metrics = defaultdict(list)
    # preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if max_depth is not None:
            mask = mask & (gt <= max_depth)
        for name, fn in DICT_METRICS.items():
            if name in ["tau", "d1", "arel"]:
                for rescale_fn in ["ssi", "si"]:
                    summary_metrics[f"{name}_{rescale_fn}"].append(
                        fn(gt[mask], eval(rescale_fn)(gt[mask], pred[mask]))
                    )
            summary_metrics[name].append(fn(gt[mask], pred[mask]).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def eval_3d(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, thresholds=None
):
    summary_metrics = defaultdict(list)
    MAX_PIXELS = 75_000  # 300_000
    ratio = min(1.0, (MAX_PIXELS / masks[0].sum()) ** 0.5)
    h_max, w_max = int(gts.shape[-2] * ratio), int(gts.shape[-1] * ratio)
    gts = F.interpolate(gts, size=(h_max, w_max), mode="nearest-exact")
    preds = F.interpolate(preds, size=(h_max, w_max), mode="nearest-exact")
    masks = F.interpolate(
        masks.float(), size=(h_max, w_max), mode="nearest-exact"
    ).bool()

    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if not torch.any(mask):
            continue
        for name, fn in DICT_METRICS_3D.items():
            summary_metrics[name].append(
                fn(gt[:, mask.squeeze()], pred[:, mask.squeeze()], thresholds).mean()
            )
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def compute_aucs(gt, pred, mask, uncertainties, steps=50, metrics=["abs_rel"]):
    dict_ = {}
    x_axis = torch.linspace(0, 1, steps=steps + 1, device=gt.device)
    quantiles = torch.linspace(0, 1 - 1 / steps, steps=steps, device=gt.device)
    zer = torch.tensor(0.0, device=gt.device)
    # revert order (high uncertainty first)
    uncertainties = -uncertainties[mask]
    gt = gt[mask]
    pred = pred[mask]
    true_uncert = {metric: -DICT_METRICS_D[metric](gt, pred) for metric in metrics}
    # get percentiles for sampling and corresponding subsets
    thresholds = torch.quantile(uncertainties, quantiles)
    subs = [(uncertainties >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    for metric in metrics:
        opt_thresholds = torch.quantile(true_uncert[metric], quantiles)
        opt_subs = [(true_uncert[metric] >= t) for t in opt_thresholds]
        sparse_curve = torch.stack(
            [DICT_METRICS[metric](gt[sub], pred[sub]) for sub in subs] + [zer], dim=0
        )
        opt_curve = torch.stack(
            [DICT_METRICS[metric](gt[sub], pred[sub]) for sub in opt_subs] + [zer],
            dim=0,
        )
        rnd_curve = DICT_METRICS[metric](gt, pred)

        dict_[f"AUSE_{metric}"] = torch.trapz(sparse_curve - opt_curve, x=x_axis)
        dict_[f"AURG_{metric}"] = rnd_curve - torch.trapz(sparse_curve, x=x_axis)

    return dict_


def eval_depth_uncertainties(
    gts: torch.Tensor,
    preds: torch.Tensor,
    uncertainties: torch.Tensor,
    masks: torch.Tensor,
    max_depth=None,
):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask, uncertainty) in enumerate(
        zip(gts, preds, masks, uncertainties)
    ):
        if max_depth is not None:
            mask = torch.logical_and(mask, gt < max_depth)
        for name, fn in DICT_METRICS.items():
            summary_metrics[name].append(fn(gt[mask], pred[mask]))
        for name, val in compute_aucs(gt, pred, mask, uncertainty).items():
            summary_metrics[name].append(val)
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


def lazy_eval_depth(
    gt_fns, pred_fns, min_depth=1e-2, max_depth=None, depth_scale=256.0
):
    summary_metrics = defaultdict(list)
    for i, (gt_fn, pred_fn) in enumerate(zip(gt_fns, pred_fns)):
        gt = TF.pil_to_tensor(Image.open(gt_fn)).to(torch.float32) / depth_scale
        pred = TF.pil_to_tensor(Image.open(pred_fn)).to(torch.float32) / depth_scale
        mask = gt > min_depth
        if max_depth is not None:
            mask_2 = gt < max_depth
            mask = torch.logical_and(mask, mask_2)
        for name, fn in DICT_METRICS.items():
            summary_metrics[name].append(fn(gt[mask], pred[mask]))

    return {name: torch.mean(vals).item() for name, vals in summary_metrics.items()}
