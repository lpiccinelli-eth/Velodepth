import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from velodepth.utils.distributed import (barrier, get_world_size,
                                         is_main_process)
from velodepth.utils.misc import remove_leading_dim, remove_padding, ssi_helper
from velodepth.utils.visualization import colorize, image_grid


def stack_mixedshape_numpy(tensor_list, dim=0):
    max_rows = max(tensor.shape[0] for tensor in tensor_list)
    max_columns = max(tensor.shape[1] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        rows, columns, *_ = tensor.shape
        pad_rows = max_rows - rows
        pad_columns = max_columns - columns

        padded_tensor = np.pad(
            tensor, ((0, pad_rows), (0, pad_columns), (0, 0)), mode="constant"
        )
        padded_tensors.append(padded_tensor)

    return np.stack(padded_tensors, axis=dim)


def original_image(batch):
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["image"].device)[
        ..., [0, 2, 1, 3]
    ]  # lrtb

    T, _, H, W = batch["data"]["depth"].shape
    batch["data"]["image"] = F.interpolate(
        batch["data"]["image"],
        (H + paddings[0][2] + paddings[0][3], W + paddings[0][1] + paddings[0][2]),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    batch["data"]["image"] = remove_padding(
        batch["data"]["image"], paddings.repeat(T, 1)
    )
    return batch


def original_image_inv(batch, preds=None):
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("padding_size", [[0] * 4])
    ]
    T, _, H, W = batch["data"]["depth"].shape
    batch["data"]["image"] = remove_padding(batch["data"]["image"], paddings * T)
    batch["data"]["image"] = F.interpolate(
        batch["data"]["image"],
        (H, W),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    if preds is not None:
        for key in ["depth"]:
            if key in preds:
                preds[key] = remove_padding(preds[key], paddings * T)
                preds[key] = F.interpolate(
                    preds[key],
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

    return batch, preds


def aggregate_metrics(metrics_all, exclude_fn=lambda name: False):
    aggregate_name = "".join(
        [name_ds[:3] for name_ds in metrics_all.keys() if not exclude_fn(name_ds)]
    )
    metrics_aggregate = defaultdict(list)
    for name_ds, metrics in metrics_all.items():
        if exclude_fn(name_ds):
            continue
        for metrics_name, metrics_value in metrics.items():
            metrics_aggregate[metrics_name].append(metrics_value)
    return {
        **{aggregate_name: {k: sum(v) / len(v) for k, v in metrics_aggregate.items()}},
        **metrics_all,
    }


GROUPS = {
    "SFoV": ["KITTI", "NYUv2Depth", "DiodeIndoor", "ETH3D", "IBims"],
    "SFoVDi": ["DiodeIndoor_F", "ETH3D_F", "IBims_F"],
    "LFoV": ["ADT", "KITTI360", "ScanNetpp_F"],
}


def aggregate_metrics_camera(metrics_all):
    available_groups = {
        k: v for k, v in GROUPS.items() if any([name in metrics_all for name in v])
    }
    for group_name, group_datasets in available_groups.items():
        metrics_aggregate = defaultdict(list)
        for dataset_name in group_datasets:
            if dataset_name not in metrics_all:
                print(
                    f"Dataset {dataset_name} not used for aggregation of {group_name}"
                )
                continue
            for metrics_name, metrics_value in metrics_all[dataset_name].items():
                metrics_aggregate[metrics_name].append(metrics_value)
        metrics_all[group_name] = {
            k: sum(v) / len(v) for k, v in metrics_aggregate.items()
        }
    return metrics_all


def log_metrics(metrics_all, step):
    for name_ds, metrics in metrics_all.items():
        for metrics_name, metrics_value in metrics.items():
            try:
                wandb.log(
                    {f"Metrics/{name_ds}/{metrics_name}": metrics_value}, step=step
                )
            except:
                print(f"Metrics/{name_ds}/{metrics_name} {round(metrics_value, 4)}")


def log_artifacts(artifacts_all, step, run_id):
    for ds_name, artifacts in artifacts_all.items():
        rgbs, gts = artifacts["rgbs"], artifacts["gts"]
        logging_imgs = [
            *rgbs,
            *gts,
            *[
                x
                for k, v in artifacts.items()
                if ("rgbs" not in k and "gts" not in k)
                for x in v
            ],
        ]
        artifacts_grid = image_grid(logging_imgs, len(artifacts), len(rgbs))
        try:
            wandb.log({f"{ds_name}_test": [wandb.Image(artifacts_grid)]}, step=step)
        except:
            print(f"Error while saving artifacts at step {step}")


def show(vals, dataset, ssi_depth=False):
    output_artifacts, additionals = {}, {}
    predictions, gts, errors, images = [], [], [], []
    for v in vals:
        image = v["image"][0].unsqueeze(0)
        gt = v["depth"][0].unsqueeze(0)
        prediction = v["depth_pred"][0].unsqueeze(0)
        H, W = gt.shape[-2:]
        aspect_ratio = H / W
        new_W = int((300_000 / aspect_ratio) ** 0.5)
        new_H = int(aspect_ratio * new_W)
        gt = F.interpolate(gt, (new_H, new_W), mode="nearest-exact")

        # Format predictions and errors for every metrics used
        prediction = F.interpolate(
            prediction,
            gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        error = torch.zeros_like(prediction)
        error[gt > dataset.min_depth] = (
            4
            * dataset.max_depth
            * torch.abs(gt - prediction)[gt > dataset.min_depth]
            / gt[gt > dataset.min_depth]
        )
        if ssi_depth:
            scale, shift = ssi_helper(gt[gt > 0], prediction[gt > 0])
            prediction = (prediction * scale + shift).clip(0.0, dataset.max_depth)
        prediction = colorize(
            prediction.squeeze().cpu().detach().numpy(),
            vmin=dataset.min_depth,
            vmax=dataset.max_depth,
            cmap="magma_r",
        )
        error = error.clip(0.0, dataset.max_depth).cpu().detach().numpy()
        error = colorize(error.squeeze(), vmin=0.001, vmax=1.0, cmap="coolwarm")
        errors.append(error)
        predictions.append(prediction)

        image = F.interpolate(
            image, gt.shape[-2:], mode="bilinear", align_corners=False, antialias=True
        )
        image = image.cpu().detach() * dataset.normalization_stats["std"].view(
            1, -1, 1, 1
        ) + dataset.normalization_stats["mean"].view(1, -1, 1, 1)
        image = (
            (255 * image)
            .clip(0.0, 255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
            .squeeze()
        )
        gt = gt.clip(0.0, dataset.max_depth).cpu().detach().numpy()
        gt = colorize(
            gt.squeeze(), vmin=dataset.min_depth, vmax=dataset.max_depth, cmap="magma_r"
        )
        gts.append(gt)
        images.append(image)

        for name, additional in v.get("infos", {}).items():
            if name not in additionals:
                additionals[name] = []
            if additional[0].shape[0] == 3:
                val = (
                    (127.5 * (additional[0] + 1))
                    .clip(0, 255)
                    .to(torch.uint8)
                    .cpu()
                    .detach()
                    .permute(1, 2, 0)
                    .numpy()
                )
            else:
                val = colorize(
                    additional[0].cpu().detach().squeeze().numpy(),
                    0.0,
                    dataset.max_depth,
                )
            additionals[name].append(val)

    output_artifacts.update(
        {
            f"predictions": stack_mixedshape_numpy(predictions),
            f"errors": stack_mixedshape_numpy(errors),
            "rgbs": stack_mixedshape_numpy(images),
            "gts": stack_mixedshape_numpy(gts),
            **{k: stack_mixedshape_numpy(v) for k, v in additionals.items()},
        }
    )
    return output_artifacts


METRIC_B = "F1"
INVERT = True
SSI_VISUALIZATION = True


def validate(
    model,
    test_loaders: Dict[str, DataLoader],
    step,
    run_id,
    context,
    idxs=(1, 100, 150, 1000),
):

    metrics_all, predictions_select = {}, {}
    world_size = get_world_size()
    for name_ds, test_loader in test_loaders.items():
        idxs = [idx % len(test_loader.dataset) for idx in idxs]
        ds_show = []
        for i, batch in enumerate(test_loader):
            with context:
                batch["data"] = {
                    k: v.to(model.device) for k, v in batch["data"].items()
                }
                preds, _ = model(batch["data"], batch["img_metas"])

            if batch["data"]["image"].ndim == 5:
                batch["data"] = remove_leading_dim(batch["data"])
            if preds["depth"].ndim == 5:
                preds = remove_leading_dim(preds)
            batch = original_image(batch)
            test_loader.dataset.accumulate_metrics(
                inputs=batch["data"],
                preds=preds,
                keyframe_idx=batch["img_metas"][0].get("keyframe_idx"),
                metrics=["depth", "points"]

            )

            # for prediction images logging
            if i * world_size in idxs:
                ii = (len(preds["depth"]) + 1) // 2 - 1
                slice_ = slice(ii, ii + 1)
                batch["data"] = {k: v[slice_] for k, v in batch["data"].items()}
                preds["depth"] = preds["depth"][slice_]
                ds_show.append({**batch["data"], **{"depth_pred": preds["depth"]}})

        barrier()

        metrics_all[name_ds] = test_loader.dataset.get_evaluation()
        predictions_select[name_ds] = show(
            ds_show, test_loader.dataset, ssi_depth=SSI_VISUALIZATION
        )

    barrier()
    if is_main_process():
        log_artifacts(artifacts_all=predictions_select, step=step, run_id=run_id)
        metrics_all = aggregate_metrics(
            metrics_all, exclude_fn=lambda name: "mono" in name
        )
        metrics_all = aggregate_metrics_camera(metrics_all)
        log_metrics(metrics_all=metrics_all, step=step)
    return metrics_all
