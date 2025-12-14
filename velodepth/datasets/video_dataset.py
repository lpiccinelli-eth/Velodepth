import io
import json
import os
from functools import partial
from math import pi, tan
from time import time
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import tables
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
from PIL import Image

from velodepth.datasets.base_dataset import BaseDataset
from velodepth.datasets.utils import DatasetFromList
from velodepth.datasets.utils_decode import (decode_camera, decode_depth,
                                             decode_flow, decode_K,
                                             decode_mask, decode_numpy,
                                             decode_rgb, decode_tensor,
                                             decode_video)
from velodepth.utils.camera import BatchCamera, Pinhole
from velodepth.utils.distributed import is_main_process


class VideoDataset(BaseDataset):
    DECODE_FNS = {
        "image": partial(decode_rgb, name="image"),
        "points": partial(decode_numpy, name="points"),
        "K": partial(decode_K, name="camera"),
        "camera_params": partial(decode_camera, name="camera"),
        "cam2w": partial(decode_tensor, name="cam2w"),
        "depth": partial(decode_depth, name="depth"),
        "flow_fwd": partial(decode_flow, name="flow_fwd"),
        "flow_bwd": partial(decode_flow, name="flow_bwd"),
        "flow_fwd_mask": partial(decode_mask, name="flow_fwd_mask"),
        "flow_bwd_mask": partial(decode_mask, name="flow_bwd_mask"),
        "video": partial(decode_video, name="image"),
    }
    default_fps = 5

    def __init__(
        self,
        image_shape: Tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: Dict[str, Any],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["video"],
        inplace_fields: list[str] = [],
        add_of: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            **kwargs,
        )
        if num_frames < 0:
            num_frames = 80
        self.num_frames = num_frames
        self.original_num_frames = num_frames
        self.decode_fields = decode_fields
        self.inplace_fields = inplace_fields
        self.fps = self.default_fps
        self.fps_range = kwargs.get("fps_range", None)
        if self.fps_range is not None:
            self.fps_range[1] = min(self.default_fps, self.fps_range[1])
        self.add_of = add_of
        self.fov = 75.0 * pi / 180.0  # degree
        self.load_dataset()

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )

        txt_file = np.array(h5file[self.split_file])  # video_paths.txt
        try:
            txt_string = txt_file.flatten()[0].decode("utf-8").strip()
        except:
            txt_string = txt_file.tostring().decode("ascii").strip()
        video_paths = [l.strip() for l in txt_string.splitlines()]
        video_paths = video_paths[10::70000]
        self.dataset = DatasetFromList(video_paths)
        h5file.close()
        self.log_load_dataset()

    def get_single_sequence(self, idx):
        self.num_frames = self.original_num_frames
        # sequence_name = self.dataset[idx]["sequence_name"]
        video_name = self.dataset[idx]
        h5_path = os.path.join(self.data_root, self.hdf5_paths[0])

        results = {}
        results = self.pre_pipeline(results)
        results.update(
            {
                (idx, 0): {
                    k: v[idx] if isinstance(v, list) else v.copy()
                    for k, v in results.items()
                }
                for idx in range(self.num_frames)
            }
        )
        results["sequence_fields"] = [(i, 0) for i in range(self.num_frames)]

        with tables.File(
            h5_path,
            mode="r",
            libver="latest",
            swmr=True,
        ) as h5file_chunk:
            for decode_field in self.decode_fields:
                results = self.DECODE_FNS[decode_field](
                    results,
                    h5file_chunk,
                    video_name,
                    num_frames=self.num_frames,
                    skip_frame_range=[
                        self.default_fps // self.fps_range[1],
                        self.default_fps // self.fps_range[0],
                    ],
                )

            results["filename"] = video_name

        if not self.test_mode:
            results = self.pre_augment(results)
        results = self.generate_dummy_camera(results)
        if self.add_of:
            results = self.generate_of(results)
        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)

        # generate dummy info (depth, mask and cameras)
        results = self.generate_dummy_gt(results)
        results = self.postprocess(results)
        return results

    def preprocess(self, results):
        self.resizer.ctx = None
        self.resizer.random_shift_x = None
        self.resizer.random_shift_y = None
        results = self.replicate(results)
        for i, seq in enumerate(results["sequence_fields"]):
            results[seq] = self.resizer(results[seq])

            for key in results[seq].get("image_fields", ["image"]):
                results[seq][key] = results[seq][key].to(torch.float32) / 255

        # update fields common in sequence
        for key in ["image_fields", "gt_fields", "mask_fields", "camera_fields"]:
            if key in results[(0, 0)]:
                results[key] = results[(0, 0)][key]

        results = self.pack_batch(results)
        return results

    def postprocess(self, results):
        # # normalize after because color aug requires [0,255]?
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.normalize(results[key], **self.normalization_stats)
        # results = self.filler(results)
        results = self.unpack_batch(results)
        results = self.masker(results)
        results = self.collecter(results)
        return results

    def generate_dummy_camera(self, results):
        T = len(results["sequence_fields"])
        H, W = results[(0, 0)]["image"].shape[-2:]
        device = results[(0, 0)]["image"].device

        fy = fx = max(W, H) / 2 / tan(self.fov / 2)
        cx, cy = W / 2 - 0.5, H / 2 - 0.5
        dummy_params = torch.tensor([fx, fy, cx, cy], device=device).unsqueeze(0)
        camera = BatchCamera.from_camera(Pinhole(params=dummy_params))
        for i in range(T):
            results[(i, 0)]["camera"] = camera
            results[(i, 0)]["cam2w"] = torch.eye(4, device=device).unsqueeze(0)
            results[(i, 0)]["camera_fields"].add("cam2w")
            results[(i, 0)]["camera_fields"].add("camera")
        return results

    def generate_dummy_gt(self, results):
        T = len(results["sequence_fields"])
        H, W = results["image"].shape[-2:]
        device = results["image"].device

        results["gt_fields"].add("depth")
        results["depth"] = torch.zeros(T, 1, H, W, device=device)
        return results

    def __getitem__(self, idx):
        load_error = False
        try:
            if isinstance(idx, (list, tuple)):
                results = [self.get_single_sequence(i) for i in idx]
            else:
                results = self.get_single_sequence(idx)
        except Exception as e:
            print(f"Error loading video {idx} for {self.__class__.__name__}: {e}")
            load_error = True
        if load_error:
            idx = np.random.randint(0, len(self.dataset))
            results = self[idx]
        return results

    def log_load_dataset(self):
        if is_main_process():
            info = f"Loaded {self.__class__.__name__} with {len(self)} videos."
            print(info)

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [2] * self.num_frames * self.num_copies
        results["valid_camera"] = [False] * self.num_frames * self.num_copies
        results["valid_pose"] = [False] * self.num_frames * self.num_copies
        return results
