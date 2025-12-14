import os

import h5py
import numpy as np

from velodepth.datasets.utils import DatasetFromList
from velodepth.datasets.video_dataset import VideoDataset


class UCF101(VideoDataset):
    default_fps = 25
    test_split = "trainlist01.txt"
    train_split = "trainlist01.txt"
    hdf5_paths = ["ucf101.hdf5"]
    depth_scale = 1.0

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db,
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["video"],
        inplace_fields: list[str] = [],
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
            num_frames=num_frames,
            decode_fields=decode_fields,
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tobytes().decode("ascii").strip()
        video_paths = [l.split(" ")[0] for l in txt_string.splitlines()]
        self.dataset = DatasetFromList(video_paths)
        h5file.close()
        self.log_load_dataset()
