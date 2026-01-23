from typing import Any

from velodepth.datasets.sequence_dataset import SequenceDataset


class ScanNet(SequenceDataset):
    min_depth = 0.005
    max_depth = 10.0
    depth_scale = 1000.0
    test_split = "test.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["ScanNetS.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, Any],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["K", "cam2w"],
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

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results


class ScanNetVid(SequenceDataset):
    min_depth = 0.01
    max_depth = 10.0
    depth_scale = 1000.0
    default_fps = 10
    test_split = "val.txt"
    train_split = "train.txt"
    sequences_file = "sequences_.json"
    hdf5_paths = ["ScanNet.hdf5"]
    
    def __init__(
        self,
        image_shape,
        split_file, 
        test_mode,
        crop=None,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["K", "cam2w"],
        **kwargs,
    ):
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
            decode_fields=decode_fields if not test_mode else [*decode_fields, "flow_fwd", "flow_fwd_mask"],
            inplace_fields=inplace_fields,
            **kwargs
        )