from velodepth.datasets.sequence_dataset import SequenceDataset


class TUM(SequenceDataset):
    min_depth = 0.001
    max_depth = 10.0
    depth_scale = 1000.0
    default_fps = 30
    test_split = "val.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["TUM_long.hdf5"]
    
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
        inplace_fields: list[str] = ["camera_params", "cam2w"],
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

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results