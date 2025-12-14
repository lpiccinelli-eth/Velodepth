from .video_dataset import VideoDataset


class SegmentAnythingV2(VideoDataset):
    default_fps = 30
    test_split = "video_paths.txt"
    train_split = "video_paths.txt"
    hdf5_paths = ["segment_anything.hdf5"]
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
