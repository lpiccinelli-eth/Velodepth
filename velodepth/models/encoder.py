from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import _cfg

from velodepth.models.backbones import (ConvNeXt, ConvNeXtV2,
                                        SwinTransformerV2, _make_dinov2_model)


def swin2_tiny(
    config,
    pretrained=None,
    *args,
    **kwargs,
):
    model = SwinTransformerV2(
        img_size=config["image_shape"],
        patch_size=4,
        window_size=config.get("window_size", 16),
        embed_dim=96,
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        pretrained_window_sizes=[12, 12, 12, 6],
        output_idx=config.get("output_idx", [2, 4, 10, 12]),
        use_shift=config.get("use_shift", True),
        use_checkpoint=config.get("use_checkpoint", False),
        frozen_stages=-1,
    )
    model.default_cfg = _cfg()
    return model


def swin2_base(
    config,
    pretrained=None,
    *args,
    **kwargs,
):
    model = SwinTransformerV2(
        img_size=config["image_shape"],
        patch_size=4,
        window_size=config.get("window_size", 12),
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        pretrained=pretrained,
        pretrained_window_sizes=[12, 12, 12, 6],
        use_shift=config.get("use_shift", True),
        use_checkpoint=config["use_checkpoint"],
        frozen_stages=-1,
    )
    model.default_cfg = _cfg()
    return model


def swin2_large(
    config,
    pretrained=None,
    *args,
    **kwargs,
):
    model = SwinTransformerV2(
        img_size=config["image_shape"],
        patch_size=4,
        window_size=config.get("window_size", 12),
        embed_dim=192,
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        pretrained=pretrained,
        pretrained_window_sizes=[12, 12, 12, 6],
        use_shift=config.get("use_shift", True),
        use_checkpoint=config["use_checkpoint"],
        frozen_stages=-1,
    )
    model.default_cfg = _cfg()
    return model


def convnextv2_nano(config, **kwargs):
    model = ConvNeXtV2(
        depths=[2, 2, 8, 2],
        dims=[80, 160, 320, 640],
        output_idx=config.get("output_idx", [2, 4, 12, 14]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print("convnextv2_nano loaded:", info)
    return model


def convnextv2_tiny(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        output_idx=config.get("output_idx", [3, 6, 15, 18]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print("convnextv2_tiny loaded:", info)
    return model


def convnextv2_base(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config["use_checkpoint"],
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config["use_checkpoint"],
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config["use_checkpoint"],
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    from velodepth.models.backbones.convnext import (HF_URL,
                                                     checkpoint_filter_fn)

    disable_progress_bars()
    repo_id, filename = HF_URL["convnext_large"]
    state_dict = torch.load(hf_hub_download(repo_id=repo_id, filename=filename))
    state_dict = checkpoint_filter_fn(state_dict, model)
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitg14(config, pretrained: str = "", **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [10, 20, 30, 40]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit
