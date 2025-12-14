dependencies = ["torch", "huggingface_hub"]

import os
import json

import torch
import huggingface_hub

from velodepth.models import VeloDepth as VeloDepth_


def VeloDepth(pretrained=True):
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", f"velodepth.json")) as f:
        config = json.load(f)
    
    model = VeloDepth_(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id=f"lpiccinelli/velodepth", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"VeloDepth is loaded with:")
        print(f"\t missing keys: {info.missing_keys}")
        print(f"\t additional keys: {info.unexpected_keys}")

    return model

