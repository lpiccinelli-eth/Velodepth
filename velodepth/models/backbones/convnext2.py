import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


def get_num_layer_for_convnext_single(var_name, depths):
    """
    Each layer is assigned distinctive layer ids
    """
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        layer_id = sum(depths[:stage_id]) + 1
        return layer_id
    
    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        layer_id = sum(depths[:stage_id]) + block_id + 1
        return layer_id
    
    else:
        return sum(depths) + 1


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1


def get_parameter_groups(model, lr, wd=1e-5, ld=1.0, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}
    skip = {}
    if skip_list is not None:
        skip = skip_list
    elif hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    num_layers = 12#sum(model.depths)
    layer_scale = list(ld ** (num_layers + 1 - i) for i in range(num_layers + 2))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip or name.endswith(".gamma") or name.endswith(".beta"):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = wd
        
        # layer_id = get_num_layer_for_convnext_single(name, model.depths)
        layer_id = get_num_layer_for_convnext(name)
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_group_names:
            scale = layer_scale[layer_id]
            cur_lr = lr * scale

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr_base": cur_lr
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr_base": cur_lr
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values()), [v["lr_base"] for k, v in parameter_group_vars.items()]
            

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., mult=4, use_checkpoint=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mult * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(mult * dim)
        self.pwconv2 = nn.Linear(mult * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
    

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3], 
                 dims=96, 
                 drop_path_rate=0., 
                 output_idx=[],
                 use_checkpoint=False
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.depths = output_idx
        self.embed_dims = [int(dim) for i, dim in enumerate(dims)]
        self.embed_dim = dims[0]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.out_norms = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j], use_checkpoint=use_checkpoint) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(len(self.stages)):
            if i == 0:
                x = self.downsample_layers[i][0](x)
                x = self.downsample_layers[i][1](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.downsample_layers[i][0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                x = self.downsample_layers[i][1](x)
            x = self.stages[i](x)
            outs.append(x.permute(0, 2, 3, 1))
        cls_tokens = [x.mean(dim=(1, 2)).unsqueeze(1).contiguous() for x in outs]
        return outs, cls_tokens

    def forward_part(self, x, slice_):
        outs = []
        for i in slice_:
            if i == 0:
                x = self.downsample_layers[i][0](x)
                x = self.downsample_layers[i][1](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.downsample_layers[i][0](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                x = self.downsample_layers[i][1](x)
            x = self.stages[i](x)
            outs.append(x.permute(0, 2, 3, 1))
        # cls_tokens = [x.mean(dim=(1, 2)).unsqueeze(1).contiguous() for x in outs]
        return outs, []

    def get_params(self, lr, wd, ld, *args, **kwargs):
        encoder_p, encoder_lr = get_parameter_groups(self, lr, wd, ld)
        return encoder_p, encoder_lr

    def freeze(self) -> None:
        for module in self.modules():
            module.eval()
        for parameters in self.parameters():
                parameters.requires_grad = False

    def clean(self, i):
        for j in range(i, len(self.stages)):
            self.stages[j] = nn.Identity()
            self.downsample_layers[j] = nn.Identity()
        return self

    @classmethod
    def build(cls, config):
        obj = globals()[config["model"]["encoder"]["name"]](config)
        return obj
