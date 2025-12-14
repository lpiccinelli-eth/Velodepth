import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckRG(nn.Module):
    def __init__(
        self,
        dim,
        current_dim,
        gate_dim,
        kernel_size: int | None = None,
        padding_mode: str = "zeros",
        dilation: int = 1,
        use_norm: bool = False,
        num_blocks: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_small = nn.Sequential(
            nn.LayerNorm(current_dim), nn.Linear(current_dim, current_dim)
        )
        self.in_flow = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, gate_dim),
            nn.LeakyReLU(),
            nn.Linear(gate_dim, dim // 4),
        )
        kernel_size = kernel_size if kernel_size is not None else 3
        self.num_blocks = num_blocks
        self.activation = nn.LeakyReLU()

        self.conv1 = nn.ModuleList([])
        self.conv2 = nn.ModuleList([])
        self.conv3 = nn.ModuleList([])
        self.norm1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        for i in range(num_blocks):
            self.conv1.append(
                nn.Conv2d(
                    dim + current_dim if i == 0 else 2 * dim,
                    dim // 4,
                    kernel_size=1,
                    padding=0,
                )
            )
            self.conv2.append(
                nn.Conv2d(
                    dim // 4,
                    dim // 4,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                    padding_mode=padding_mode,
                )
            )
            self.conv3.append(nn.Conv2d(dim // 4, dim, kernel_size=1, padding=0))
            self.norm1.append(
                nn.GroupNorm(dim // 16, dim // 4) if use_norm else nn.Identity()
            )
            self.norm2.append(
                nn.GroupNorm(dim // 16, dim // 4) if use_norm else nn.Identity()
            )

    def forward(
        self,
        upstream,
        to_be_compensated,
        original_features,
        flow_features,
        *args,
        **kwargs,
    ):
        B, C, H, W = to_be_compensated.shape
        upstream = self.in_small(upstream.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        upstream = F.interpolate(
            upstream, size=(H, W), mode="bilinear", align_corners=False
        )

        flow_features = F.interpolate(
            flow_features, size=(H, W), mode="bilinear", align_corners=False
        )
        gate = self.in_flow(flow_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for i in range(self.num_blocks):
            out = torch.cat([upstream, original_features - to_be_compensated], dim=1)
            out = self.conv1[i](out)
            out = self.norm1[i](out)
            out = self.activation(out)
            out = out * torch.sigmoid(gate)
            out = self.conv2[i](out)
            out = self.norm2[i](out)
            out = self.activation(out)
            out = self.conv3[i](out)
            upstream = out + to_be_compensated

        return upstream


class FusionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        previous_dim: int | None = None,
        use_norm: bool = True,
        kernel_size: int | None = None,
        padding_mode: str = "reflect",
        num_blocks: int = 2,
        gate_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.merge = BottleneckRG(
            hidden_dim,
            current_dim=input_dim,
            previous_dim=previous_dim,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            use_norm=use_norm,
            num_blocks=num_blocks,
            gate_dim=gate_dim,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, to_be_compensated, upstream, *args, **kwargs):
        upstream = self.merge(
            upstream=upstream, to_be_compensated=to_be_compensated, *args, **kwargs
        )
        return upstream
