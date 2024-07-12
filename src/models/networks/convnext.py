# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Block1D(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm1D(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 convs, implemented with linear layers
        # self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x.permute(0, 2, 1))
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm1D(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, length, channels) while channels_first corresponds to inputs with
    shape (batch_size, channels, length).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ModConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, input_dim=264, output_dim=384,
                 depths=[3, 3, 9], dims=[12, 24, 48], drop_path_rate=0.3,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.hparams = dict(in_chans=in_chans,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            depths=depths,
                            dims=dims,
                            drop_path_rate=drop_path_rate,
                            layer_scale_init_value=layer_scale_init_value,
                            head_init_scale=head_init_scale)

        self.linear = nn.Linear(input_dim, 256)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.fc = nn.Linear(dims[-1], output_dim)
        # self.fc.weight.data.mul_(head_init_scale)
        # self.fc.bias.data.mul_(head_init_scale)

        self.fc = ToeplitzLinear(dims[-1], output_dim)

        self.apply(self._init_weights)
        self.final_norm = nn.Softmax(dim=-1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.size(0), x.size(1), 16, 16)
        # x = x.unsqueeze(2).repeat(1, 1, 264, 1)
        x = self.forward_features(x)
        x = self.fc(x)
        return self.final_norm(x)


class ConvNeXt1D(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, input_dim=264, output_dim=384,
                 depths=[3, 3, 9, 3], dims=[24, 36, 48, 72], drop_path_rate=0.3,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.hparams = dict(in_chans=in_chans,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            depths=depths,
                            dims=dims,
                            drop_path_rate=drop_path_rate,
                            layer_scale_init_value=layer_scale_init_value,
                            head_init_scale=head_init_scale)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=14,
                      padding=7, stride=1),
            nn.LeakyReLU(0.2),
            LayerNorm1D(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(0.1)
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm1D(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=14, padding=7, stride=2),
                nn.LeakyReLU(0.2),
            )
            self.downsample_layers.append(downsample_layer)

        # self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0
        # for i in range(len(dims)):
        #     stage = nn.Sequential(
        #         *[Block1D(dim=dims[i], drop_path=dp_rates[cur + j],
        #                   layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.fc = nn.Linear(dims[-1], output_dim)
        # self.fc.weight.data.mul_(head_init_scale)
        # self.fc.bias.data.mul_(head_init_scale)

        self.fc = ToeplitzLinear(2448, output_dim)
        self.flatten = nn.Flatten(start_dim=1)
        self.apply(self._init_weights)
        self.final_norm = nn.Softmax(dim=-1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            # x = self.stages[i](x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return self.final_norm(x)


class Blk(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.norm = LayerNorm1D(dim, eps=1e-6)
        # self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 convs, implemented with linear layers
        # # self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        # # self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=15,
                      padding=7,
                      stride=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        # input = x
        #
        # x = self.dwconv(x)
        # x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        # x = self.norm(x)
        # x = self.pwconv1(x.permute(0, 2, 1))
        # x = self.act(x)
        # x = self.pwconv2(x)
        # x = x.permute(0, 2, 1)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        # x = input + self.drop_path(x)

        x = x + self.conv(x)
        return x


class Cxt(nn.Module):
    def __init__(self, in_chans=1, input_dim=264, output_dim=384,
                 depths=[1, 1, 1, 1, 1], dims=[40, 30, 30, 10, 3], drop_path_rate=0.3,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.hparams = dict(in_chans=in_chans,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            depths=depths,
                            dims=dims,
                            drop_path_rate=drop_path_rate,
                            layer_scale_init_value=layer_scale_init_value,
                            head_init_scale=head_init_scale)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=15,
                      padding=7, stride=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(1):
            stage = nn.Sequential(
                *[
                    Blk(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        conv_layers = []
        for i in range(len(dims) - 1):
            conv_layers.extend([
                nn.Conv1d(in_channels=dims[i],
                          out_channels=dims[i + 1],
                          kernel_size=1,
                          padding=0,
                          stride=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(p=0.1)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.layernorm = nn.LayerNorm(normalized_shape=[in_chans, input_dim])
        self.flatten = nn.Flatten(start_dim=1)

        self.fc = ToeplitzLinear(dims[-1] * input_dim, output_dim)
        self.final_norm = nn.Softmax(dim=-1)

    def forward_features(self, x):
        x = self.conv1(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.layernorm(x)

        x = self.forward_features(x)

        x = self.conv_layers(x)

        x = self.flatten(x)

        x = self.fc(x)
        return self.final_norm(x)


if __name__ == '__main__':
    model = Cxt(in_chans=1,
                depths=[3, 3, 3, 3, 3], dims=[40, 30, 30, 10, 3], drop_path_rate=0.2,
                layer_scale_init_value=1e-6, head_init_scale=1.)
    x = torch.randn(12, 1, 264)
    result = model.forward(x)
    print(torch.argmax(result, dim=1))
