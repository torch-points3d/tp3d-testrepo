from typing import Any, List, Optional
import torch

import sys

from torch_points3d.core.common_modules import Seq, Identity
import torch_points3d.applications.modules.SparseConv3d.nn as snn
from omegaconf import DictConfig
from omegaconf import OmegaConf


class ResBlock(torch.nn.Module):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either MinkowskConvolution or MinkowskiConvolutionTranspose
    dimension:
        Dimension of the spatial grid
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        convolution: Any,
        bn: Any,
        bn_args: DictConfig = OmegaConf.create(),
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = snn.create_activation_function(activation)
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1))
            .append(bn(output_nc, **bn_arg))
            .append(self.activation)
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1))
            .append(bn(output_nc, **bn_args))
            .append(self.activation)
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(input_nc, output_nc, kernel_size=1, stride=1)).append(bn(output_nc, **bn_args))
            )
        else:
            self.downsample = None

    def forward(self, x: snn.SparseTensor):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class BottleneckBlock(torch.nn.Module):
    """
    Bottleneck block with residual
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        convolution: Any,
        bn: Any,
        bn_args: DictConfig = OmegaConf.create(),
        reduction: int = 4,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = snn.create_activation_function(activation)
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc // reduction, kernel_size=1, stride=1))
            .append(bn(output_nc // reduction, **bn_args))
            .append(self.activation)
            .append(
                convolution(
                    output_nc // reduction,
                    output_nc // reduction,
                    kernel_size=3,
                    stride=1,
                )
            )
            .append(bn(output_nc // reduction, **bn_args))
            .append(self.activation)
            .append(
                convolution(
                    output_nc // reduction,
                    output_nc,
                    kernel_size=1,
                )
            )
            .append(bn(output_nc, **bn_args))
            .append(self.activation)
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(input_nc, output_nc, kernel_size=1, stride=1)).append(bn(output_nc, **bn_args))
            )
        else:
            self.downsample = None

    def forward(self, x: snn.SparseTensor):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


_res_blocks = sys.modules[__name__]


class ResNetDown(torch.nn.Module):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = "Conv3d"
    BATCHNORM = "BatchNorm"

    def __init__(
        self,
        down_conv_nn: List[int] = [],
        kernel_size: int = 2,
        dilation: int = 1,
        stride: int = 2,
        N: int = 1,
        block: str = "ResBlock",
        activation: torch.nn.Module = torch.nn.ReLU(),
        bn_args: Optional[DictConfig] = None,
        **kwargs,
    ):
        assert len(down_conv_nn) == 2
        if bn_args is None:
            bn_args = OmegaConf.create()
        else:
            bn_args = bn_args.to_dict()
        block = getattr(_res_blocks, block)
        super().__init__()
        if stride > 1:
            conv1_output = down_conv_nn[0]
        else:
            conv1_output = down_conv_nn[1]

        conv = getattr(snn, self.CONVOLUTION)
        bn = getattr(snn, self.BATCHNORM)
        self.conv_in = (
            Seq()
            .append(
                conv(
                    in_channels=down_conv_nn[0],
                    out_channels=conv1_output,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
            .append(bn(conv1_output, **bn_args))
            .append(snn.create_activation_function(activation))
        )

        if N > 0:
            self.blocks = Seq()
            for _ in range(N):
                self.blocks.append(
                    block(conv1_output, down_conv_nn[1], conv, bn=bn, bn_args=bn_args, activation=activation)
                )
                conv1_output = down_conv_nn[1]
        else:
            self.blocks = None

    def forward(self, x: snn.SparseTensor):
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class ResNetUp(ResNetDown):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = "Conv3dTranspose"

    def __init__(
        self,
        up_conv_nn: List = [],
        kernel_size: int = 2,
        dilation: int = 1,
        stride: int = 2,
        N: int = 1,
        block: str = "ResBlock",
        activation: torch.nn.Module = torch.nn.ReLU(),
        bn_args: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__(
            down_conv_nn=up_conv_nn,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            N=N,
            activation=activation,
            bn_args=bn_args,
            block=block ** kwargs,
        )

    def forward(self, x: snn.SparseTensor, skip: Optional[snn.SparseTensor]):
        if skip is not None:
            inp = snn.cat(x, skip)
        else:
            inp = x
        return super().forward(inp)
