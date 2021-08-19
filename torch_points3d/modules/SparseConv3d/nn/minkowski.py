import torch
import MinkowskiEngine as ME


class Conv3d(ME.MinkowskiConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=3,
        )


class Conv3dTranspose(ME.MinkowskiConvolutionTranspose):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            dimension=3,
        )


class BatchNorm(ME.MinkowskiBatchNorm):
    def __repr__(self):
        return self.bn.__repr__()


class ReLU(ME.MinkowskiReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace=False)


def cat(*args):
    return ME.cat(*args)


def SparseTensor(feats, coordinates, batch, device=torch.device("cpu")):
    if batch.dim() == 1:
        batch = batch.unsqueeze(-1)
    coords = torch.cat([batch.int(), coordinates.int()], -1)
    return ME.SparseTensor(features=feats, coordinates=coords, device=device)


class MinkowskiNonlinearityBase(MinkowskiModuleBase):
    """
    taken from https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/MinkowskiNonlinearity.py
    """

    def __init__(self, module):
        super(MinkowskiNonlinearityBase, self).__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input.F)
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        return self.module.__class__.__name__ + "()"


def create_activation_function(activation: torch.nn.Module = torch.nn.ReLU()):
    return MinkowskiNonlinearityBase(module=activation)
