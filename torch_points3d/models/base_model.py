import torch.nn as nn

from torch_points3d.core.instantiator import Instantiator


class PointCloudBaseModel(nn.Module):
    def __init__(self, instantiator: Instantiator):
        super().__init__()

        self.instantiator = instantiator

    def set_input(self, data):
        raise (NotImplementedError("set_input needs to be defined!"))

    def forward(self):
        raise (NotImplementedError("forward needs to be defined!"))
