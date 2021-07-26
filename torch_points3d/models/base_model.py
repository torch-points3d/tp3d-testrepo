from typing import Dict, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data

from torch_points3d.core.instantiator import Instantiator


class PointCloudBaseModel(nn.Module):
    def __init__(self, instantiator: Instantiator):
        super().__init__()

        self.instantiator = instantiator

    def set_input(self, data: Data) -> None:
        raise (NotImplementedError("set_input needs to be defined!"))

    def forward(self) -> Union[torch.Tensor, None]:
        raise (NotImplementedError("forward needs to be defined!"))

    def get_losses(self) -> Union[torch.Tensor, None]:
        raise (NotImplementedError("get_losses needs to be defined!"))