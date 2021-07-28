from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from torch_points3d.core.instantiator import Instantiator


class PointCloudBaseModel(nn.Module):
    def __init__(self, instantiator: Instantiator):
        super().__init__()

        self.instantiator = instantiator
        self._losses: Dict[str, float] = {}

    def set_input(self, data: Data) -> None:
        raise (NotImplementedError("set_input needs to be defined!"))

    def forward(self) -> Optional[torch.Tensor]:
        raise (NotImplementedError("forward needs to be defined!"))

    def compute_loss(self):
        raise (NotImplementedError("get_losses needs to be defined!"))

    def get_losses(self) -> Optional[Dict["str", torch.Tensor]]:
        return self._losses

    def get_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        return the outputs to track for the metrics
        """
        raise (NotImplementedError("outputs need to be defined"))
