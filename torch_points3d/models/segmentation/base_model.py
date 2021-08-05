from omegaconf import DictConfig
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from torch_points3d.core.instantiator import Instantiator
from torch_points3d.models.base_model import PointCloudBaseModel


class SegmentationBaseModel(PointCloudBaseModel):
    def __init__(
        self,
        instantiator: Instantiator,
        num_classes: int,
        backbone: DictConfig,
        criterion: DictConfig,
        conv_type: Optional[str] = None,
    ):
        super().__init__(instantiator)

        print(backbone)
        self.backbone = self.instantiator.backbone(backbone)
        self.criterion = self.instantiator.instantiate(criterion)

        self.head = nn.Sequential(nn.Linear(self.backbone.output_nc, num_classes))

    def set_input(self, data: Data) -> None:
        self.batch_idx = data.batch.squeeze()
        self.input = data
        if data.y is not None:
            self.labels = data.y
        else:
            self.labels = None

    def forward(self) -> Optional[torch.Tensor]:
        features = self.backbone(self.input).x
        logits = self.head(features)
        self._output = F.log_softmax(logits, dim=-1)
        loss = self.compute_losses()
        return loss

    def compute_losses(self):
        """
        compute every loss. store the total loss in an attribute _loss
        """
        if self.labels is not None and self.criterion is not None:
            self._losses["loss"] = self.criterion(self._output, self.labels)
            return self._losses["loss"]
        else:
            return None

    def get_outputs(self) -> Dict[str, torch.Tensor]:
        """
        return the outputs to track for the metrics
        """
        return {"labels": self.labels, "preds": self._output}
