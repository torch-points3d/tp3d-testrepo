from omegaconf import DictConfig

import torch.nn as nn
import torch.nn.functional as F

from torch_points3d.core.instantiator import Instantiator
from torch_points3d.models.base_model import PointCloudBaseModel


class SegmentationBaseModel(PointCloudBaseModel):
    def __init__(self, instantiator: Instantiator, num_classes: int, backbone: DictConfig, criterion: DictConfig):
        super().__init__(instantiator)

        print(backbone)
        self.backbone = self.instantiator.backbone(backbone)
        self.criterion = self.instantiator.instantiate(criterion)

        self.head = nn.Sequential(nn.Linear(self.backbone.output_nc, num_classes))

    def set_input(self, data):
        self.batch_idx = data.batch.squeeze()
        self.input = data
        if data.y is not None:
            self.labels = data.y
        else:
            self.labels = None

    def forward(self):
        features = self.backbone(self.input).x
        logits = self.head(features)
        self.output = F.log_softmax(logits, dim=-1)

        # only compute loss if loss is defined and the dset has labels
        if self.labels is not None and self.criterion is not None:
            return self.criterion(self.output, self.labels)
