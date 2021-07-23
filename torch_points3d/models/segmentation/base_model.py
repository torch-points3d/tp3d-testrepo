from torch_points3d.models.base_model import PointCloudBaseModel
import torch.nn.functional as F
import torch.nn as nn

# Overrides PointCloudBaseModel to create segmentation-specific models
class SegmentationBaseModel(PointCloudBaseModel):

    # Most segmentation models will share the same head and criterion
    # Note that we don't even have to create a sparseconv3d subclass of PointCloudBaseModel
    # because the actual training logic is contained all within this and PointCloudBaseModel
    # so all we need to do is define a `backend` (in this case sparseconv3d)
    def _build_model(self, backbone_cfg):
        super()._build_model(backbone_cfg)

        self.head = nn.Sequential(nn.Linear(self.backbone.output_nc, self.num_classes))
        self.criterion = F.nll_loss
