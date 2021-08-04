import pytest
import sys
import os
import torch
from omegaconf import OmegaConf

from torch_geometric.data import Batch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.append(".")

from torch_points3d.models.segmentation.base_model import SegmentationBaseModel
from torch_points3d.core.instantiator import HydraInstantiator
from .conftest import ScriptRunner

@pytest.mark.skip("For now we skip the tests...")
def test_forward(self):
    option_dataset = OmegaConf.create({"feature_dimension": 1, "num_classes": 10})
    option_criterion = OmegaConf.create({"_target_": "torch.nn.NLLLoss"})
    instantiator = HydraInstantiator()

    model = SegmentationBaseModel(instantiator, 10, option_backbone, option_criterion)

    pos = torch.randn(1000, 3)
    coords = torch.round(pos * 10000)
    x = torch.ones(1000, 6)
    batch = torch.zeros(1000).long()
    y = torch.randint(0, 10, (1000,))
    data = Batch(pos=pos, x=x, batch=batch, y=y, coords=coords)
    model.set_input(data)
    model.forward()


def test_s3dis_run(script_runner):
    model = "segmentation/sparseconv3d/ResUNet32"
    dataset = "segmentation/s3dis/s3dis1x1"
    script_runner.hf_train(dataset=dataset, model=model)
