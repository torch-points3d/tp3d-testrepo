import unittest
import sys
import os
import torch
from omegaconf import OmegaConf

from torch_geometric.data import Batch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.append('.')

from torch_points3d.models.segmentation.sparseconv3d import APIModel


class TestAPIModel(unittest.TestCase):
    def test_forward(self):
        option_dataset = OmegaConf.create({"feature_dimension": 1, "num_classes": 10})

        option = OmegaConf.load(os.path.join(ROOT, "conf", "models", "segmentation", "sparseconv3d.yaml"))
        name_model = list(option.keys())[0]
        model = APIModel(option[name_model], option_dataset)

        pos = torch.randn(1000, 3)
        coords = torch.round(pos * 10000)
        x = torch.ones(1000, 1)
        batch = torch.zeros(1000).long()
        y = torch.randint(0, 10, (1000,))
        data = Batch(pos=pos, x=x, batch=batch, y=y, coords=coords)
        model.set_input(data)
        model.forward()


if __name__ == "__main__":
    unittest.main()
