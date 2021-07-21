import unittest
import sys
import os

from omegaconf import OmegaConf

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)

from torch_points3d.model.segmentation.sparseconv3d import APIModel



class TestAPIModel(unittest.TestCase):

    def test_load(self):
        option_dataset = OmegaConf.create({
            "feature_dimension": 1,
            "num_classes": 10
        })
        
        option = OmegaConf.load(os.path.join(ROOT, "conf", "models", "segmentation", "sparseconv3d.yaml"))
        name_model = list(option.keys())[0]
        model = APIModel(option[name_model], option_dataset)


if __name__ == "__main__":
    unittest.main()
