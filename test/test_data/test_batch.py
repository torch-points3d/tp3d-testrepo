import unittest
import torch
from torch_geometric.data import Data
import numpy as np

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
sys.path.append(ROOT)

from torch_points3d.data.batch import SimpleBatch



def test_fromlist():
    nb_points = 100
    pos = torch.randn((nb_points, 3))
    y = torch.tensor([range(10) for i in range(pos.shape[0])], dtype=torch.float)
    d = Data(pos=pos, y=y)
    b = SimpleBatch.from_data_list([d, d])
    np.testing.assert_equal(b.pos.size(), (2, 100, 3))
    np.testing.assert_equal(b.y.size(), (2, 100, 10))



