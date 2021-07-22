import torch
import os
import sys
import unittest
import pytest
import numpy as np


DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.append('.')

from torch_points3d.metrics.segmentation.segmentation_tracker import compute_intersection_union_per_class
from torch_points3d.metrics.segmentation.segmentation_tracker import compute_average_intersection_union
from torch_points3d.metrics.segmentation.segmentation_tracker import compute_overall_accuracy
from torch_points3d.metrics.segmentation.segmentation_tracker import compute_mean_class_accuracy



def test_compute_intersection_union_per_class():
    matrix = torch.tensor([[4, 1], [2, 10]])
    iou, _ = compute_intersection_union_per_class(matrix)
    miou = compute_average_intersection_union(matrix)
    np.testing.assert_allclose(iou[0].item(), 4 / (4.0 + 1.0 + 2.0))
    np.testing.assert_allclose(iou[1].item(), 10 / (10.0 + 1.0 + 2.0))
    np.testing.assert_allclose(iou.mean().item(), miou.item())

def test_compute_overall_accuracy():
    list_matrix  = [
        torch.tensor([[4, 1], [2, 10]]).float(),
        torch.tensor([[4, 1], [2, 10]]).int(),
        torch.tensor([[0, 0], [0, 0]]).float()
    ]
    list_answer = [
        (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0),
        (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0),
        0.0
    ]
    for i in range(len(list_matrix)):
        acc = compute_overall_accuracy(list_matrix[i])
        if(isinstance(acc, torch.Tensor)):
            np.testing.assert_allclose(acc.item(), list_answer[i])
        else:
            np.testing.assert_allclose(acc, list_answer[i])


def test_compute_mean_class_accuracy():
    matrix = torch.tensor([[4, 1], [2, 10]]).float()
    macc = compute_mean_class_accuracy(matrix)
    np.testing.assert_allclose(macc.item(), (4/5 + 10/12)*0.5)



@pytest.mark.parametrize("missing_as_one, answer", [pytest.param(False, (0.5 + 0.5) / 2), pytest.param(True, (0.5 + 1 + 0.5) / 3)])
def test_test_getMeanIoUMissing(missing_as_one, answer):
    matrix = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(compute_average_intersection_union(matrix, missing_as_one=missing_as_one).item(), answer)
        
