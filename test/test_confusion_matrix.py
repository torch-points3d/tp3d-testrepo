import unittest
import pytest


class TestConfusionMatrix(unittest.TestCase):

    def test_compute_intersection_union_per_class(self):
        matrix = torch.tensor([[4, 1], [2, 10]])
        iou = compute_intersection_union_per_class(matrix)
        miou = compute_average_intersection_union(matrix)
        print(iou)
        self.assertAlmostEqual(iou[0].item(), 4 / (4.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou[1].item(), 10 / (10.0 + 1.0 + 2.0))
        self.assertAlmostEqual(iou.mean().item(), miou.item())

    def test_compute_overall_accuracy(self):
        list_matrix  = [
            matrix = torch.tensor([[4, 1], [2, 10]]).float(),
            matrix = torch.tensor([[4, 1], [2, 10]]).int(),
            matrix = torch.tensor([[0, 0], [0, 0]]).float()
        ]
        list_answer = [
            (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0),
            (4.0+10.0)/(4.0 + 10.0 + 1.0 +2.0),
            0.0
        ]
        for i in range(len(list_matrix)):
            acc = compute_overall_accuracy(list_matrix[i])
            self.assertAlmostEqual(acc.item(), list_answer[i])


    def test_compute_mean_class_accuracy(self):
        matrix = torch.tensor([[4, 1], [2, 10]]).float()
        macc = compute_mean_class_accuracy(matrix)
        self.assertAlmostEqual(macc.item(), (4/5 + 10/12)*0.5)


    @pytest.mark.parametrize(["missing_as_one", "answer"], [
        pytest.param(True, (0.5 + 0.5) / 2,)
        pytest.param(False, (0.5 + 1 + 0.5) / 3)
    ])
    def test_test_getMeanIoUMissing(self, missing_as_one, answer):
        matrix = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(compute_average_intersection_union(matrix, missing_as_one=missing_as_one), answer)
        
