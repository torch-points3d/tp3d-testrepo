import numpy as np
import torch
import sys
import os

import pytest


from torch_geometric.data import Data

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)
sys.path.append(".")

from torch_points3d.metrics.segmentation.segmentation_tracker import SegmentationTracker


class MockDataset:
    INV_OBJECT_LABEL = {0: "first", 1: "wall", 2: "not", 3: "here", 4: "hoy"}
    pos = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [-1, 0, 0]]).float()
    test_label = torch.tensor([1, 1, 0, 0])

    def __init__(self):
        self.num_classes = 2

    @property
    def test_data(self):
        return Data(pos=self.pos, y=self.test_label)

    def has_labels(self, stage):
        return True


class MockModel:
    def __init__(self):
        self.iter = 0
        self.losses = [
            {"loss_1": torch.tensor(1.0), "loss_2": torch.tensor(2.0)},
            {"loss_1": torch.tensor(2.0), "loss_2": torch.tensor(2.0)},
            {"loss_1": torch.tensor(1.0), "loss_2": torch.tensor(2.0)},
            {"loss_1": torch.tensor(1.0), "loss_2": torch.tensor(2.0)},
        ]
        self.outputs = [
            torch.tensor([[0, 1], [0, 1]]),
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[1, 0], [1, 0], [1, 0]]),
        ]
        self.labels = [torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([0, 0, -100])]
        self.batch_idx = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0, 1])]

    def get_input(self):
        return Data(pos=MockDataset.pos[:2, :], origin_id=torch.tensor([0, 1]))

    def get_output(self):
        return self.outputs[self.iter].float()

    def get_labels(self):
        return self.labels[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch(self):
        return self.batch_idx[self.iter]

    @property
    def device(self):
        return "cpu"


def test_forward():
    tracker = SegmentationTracker(num_classes=2, stage="train")
    model = MockModel()
    output = {"preds": model.get_output(), "labels": model.get_labels()}
    losses = model.get_current_losses()
    metrics = tracker(output, losses)
    # metrics = tracker.get_metrics()

    for k in ["train_acc", "train_miou", "train_macc"]:
        np.testing.assert_allclose(metrics[k], 100, rtol=1e-5)
    model.iter += 1
    output = {"preds": model.get_output(), "labels": model.get_labels()}
    losses = model.get_current_losses()
    metrics = tracker(output, losses)
    # metrics = tracker.get_metrics()
    metrics = tracker.finalise()
    for k in ["train_acc", "train_macc"]:
        assert metrics[k] == 50
        np.testing.assert_allclose(metrics["train_miou"], 25, atol=1e-5)
        assert metrics["train_loss_1"] == 1.5

    tracker = SegmentationTracker(num_classes=2, stage="test")
    model.iter += 1
    output = {"preds": model.get_output(), "labels": model.get_labels()}
    losses = model.get_current_losses()
    metrics = tracker(output, losses)
    # metrics = tracker.get_metrics()
    for name in ["test_acc", "test_miou", "test_macc"]:
        np.testing.assert_allclose(metrics[name].item(), 0, atol=1e-5)


@pytest.mark.parametrize("finalise", [pytest.param(True), pytest.param(False)])
def test_ignore_label(finalise):
    tracker = SegmentationTracker(num_classes=2, ignore_label=-100, stage="test")
    model = MockModel()
    model.iter = 3
    output = {"preds": model.get_output(), "labels": model.get_labels()}
    losses = model.get_current_losses()
    metrics = tracker(output, losses)
    if not finalise:
        # metrics = tracker.get_metrics()
        for k in ["test_acc", "test_miou", "test_macc"]:
            np.testing.assert_allclose(metrics[k], 100)
    else:
        tracker.finalise()
        with pytest.raises(RuntimeError):
            tracker(output)
