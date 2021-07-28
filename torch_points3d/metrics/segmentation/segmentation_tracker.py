import torch
from typing import Dict, Optional, Tuple, Any, Union
from torchmetrics import ConfusionMatrix
from torchmetrics import Metric

from torch_points3d.metrics.base_tracker import BaseTracker
import torch_points3d.metrics.segmentation.metrics as mt


class SegmentationTracker(BaseTracker):
    """
    track different registration metrics
    """

    def __init__(
        self,
        num_classes: int,
        stage: str = "train",
        ignore_label: int = -1,
        eps: float = 1e-8,
    ):
        super().__init__(stage)
        self._ignore_label = ignore_label
        self._num_classes = num_classes
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=self._num_classes)
        self.eps = eps

    def compute_metrics_from_cm(self, matrix: torch.Tensor) -> Dict[str, Any]:
        acc = mt.compute_overall_accuracy(matrix)
        macc = mt.compute_mean_class_accuracy(matrix)
        miou = mt.compute_average_intersection_union(matrix)
        iou_per_class, _ = mt.compute_intersection_union_per_class(matrix, eps=self.eps)
        iou_per_class_dict = {f"{self.stage}_iou_class_{i}": (100 * v) for i, v in enumerate(iou_per_class)}
        res = {
            "{}_acc".format(self.stage): 100 * acc,
            "{}_macc".format(self.stage): 100 * macc,
            "{}_miou".format(self.stage): 100 * miou,
        }
        res = dict(**res, **iou_per_class_dict)
        return res

    def track(self, output, **kwargs) -> Dict[str, Any]:
        mask = output["labels"] != self._ignore_label
        matrix = self.confusion_matrix_metric(output["preds"][mask], output["labels"][mask])
        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def _finalise(self):
        matrix = self.confusion_matrix_metric.compute()
        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def reset(self, stage: str = "train"):
        super().reset(stage)
