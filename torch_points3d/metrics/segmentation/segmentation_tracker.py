import torch
from typing import Dict, Optional, Tuple, Any, Union
from torchmetrics import ConfusionMatrix
from torchmetrics import Metric

from torch_points3d.metrics.base_tracker import BaseTracker


def compute_average_intersection_union(confusion_matrix: torch.Tensor, missing_as_one: bool = False) -> torch.Tensor:
    """
    compute intersection over union on average from confusion matrix
    Parameters
    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    missing_as_one: bool, default: False
    """

    values, existing_classes_mask = compute_intersection_union_per_class(confusion_matrix, return_existing_mask=True)
    if torch.sum(existing_classes_mask) == 0:
        return torch.sum(existing_classes_mask)
    if missing_as_one:
        values[~existing_classes_mask] = 1
        existing_classes_mask[:] = True
    return torch.sum(values[existing_classes_mask]) / torch.sum(existing_classes_mask)


def compute_mean_class_accuracy(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    compute intersection over union on average from confusion matrix
    
    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """
    total_gts = confusion_matrix.sum(1)
    labels_presents = torch.where(total_gts > 0)[0]
    if(len(labels_presents) == 0):
        return total_gts[0]
    ones = torch.ones_like(total_gts)
    max_ones_total_gts = torch.cat([total_gts[None, :], ones[None, :]], 0).max(0)[0]
    re = (torch.diagonal(confusion_matrix)[labels_presents].float() / max_ones_total_gts[labels_presents]).sum()
    return re / float(len(labels_presents))

def compute_overall_accuracy(confusion_matrix: torch.Tensor) -> Union[int, torch.Tensor]:
    """
    compute overall accuracy from confusion matrix
    
    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """
    all_values = confusion_matrix.sum()
    if all_values == 0:
        return 0
    matrix_diagonal = torch.trace(confusion_matrix)
    return matrix_diagonal.float() / all_values

def compute_intersection_union_per_class(confusion_matrix: torch.Tensor, return_existing_mask: bool = False, eps: float = 1e-8) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    compute intersection over union per class from confusion matrix
    
    Parameters
    ----------
    confusion_matrix: torch.Tensor
      square matrix
    """

    TP_plus_FN = confusion_matrix.sum(0)
    TP_plus_FP = confusion_matrix.sum(1)
    TP = torch.diagonal(confusion_matrix)
    union = TP_plus_FN + TP_plus_FP - TP
    iou = eps + TP / (union + eps)
    existing_class_mask = union > 1e-3
    if return_existing_mask:
        return iou, existing_class_mask
    else:
        return iou, None



class SegmentationTracker(BaseTracker):
    """
    track different registration metrics
    """

    def __init__(
            self, num_classes: int,
            stage: str = "train",
            ignore_label: int = -1,
            eps: float = 1e-8,):
        super().__init__(stage)
        self._ignore_label = ignore_label
        self._num_classes = num_classes
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=self._num_classes)
        self.eps = eps

    def compute_metrics_from_cm(self, matrix: torch.Tensor) -> Dict[str, Any]:
        acc = compute_overall_accuracy(matrix)
        macc = compute_mean_class_accuracy(matrix)
        miou = compute_average_intersection_union(matrix)
        iou_per_class, _ = compute_intersection_union_per_class(matrix, eps=self.eps)
        iou_per_class_dict = {i: 100 * v for i, v in enumerate(iou_per_class)}
        return {
            "{}_acc".format(self.stage): 100 * acc,
            "{}_macc".format(self.stage): 100 * macc,
            "{}_miou".format(self.stage): 100 * miou,
            "{}_iou_per_class".format(self.stage): iou_per_class_dict,
        }


    def track(self, output, **kwargs) -> Dict[str, Any]:
        mask = output['targets'] != self._ignore_label
        matrix = self.confusion_matrix_metric(output['preds'][mask], output['targets'][mask])
        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def _finalise(self):
        matrix = self.confusion_matrix_metric.compute()
        segmentation_metrics = self.compute_metrics_from_cm(matrix)
        return segmentation_metrics

    def reset(self, stage: str = "train"):
        super().reset(stage)
        



