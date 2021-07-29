import torch
from typing import Optional, Tuple, Union


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
    if len(labels_presents) == 0:
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


def compute_intersection_union_per_class(
    confusion_matrix: torch.Tensor, return_existing_mask: bool = False, eps: float = 1e-8
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
