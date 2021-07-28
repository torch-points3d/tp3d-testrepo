from typing import Any, Dict, Optional
import torch
from torch import nn
from torchmetrics import AverageMeter


class BaseTracker(nn.Module):
    """
    pytorch Module to manage the losses and the metrics
    """

    def __init__(self, stage: str = "train"):
        super().__init__()
        self.stage: str = stage
        self._finalised: bool = False
        self.loss_metrics: nn.ModuleDict = nn.ModuleDict()

    def track(self, output_model, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def track_loss(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_loss = dict()
        for key, loss in losses.items():
            loss_key = f"{self.stage}_{key}"
            if loss_key not in self.loss_metrics.keys():
                self.loss_metrics[loss_key] = AverageMeter().to(loss)
            val = self.loss_metrics[loss_key](loss)
            out_loss[loss_key] = val
        return out_loss

    def forward(
        self, output_model: Dict[str, Any], losses: Optional[Dict[str, torch.Tensor]] = None, *args, **kwargs
    ) -> Dict[str, Any]:
        if self._finalised:
            raise RuntimeError("Cannot track new values with a finalised tracker, you need to reset it first")
        tracked_metric = self.track(output_model, *args, **kwargs)
        if losses is not None:
            tracked_loss = self.track_loss(losses)
            tracked_results = dict(**tracked_loss, **tracked_metric)
        else:
            tracked_results = tracked_metric
        return tracked_results

    def _finalise(self) -> Dict[str, Any]:
        raise NotImplementedError("method that aggregae metrics")

    def finalise(self) -> Dict[str, Any]:
        metrics = self._finalise()
        self._finalised = True
        loss_metrics = self.get_final_loss_metrics()
        final_metrics = {**loss_metrics, **metrics}
        return final_metrics

    def get_final_loss_metrics(self):
        metrics = dict()
        for key, m in self.loss_metrics.items():
            metrics[key] = m.compute()
        self.loss_metrics = nn.ModuleDict()
        return metrics

    def reset(self, stage: str = "train"):
        self._finalised = False
        self.stage = stage
        self.loss_metrics = nn.ModuleDict()
