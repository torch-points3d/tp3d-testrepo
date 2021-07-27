import logging
from typing import Optional, TYPE_CHECKING

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from torch_points3d.datasets.base_dataset import PointCloudDataModule

if TYPE_CHECKING:
    # avoid circular imports
    from torch_points3d.tasks.base_model import PointCloudBaseModule
    from torch_points3d.models.base_model import PointCloudBaseModel


class Instantiator:
    def model(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def optimizer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def scheduler(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def data_module(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def logger(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def trainer(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement method")


class HydraInstantiator(Instantiator):
    def litmodel(self, cfg: DictConfig) -> "PointCloudBaseModule":
        return self.instantiate(cfg, instantiator=self)

    def model(self, cfg: DictConfig) -> "PointCloudBaseModel":
        return self.instantiate(cfg, self)

    def tracker(self, cfg: DictConfig, stage: str = ""):
        return self.instantiate(cfg, stage=stage)

    def backbone(self, cfg: DictConfig):
        return self.instantiate(cfg)

    def optimizer(self, model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
        return self.instantiate(cfg, model.parameters())

    def scheduler(self, cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return self.instantiate(cfg, optimizer=optimizer)

    def data_module(self, cfg: DictConfig) -> PointCloudDataModule:

        return self.instantiate(cfg)

    def logger(self, cfg: DictConfig) -> Optional[logging.Logger]:
        if cfg.get("log"):
            if isinstance(cfg.trainer.logger, bool):
                return cfg.trainer.logger
            return self.instantiate(cfg.trainer.logger)

    def trainer(self, cfg: DictConfig, **kwargs) -> pl.Trainer:
        return self.instantiate(cfg, **kwargs)

    def instantiate(self, *args, **kwargs):
        return hydra.utils.instantiate(*args, **kwargs)
