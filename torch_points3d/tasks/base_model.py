from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from omegaconf import DictConfig

from torch_points3d.core.instantiator import Instantiator
from torch_points3d.core.config import OptimizerConfig, SchedulerConfig


class PointCloudBaseModule(pl.LightningModule):
    def __init__(
        self,
        model: DictConfig,
        optimizer: OptimizerConfig,
        instantiator: Instantiator,
        scheduler: SchedulerConfig = None,  # scheduler shouldn't be required
    ):
        super().__init__()
        # some optimizers/schedulers need parameters only known dynamically
        # allow users to override the getter to instantiate them lazily
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.instantiator = instantiator

        self._init_model(model)

    def _init_model(self, model_cfg):
        print(model_cfg)
        self.model = self.instantiator.model(model_cfg)

    def configure_optimizers(self) -> Dict:
        """Prepare optimizer and scheduler"""
        optims = {}

        optims["optimizer"] = self.instantiator.optimizer(self, self.optimizer_cfg)

        if self.scheduler_cfg is not None:
            # compute_warmup needs the datamodule to be available when `self.num_training_steps`
            # is called that is why this is done here and not in the __init__
            self.scheduler_cfg.num_training_steps, self.scheduler_cfg.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.scheduler_cfg.num_training_steps,
                num_warmup_steps=self.scheduler_cfg.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.scheduler_cfg.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.scheduler_cfg.num_warmup_steps}")
            scheduler = self.instantiator.scheduler(self.scheduler_cfg, self.optimizer)
            optims["lr_scheduler"] = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return optims

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def setup(self, stage: str):
        self.configure_metrics(stage)

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """
        Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module,
        and initialize any data specific metrics.
        """

    def training_step(self, batch, batch_idx):
        self.model.set_input(batch)
        return self.model.forward()

    def validation_step(self, batch, batch_idx):
        self.model.set_input(batch)
        return self.model.forward()

    def testing_step(self, batch, batch_idx):
        self.model.set_input(batch)
        return self.model.forward()
