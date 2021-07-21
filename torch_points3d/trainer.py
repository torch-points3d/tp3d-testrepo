from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.distributed import rank_zero_info
from hydra.utils.instantiate as hydra_instantiate

from torch_points3d.datasets.dataset_factory import instantiate_dataset, convert_to_lightning_data_module
from torch_points3d.datasets.dataset_factory import instantiate_dataset


class LitTrainer:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    def instantiate_trainer(self):
        trainer = hydra_instantiate(self._cfg.trainer)
        return trainer

    def instantiate_dataset_and_model(self):
        dataset: BaseDataset = instantiate_dataset(self._cfg.data)
        model: BaseModel = instantiate_model(copy.deepcopy(cfg), dataset)
        model.instantiate_optimizers(cfg) # we will change it and instantiate the optimizers separately
        model.set_pretrained_weights()
        dataset.create_dataloaders(
            cfg.training.batch_size,
            cfg.training.shuffle,
            cfg.training.num_workers,
            model.conv_type == "PARTIAL_DENSE" and getattr(cfg.training, "precompute_multi_scale", False),
        )
        data_module = convert_to_lightning_data_module(dataset)
        return model, data_module

    def train(self):


        # model.tracker_options = cfg.get("tracker_options", {})
        # model.trackers = data_module.trackers
        model, data_module = self.instantiate_dataset_and_model()
        trainer = self.instantiate_trainer()
        trainer.fit(model, data_module)
