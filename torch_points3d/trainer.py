
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.distributed import rank_zero_info

# from hydra.utils.instantiate as hydra_instantiate

from torch_points3d.model.base_model import PointCloudBaseModel
from torch_points3d.dataset.base_dataset import PointCloudDataModule, PointCloudDataConfig, PointCloudDataModule
from torch_points3d.core.instantiator import HydraInstantiator, Instantiator
from torch_points3d.core.config import TaskConfig, TrainerConfig


class LitTrainer:
    def __init__(
        self,
        instantiator: Instantiator,
        dataset: PointCloudDataConfig = PointCloudDataConfig(),
        model: TaskConfig = TaskConfig(),
        trainer: TrainerConfig = TrainerConfig(),
    ):

        # move these instantiations to `train`?
        print(dataset)
        self.data_module: PointCloudDataModule = instantiator.data_module(dataset)
        if self.data_module is None:
            raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
        if not isinstance(self.data_module, LightningDataModule):
            raise ValueError(
                "The instantiator did not return a DataModule instance." " Hydra hint: is `dataset._target_` defined?`"
            )
        self.data_module.setup("fit")
        print(self.data_module)

        self.model: PointCloudBaseModel = instantiator.model(model)
        self.trainer = instantiator.trainer(
            trainer,
            # logger=logger,
        )

    def train(self):
        self.trainer.fit(self.model, self.data_module)
