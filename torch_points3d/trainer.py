import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.distributed import rank_zero_info


# from hydra.utils.instantiate as hydra_instantiate

from torch_points3d.tasks.base_model import PointCloudBaseModule
from torch_points3d.datasets.base_dataset import PointCloudDataModule, PointCloudDataConfig, PointCloudDataModule
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
        self.data_module: PointCloudDataModule = instantiator.data_module(dataset)
        if self.data_module is None:
            raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
        if not isinstance(self.data_module, LightningDataModule):
            raise ValueError(
                "The instantiator did not return a DataModule instance." " Hydra hint: is `dataset._target_` defined?`"
            )
        self.data_module.setup("fit")

        self.litmodel: PointCloudBaseModule = instantiator.litmodel(model)
        print(self.litmodel)
        self.trainer = instantiator.trainer(
            trainer,
            logger=None,  # eventually add logger config back in
        )

    def train(self):
        self.trainer.fit(self.litmodel, self.data_module)
