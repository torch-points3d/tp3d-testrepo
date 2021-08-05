from typing import List
import os
import os.path as osp
import pytest

from hydra import compose, initialize
from hydra.test_utils.test_utils import find_parent_dir_containing

from torch_points3d.trainer import LitTrainer
from torch_points3d.core.instantiator import HydraInstantiator


class ScriptRunner:

    @staticmethod
    def find_hydra_conf_dir(config_dir: str = "conf") -> str:
        """
        Util function to find the hydra config directory from the main repository for testing.
        Args:
            config_dir: Name of config directory.
        Returns: Relative config path
        """
        parent_dir = find_parent_dir_containing(config_dir)
        relative_conf_dir = osp.relpath(parent_dir, os.path.dirname(__file__))
        return osp.join(relative_conf_dir, config_dir)

    def train(self, cmd_args: List[str]) -> None:
        relative_conf_dir = self.find_hydra_conf_dir()
        with initialize(config_path=relative_conf_dir, job_name="test_app"):
            cfg = compose(config_name="config", overrides=cmd_args)
            instantiator = HydraInstantiator()
            trainer = LitTrainer(
                instantiator,
                dataset=cfg.get("dataset"),
                trainer=cfg.get("trainer"),
                model=cfg.get("model"))
            trainer.train()

    def hf_train(self, dataset: str, model: str, num_workers: int = 0, fast_dev_run: int = 1):
        cmd_args = []
        cmd_args.extend([
            f'model={model}',
            f'dataset={dataset}',
            f'trainer.max_epochs=1',
            f'training.num_workers=1'
        ])
        self.train(cmd_args)


@pytest.fixture(scope="session")
def script_runner() -> ScriptRunner:
    return ScriptRunner()
