import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from torch_points3d.trainer import LitTrainer
from torch_points3d.core.instantiator import HydraInstantiator, Instantiator
from dataclasses import dataclass 
from hydra.core.config_store import ConfigStore
from typing import List, Any, Type
from omegaconf import MISSING, OmegaConf
from omegaconf._utils import is_structured_config

OmegaConf.register_new_resolver("get_filename", lambda x: x.split("/")[-1])


@dataclass 
class TrainingDataConfig:
    batch_size: int = 32
    num_workers: int = 0
    lr: float = MISSING

# We seperate the dataset "cfg" from the actual dataset object
# so that we can pass the "cfg" into the dataset constructors as a DictConfig
# instead of as unwrapped parameters
@dataclass
class BaseDataConfig:
    batch_size: int = 32
    num_workers: int = 0
    dataroot: str = "data"

@dataclass
class BaseDataset:
    _target_: str
    cfg: BaseDataConfig

@dataclass
class S3DISDataConfig(BaseDataConfig):
    fold: int =  6

@dataclass 
class S3DISDataset(BaseDataset):
    cfg: S3DISDataConfig

@dataclass
class Config:
    dataset: Any
    training: TrainingDataConfig
    pretty_print: bool = False

def show(x):
    print(f"type: {type(x).__name__}, value: {repr(x)}")

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="dataset", name="dataset_s3dis", node=S3DISDataset)
cs.store(group="training", name="base_trainer", node=TrainingDataConfig)

@hydra.main(config_path="conf", config_name="test_config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.get("pretty_print"):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    dset = cfg.get("dataset")
    show(dset)
    show(dset.cfg)
    dset_cfg = dset.cfg
    # for some reason the cfg object will lose its typing information if hydra passes it to the target class
    # so we pass it manually ourselves and keep the typing info
    delattr(dset, "cfg")
    hydra.utils.instantiate(dset, dset_cfg)


if __name__ == "__main__":
    main()
