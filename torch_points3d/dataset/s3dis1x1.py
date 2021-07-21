from typing import Any, Callable, Dict, Optional, Sequence
from omegaconf import MISSING, DictConfig
from dataclasses import dataclass

import hydra.utils
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_points3d.dataset.base_dataset import PointCloudDataModule, PointCloudDataConfig

from torch_geometric.datasets import S3DIS as S3DIS1x1


@dataclass
class S3DISDataConfig(PointCloudDataConfig):
    batch_size: int = 32
    num_workers: int = 0
    fold: int = 6

def show(x):
    print(f"type: {type(x).__name__}, value: {repr(x)}")

class s3dis_data_module(PointCloudDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        show(cfg)
        cfg.num_workers = "aj"
        show(cfg)
        # print("pre_transform: ", self.cfg.pre_transform)
        # self.ds = {
        #     "train": S3DIS1x1(
        #         self.cfg.dataroot,
        #         test_area=self.cfg.fold,
        #         train=True,
        #         pre_transform=self.cfg.pre_transform,
        #         transform=self.cfg.train_transform,
        #     ),
        #     "test": S3DIS1x1(
        #         self.cfg.dataroot,
        #         test_area=self.cfg.fold,
        #         train=False,
        #         pre_transform=self.cfg.pre_transform,
        #         transform=self.cfg.train_transform,
        #     ),
        # }
