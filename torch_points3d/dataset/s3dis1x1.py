from typing import Any, Callable, Dict, Optional, Sequence
from omegaconf import MISSING
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


class s3dis_data_module(PointCloudDataModule):
    def __init__(self, cfg: S3DISDataConfig = S3DISDataConfig()) -> None:
        super().__init__(cfg)

        self.ds = {
            "train": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=True,
                pre_transform=self.cfg.pre_transform,
                transform=self.cfg.train_transform,
            ),
            "test": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=False,
                pre_transform=self.cfg.pre_transform,
                transform=self.cfg.train_transform,
            ),
        }
