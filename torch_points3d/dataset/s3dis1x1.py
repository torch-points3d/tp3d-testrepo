from typing import Any, Callable, Dict, Optional, Sequence
from omegaconf import MISSING
from dataclasses import dataclass

import hydra.utils
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch.utils.data import DataLoader

from torch_geometric.datasets import S3DIS as S3DIS1x1


@dataclass
class S3DISDataConfig:
    batch_size: int = 32
    num_workers: int = 0
    fold: int = 6
    dataroot: str = "data"
    pre_transform: Sequence[Any] = None
    train_transform: Sequence[Any] = None
    test_transform: Sequence[Any] = None


class s3dis_data_module(pl.LightningDataModule):
    def __init__(self, cfg: S3DISDataConfig = S3DISDataConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.__instantiate_transform()

        self.ds = {
            "train": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=True,
                pre_transform=self.pre_transform,
                transform=self.train_transform,
            ),
            "test": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=False,
                pre_transform=self.pre_transform,
                transform=self.test_transform,
            ),
        }

    def __instantiate_transform(self):
        for transform_type in ["pre_transform", "train_transform", "test_transform"]:
            if transform_type in self.cfg:
                transforms = []
                for transform in self.cfg[transform_type]:
                    transforms.append(hydra.utils.instantiate(transform))
                transform = T.Compose(transforms)
                setattr(self, transform_type, transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size
