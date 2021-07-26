from typing import Any, Callable, Dict, Optional, Sequence
from dataclasses import dataclass

import hydra
import torch_geometric
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_points3d.core.config import BaseDataConfig


@dataclass
class PointCloudDataConfig(BaseDataConfig):
    batch_size: int = 32
    num_workers: int = 0
    dataroot: str = "data"
    pre_transform: Sequence[Any] = None
    train_transform: Sequence[Any] = None
    test_transform: Sequence[Any] = None


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg: PointCloudDataConfig = PointCloudDataConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = None

        self.cfg.dataroot = hydra.utils.to_absolute_path(self.cfg.dataroot)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"], batch_size=self.batch_size, num_workers=self.cfg.num_workers, collate_fn=self.collate_fn,
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

    @property
    def collate_fn(self) -> Optional[Callable]:
        return torch_geometric.data.batch.Batch.from_data_list

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}