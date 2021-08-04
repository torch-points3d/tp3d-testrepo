from typing import Any, Callable, Dict, Optional, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import hydra
import torch_geometric
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_points3d.core.config import BaseDataConfig
from torch_geometric.data import Data
from torch_points3d.data.multiscale_data import MultiScaleBatch
from torch_points3d.data.batch import SimpleBatch

from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import ConvolutionFormatFactory


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
        self.ds: Optional[Dict[str, Dataset]] = None
        self.cfg.dataroot = hydra.utils.to_absolute_path(self.cfg.dataroot)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.ds["train"]
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.ds["validation"]
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds.keys():
            return self._dataloader(
                self.ds["test"]
            )

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def collate_fn(self) -> Optional[Callable]:
        return torch_geometric.data.batch.Batch.from_data_list

    @staticmethod
    def _collate_fn(batch: Data, collate_fn: Callable, pre_collate_transform: Optional[Callable] = None):
        if pre_collate_transform:
            batch = pre_collate_transform(batch)
        return collate_fn(batch)

    @staticmethod
    def _get_collate_function(conv_type: str, is_multiscale: bool, pre_collate_transform: Optional[Callable] = None):
        is_dense: bool = ConvolutionFormatFactory.check_is_dense_format(conv_type)
        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower():
                fn = MultiScaleBatch.from_data_list
            else:
                raise NotImplementedError(
                    "MultiscaleTransform is activated and supported only for partial_dense format"
                )
        else:
            if is_dense:
                fn = SimpleBatch.from_data_list
            else:
                fn = torch_geometric.data.batch.Batch.from_data_list
        return partial(PointCloudDataModule._collate_fn, collate_fn=fn, pre_collate_transform=pre_collate_transform)

    def _dataloader(self, dataset: Dataset, pre_batch_collate_transform: Optional[Callable] = None, conv_type: str = "partial_dense", precompute_multi_scale: bool = False, **kwargs):
        batch_collate_function = self.__class__._get_collate_function(
            conv_type, precompute_multi_scale, pre_batch_collate_transform
        )
        num_workers = self.cfg.num_workers
        persistent_workers = (num_workers > 0)
        
        dataloader = partial(
            DataLoader, collate_fn=batch_collate_function, worker_init_fn=np.random.seed,
            persistent_workers=persistent_workers,
            batch_size=self.batch_size,
            num_workers=num_workers
        )
        return dataloader(dataset, **kwargs)

    @property
    def model_data_kwargs(self) -> Dict:
        """
        Override to provide the model with additional kwargs.
        This is useful to provide the number of classes/pixels to the model or any other data specific args
        Returns: Dict of args
        """
        return {}
