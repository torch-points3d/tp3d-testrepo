from typing import Any, Callable, Dict, Optional, Sequence
from omegaconf import MISSING
from dataclasses import dataclass

import hydra.utils
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_points3d.datasets.geometric_dataset import geometric_dataset

from torch_geometric.datasets import S3DIS as S3DIS1x1
from torch_points3d.core.data_transform import AddOnes, GridSampling3D

# @dataclass
# class S3DISDataConfig(PointCloudDataConfig):
#     batch_size: int = 32
#     num_workers: int = 0
#     fold: int = 6

S3DIS_NUM_CLASSES = 13

class s3dis_data_module(geometric_dataset):
    num_classes = S3DIS_NUM_CLASSES

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        temp_transforms = T.Compose([GridSampling3D(0.05, quantize_coords=True, mode="mean")])
        self.ds = {
            "train": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=True,
                pre_transform=self.cfg.pre_transform,
                transform=temp_transforms,
            ),
            # prefer "validation" over "test" for datasets that only have train/test
            "validation": S3DIS1x1(
                self.cfg.dataroot,
                test_area=self.cfg.fold,
                train=False,
                pre_transform=self.cfg.pre_transform,
                transform=temp_transforms,
            ),
        }
