from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np
import pytest
import torch

from torch_geometric.data import Data

from torch_points3d.datasets.base_dataset import PointCloudDataModule, PointCloudDataConfig
from test.mockdatasets import SegmentationMockDataset

@dataclass
class MockConfig(PointCloudDataConfig):
    batch_size: int = 16
    num_workers: int = 0
    size: int = 3
    is_same_size: bool = False
    conv_type: str = "dense"
    multiscale: bool = False
    num_classes: int = 2


class SegmentationMockDataLoader(PointCloudDataModule):

    def __init__(self, cfg, transform: Optional[Callable] = None):
        super().__init__(cfg)

        self.ds = {
            "train": SegmentationMockDataset(train=True, transform=transform, size=self.cfg.size, is_same_size=self.cfg.is_same_size, num_classes=self.cfg.num_classes),
            "validation": SegmentationMockDataset(train=False, transform=transform, size=self.cfg.size, is_same_size=self.cfg.is_same_size, num_classes=self.cfg.num_classes)
        }


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_classes", [2, 4, 6, 9, 10])
@pytest.mark.parametrize("size", [3, 10, 100])
@pytest.mark.parametrize("conv_type, is_same_size, multiscale",
                         [pytest.param("dense", True, False),
                          pytest.param("dense", False, False, marks=pytest.mark.xfail),
                          pytest.param("partial_dense", True, False),
                          pytest.param("partial_dense", False, False),
                          pytest.param("partial_dense", True, True, marks=pytest.mark.xfail),
                          pytest.param("sparse", True, False),
                         ])
def test_dataloader(conv_type, is_same_size, size, multiscale, num_classes, batch_size):
    cfg = MockConfig(conv_type=conv_type, is_same_size=is_same_size, size=size, multiscale=multiscale, num_classes=2, batch_size=batch_size)
    dataloader = SegmentationMockDataLoader(cfg)

    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    for loader in [train_dataloader, val_dataloader]:
        # test len
        np.testing.assert_equal(len(loader.dataset), size)
        # test batch collate
        batch = next(iter(train_dataloader))
        num_samples = PointCloudDataModule.get_num_samples(batch, conv_type)
        np.testing.assert_equal(num_samples, min(batch_size, size))
        if(is_same_size):
            if(conv_type.lower() == "dense"):
                np.testing.assert_equal(batch.pos.size(1), 1000)
            else:
                for i in range(min(batch_size, size)):
                    np.testing.assert_equal(batch.pos[batch.batch == i].shape[0], 1000)
        if(multiscale):
            # test downsample and upsample are
            pass

