from torch_points3d.datasets.base_dataset import PointCloudDataModule

# Provides shortcuts when we know the children datasets are from torch_geometric
class geometric_dataset(PointCloudDataModule):

    # for ex. a geometric dataset already provides a num_features wrapper
    @property  # type: ignore
    def feature_dimension(self):
        for dset in self.ds.values():
            return dset.num_features