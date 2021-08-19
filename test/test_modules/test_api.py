from torch_points3d.core.data_transform import GridSampling3D
from test.mockdatasets import MockDatasetGeometric, MockDataset
import pytest
import torch
from omegaconf import OmegaConf

from torch_points3d.applications.sparseconv3d import SparseConv3d
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.applications.kpconv import KPConv


seed = 0
torch.manual_seed(seed)
device = "cpu"


@pytest.mark.parametrize("architecture", [pytest.param("unet"), pytest.param("encoder", marks=pytest.mark.xfail)])
@pytest.mark.parametrize("input_nc", [0, 3])
@pytest.mark.parametrize("num_layers", [4])
@pytest.mark.parametrize("grid_sampling", [0.02, 0.04])
@pytest.mark.parametrize("in_feat", [32])
@pytest.mark.parametrize("output_nc", [None, 32])
def test_kpconv(architecture, input_nc, num_layers, grid_sampling, in_feat, output_nc):
    if output_nc is not None:
        model = KPConv(
            architecture=architecture,
            input_nc=input_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            output_nc=output_nc,
            config=None,
        )
    else:
        model = KPConv(
            architecture=architecture,
            input_nc=input_nc,
            in_feat=in_feat,
            in_grid_size=grid_sampling,
            num_layers=num_layers,
            config=None,
        )

    dataset = MockDatasetGeometric(input_nc + 1, transform=GridSampling3D(0.01), num_points=128)
    assert len(model._modules["down_modules"]) == num_layers + 1
    assert len(model._modules["inner_modules"]) == 1
    assert len(model._modules["up_modules"]) == 4
    if output_nc is None:
        assert not model.has_mlp_head
        assert model.output_nc == in_feat

    try:
        data_out = model.forward(dataset[0])
        assert data_out.x.shape[1] == in_feat
    except Exception as e:
        print("Model failing:")
        print(model)
        raise e


@pytest.mark.skip("RSConv is not yet implemented")
def test_pn2():

    input_nc = 2
    num_layers = 3
    output_nc = 5
    model = PointNet2(
        architecture="unet",
        input_nc=input_nc,
        output_nc=output_nc,
        num_layers=num_layers,
        multiscale=True,
        config=None,
    )
    dataset = MockDataset(input_nc, num_points=512)
    self.assertEqual(len(model._modules["down_modules"]), num_layers - 1)
    self.assertEqual(len(model._modules["inner_modules"]), 1)
    self.assertEqual(len(model._modules["up_modules"]), num_layers)

    try:
        data_out = model.forward(dataset[0])
        self.assertEqual(data_out.x.shape[1], output_nc)
    except Exception as e:
        print("Model failing:")
        print(model)
        raise e


@pytest.mark.skip("RSConv is not yet implemented")
def test_rsconv():
    from torch_points3d.applications.rsconv import RSConv

    input_nc = 2
    num_layers = 4
    output_nc = 5
    model = RSConv(
        architecture="unet",
        input_nc=input_nc,
        output_nc=output_nc,
        num_layers=num_layers,
        multiscale=True,
        config=None,
    )
    dataset = MockDataset(input_nc, num_points=1024)
    self.assertEqual(len(model._modules["down_modules"]), num_layers)
    self.assertEqual(len(model._modules["inner_modules"]), 2)
    self.assertEqual(len(model._modules["up_modules"]), num_layers)

    try:
        data_out = model.forward(dataset[0])
        self.assertEqual(data_out.x.shape[1], output_nc)
    except Exception as e:
        print("Model failing:")
        print(model)
        raise e


@pytest.mark.skip("RSConv is not yet implemented")
def test_sparseconv3d():

    input_nc = 3
    num_layers = 4
    in_feat = 32
    out_feat = in_feat * 3
    model = SparseConv3d(
        architecture="unet",
        input_nc=input_nc,
        in_feat=in_feat,
        num_layers=num_layers,
        config=None,
    )
    dataset = MockDatasetGeometric(input_nc, transform=GridSampling3D(0.01, quantize_coords=True), num_points=128)
    self.assertEqual(len(model._modules["down_modules"]), num_layers + 1)
    self.assertEqual(len(model._modules["inner_modules"]), 1)
    self.assertEqual(len(model._modules["up_modules"]), 4 + 1)
    self.assertFalse(model.has_mlp_head)
    self.assertEqual(model.output_nc, out_feat)

    try:
        data_out = model.forward(dataset[0])
        self.assertEqual(data_out.x.shape[1], out_feat)
    except Exception as e:
        print("Model failing:")
        print(model)
        print(e)
