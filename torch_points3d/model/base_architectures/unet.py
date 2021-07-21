from torch import nn
from torch_geometric.nn import (
    global_max_pool,
    global_mean_pool,
    fps,
    radius,
    knn_interpolate,
)
from torch.nn import (
    Linear as Lin,
    ReLU,
    LeakyReLU,
    BatchNorm1d as BN,
    Dropout,
)
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import logging
import copy

# from torch_points3d.models.base_model import BaseModel
from torch_points3d.core.common_modules.base_modules import Identity


log = logging.getLogger(__name__)


SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]


def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

class BaseFactory:
    def __init__(self, module_name_down, module_name_up, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        else:
            return getattr(self.modules_lib, self.module_name_down, None)

############################# UNWRAPPED UNET BASE ###################################


class UnwrappedUnetBasedModel(nn.Module):
    """Create a Unet unwrapped generator"""

    def _save_sampling_and_search(self, down_conv):
        sampler = getattr(down_conv, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] += sampler
        else:
            self._spatial_ops_dict["sampler"].append(sampler)

        neighbour_finder = getattr(down_conv, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] += neighbour_finder
        else:
            self._spatial_ops_dict["neighbour_finder"].append(neighbour_finder)

    def _save_upsample(self, up_conv):
        upsample_op = getattr(up_conv, "upsample_op", None)
        if upsample_op:
            self._spatial_ops_dict["upsample_op"].append(upsample_op)

    def __init__(self, opt, model_type, modules_lib):
        """Construct a Unet unwrapped generator

        The layers will be appended within lists with the following names
        * down_modules : Contains all the down module
        * inner_modules : Contain one or more inner modules
        * up_modules: Contains all the up module

        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet

        For a recursive implementation. See UnetBaseModel.

        opt is expected to contains the following keys:
        * down_conv
        * up_conv
        * OPTIONAL: innermost

        """
        opt = copy.deepcopy(opt)
        super(UnwrappedUnetBasedModel, self).__init__()
        # detect which options format has been used to define the model
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}

        if is_list(opt.down_conv) or "down_conv_nn" not in opt.down_conv:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt, model_type, modules_lib)

    def _collect_sampling_ids(self, list_data):
        def extract_matching_key(keys, start_token):
            for key in keys:
                if key.startswith(start_token):
                    return key
            return None

        d = {}
        if self.save_sampling_id:
            for idx, data in enumerate(list_data):
                key = extract_matching_key(data.keys, "sampling_id")
                if key:
                    d[key] = getattr(data, key)
        return d

    def _get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def _create_inner_modules(self, args_innermost, modules_lib):
        inners = []
        if is_list(args_innermost):
            for inner_opt in args_innermost:
                module_name = self._get_from_kwargs(inner_opt, "module_name")
                inner_module_cls = getattr(modules_lib, module_name)
                inners.append(inner_module_cls(**inner_opt))

        else:
            module_name = self._get_from_kwargs(args_innermost, "module_name")
            inner_module_cls = getattr(modules_lib, module_name)
            inners.append(inner_module_cls(**args_innermost))

        return inners

    def _init_from_compact_format(self, opt, model_type, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """

        self.down_modules = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        self.save_sampling_id = opt.down_conv.get("save_sampling_id")

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_cls_name = opt.up_conv.module_name if opt.get("up_conv") is not None else None
        self._factory_module = factory_module_cls(
            down_conv_cls_name, up_conv_cls_name, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules.append(down_module)

        # Up modules
        if up_conv_cls_name:
            for i in range(len(opt.up_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv, i, "UP")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_modules.append(up_module)

    def _get_factory(self, model_name, modules_lib) -> BaseFactory:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactory
        return factory_module_cls

    def _fetch_arguments_from_list(self, opt, index):
        """Fetch the arguments for a single convolution from multiple lists
        of arguments - for models specified in the compact format.
        """
        args = {}
        for o, v in opt.items():
            name = str(o)
            if is_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_list(v):
                    v = list(v)
                args[name] = v
        return args

    def _fetch_arguments(self, conv_opt, index, flow):
        """Fetches arguments for building a convolution (up or down)

        Arguments:
            conv_opt
            index in sequential order (as they come in the config)
            flow "UP" or "DOWN"
        """
        args = self._fetch_arguments_from_list(conv_opt, index)
        args["conv_cls"] = self._factory_module.get_module(flow)
        args["index"] = index
        return args

    def _flatten_compact_options(self, opt):
        """Converts from a dict of lists, to a list of dicts"""
        flattenedOpts = []

        for index in range(int(1e6)):
            try:
                flattenedOpts.append(DictConfig(self._fetch_arguments_from_list(opt, index)))
            except IndexError:
                break

        return flattenedOpts

    def forward(self, data, precomputed_down=None, precomputed_up=None, **kwargs):
        """This method does a forward on the Unet assuming symmetrical skip connections

        Parameters
        ----------
        data: torch.geometric.Data
            Data object that contains all info required by the modules
        precomputed_down: torch.geometric.Data
            Precomputed data that will be passed to the down convs
        precomputed_up: torch.geometric.Data
            Precomputed data that will be passed to the up convs
        """
        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=precomputed_down)
            stack_down.append(data)
        data = self.down_modules[-1](data, precomputed=precomputed_down)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)

        sampling_ids = self._collect_sampling_ids(stack_down)

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((data, stack_down.pop()), precomputed=precomputed_up)

        for key, value in sampling_ids.items():
            setattr(data, key, value)
        return data
