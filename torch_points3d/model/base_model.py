import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

from torch_geometric.data import Data





class BaseModel(nn.Module):

    def __init__(self, opt: DictConfig):
        super(BaseModel, self).__init__()
        self.opt = opt

    def set_input(self, data: Data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
         """
        raise NotImplementedError

    

