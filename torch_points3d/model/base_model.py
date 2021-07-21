import torch
import torch.nn as nn





class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
         """
        raise NotImplementedError

    

