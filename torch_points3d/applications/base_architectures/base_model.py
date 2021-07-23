import torch.nn as nn

# This is the base class of the Backbone/API models, that provides useful functions to use within these models
class BaseModel(nn.Module):

    # When creating new tensors (esp sparsetensors), we need to be able to send them to the correct device
    # as ptl won't do it automatically
    @property
    def device(self):
        return next(self.parameters()).device
