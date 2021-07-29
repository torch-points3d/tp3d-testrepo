import torch

class BaseInternalLossModule(torch.nn.Module):
    """ABC for modules which have internal loss(es)"""

    @abstractmethod
    def get_internal_losses(self) -> Dict[str, Any]:
        pass
