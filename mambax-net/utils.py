from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


# ──────────────────────────────────────────────────────────────────────────────
# LR scheduler
# ──────────────────────────────────────────────────────────────────────────────

class PolyLRScheduler(LambdaLR):
    """
    Polynomial LR decay — identical to nnUNet's PolyLRScheduler.

        lr = initial_lr × (1 - epoch / max_epochs) ^ exponent

    The LR decreases slowly for most of training and drops sharply near the
    end, which empirically outperforms cosine annealing for segmentation tasks.

    Args:
        optimizer   : the SGD (or any) optimizer
        max_epochs  : total number of training epochs
        exponent    : polynomial exponent (nnUNet default: 0.9)
    """
    def __init__(self, optimizer: optim.Optimizer,
                 max_epochs: int, exponent: float = 0.9):
        self.max_epochs = max_epochs
        self.exponent   = exponent
        super().__init__(optimizer, lr_lambda=self._factor)

    def _factor(self, epoch: int) -> float:
        # epoch is 0-indexed inside LambdaLR
        return (1 - epoch / self.max_epochs) ** self.exponent