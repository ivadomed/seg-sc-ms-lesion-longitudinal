import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss, background excluded (nnU-Net's do_bg=False).

    `batch_dice` (nnU-Net 3d_fullres default, read from plans.json): when True,
    the whole batch is treated as a single volume — intersection/union are summed
    over the batch dim before dividing. This is far more stable than the per-
    sample average for sparse lesions, where individual patches may be empty.
    """
    def __init__(self, n_classes: int, smooth: float = 1e-5, batch_dice: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(preds, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1).long(), 1.0)

        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        tgt_flat   = targets_oh.view(*probs_flat.shape)

        if self.batch_dice:
            # Sum over batch (0) and spatial (-1) → one Dice per class.
            intersection = (probs_flat * tgt_flat).sum(dim=(0, -1))
            union        = probs_flat.sum(dim=(0, -1)) + tgt_flat.sum(dim=(0, -1))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice[1:].mean()

        intersection = (probs_flat * tgt_flat).sum(-1)
        union        = probs_flat.sum(-1) + tgt_flat.sum(-1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice[:, 1:].mean()


class CombinedLoss(nn.Module):
    def __init__(self, n_classes: int, batch_dice: bool = True):
        super().__init__()
        self.dice = DiceLoss(n_classes, batch_dice=batch_dice)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(preds, targets) + self.ce(preds, targets.long())


class DeepSupervisionLoss(nn.Module):
    """Apply `loss` at every deep-supervision scale and return the weighted sum.

    Mirrors nnU-Net's DeepSupervisionWrapper: the network emits one prediction
    per decoder resolution (highest-res first); the target is downsampled
    (nearest) to each prediction's spatial size, and the per-scale losses are
    combined with `weights`. Scales whose weight is 0 (nnU-Net zeroes the
    coarsest resolution) are skipped. A single-tensor input (e.g. validation /
    inference) is passed straight through to `loss`.
    """
    def __init__(self, loss: nn.Module, weights):
        super().__init__()
        self.loss = loss
        # Plain Python floats: numpy scalars (np.float64) * torch.Tensor can
        # return a numpy object instead of a tensor, which breaks autograd/AMP.
        self.weights = [float(w) for w in weights]

    def forward(self, preds, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(preds, (list, tuple)):
            return self.loss(preds, target)

        total = 0.0
        for pred, w in zip(preds, self.weights):
            if w == 0:
                continue
            if pred.shape[2:] == target.shape[1:]:
                t = target
            else:
                t = F.interpolate(target.unsqueeze(1).float(),
                                  size=pred.shape[2:], mode="nearest").squeeze(1)
            total = total + w * self.loss(pred, t)
        return total