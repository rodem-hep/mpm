"""Custom loss functions and methods to calculate them."""

from typing import Tuple, Union

import torch as T
import torch.nn as nn
from geomloss import SamplesLoss

# from jetnet.losses import EMDLoss


class VAELoss(nn.Module):
    """The Kullback-Leibler divergence to unit normal loss used for VAEs."""

    def forward(self, means: T.Tensor, log_stds: T.Tensor) -> T.Tensor:
        """
        args:
            means: The set of mean values
            log_stds: The natural logarithm (base e) of the standard deviations
        returns:
            loss per sample (no batch reduction)
        """
        return kld_to_norm(means, log_stds)


def champfer_loss(mask_a, pc_a, mask_b, pc_b):
    """Returns the champfer loss between two point clouds with different
    cardinality and masking."""

    # Calculate the distance matrix (squared) between the outputs and targets
    matrix_mask = mask_a.bool().unsqueeze(-1) & mask_b.bool().unsqueeze(-2)
    dist_matrix = T.cdist(pc_a, pc_b)

    # Ensure the distances between fake nodes take the padding value
    dist_matrix = dist_matrix.masked_fill(~matrix_mask, 1e8)

    # Get the sum of the minimum along each axis, square, and scale by the weights
    min1 = T.min(dist_matrix, dim=-1)[0] ** 2 * mask_a  # Zeros out the padded
    min2 = T.min(dist_matrix, dim=-2)[0] ** 2 * mask_b

    # Add the two metrics together (no batch reduction)
    return T.sum(min1, dim=-1) + T.sum(min2, dim=-1)


def kld_to_norm(means: T.Tensor, log_stds: T.Tensor, reduce="mean") -> T.Tensor:
    """Calculate the KL-divergence to a unit normal distribution."""
    loss = 0.5 * (means * means + (2 * log_stds).exp() - 2 * log_stds - 1)
    if reduce == "mean":
        return loss.mean()
    if reduce == "dim_mean":
        return loss.mean(dim=-1)
    if reduce == "sum":
        return loss.sum()
    if reduce == "none":
        return loss
    raise RuntimeError(f"Unrecognized reduction arguments: {reduce}")


def kld_with_OE(
    means: T.Tensor, log_stds: T.Tensor, labels=T.Tensor, reduce="mean"
) -> T.Tensor:
    """Calculate the KL-divergence to a unit normal distribution."""
    loss = kld_to_norm(means, log_stds, reduce="none")

    # All classes not equal to zero have the loss terms flipped (maximise loss)
    loss = loss * (1 - 2 * labels).unsqueeze(-1)
    loss = T.clamp_min(loss, -3)

    if reduce == "mean":
        return loss.mean()
    if reduce == "dim_mean":
        return loss.mean(dim=-1)
    if reduce == "sum":
        return loss.sum()
    if reduce == "none":
        return loss
    raise RuntimeError(f"Unrecognized reduction arguments: {reduce}")


class GeomWrapper(nn.Module):
    """This is a wrapper class for the geomloss package which by default
    renables all gradients after a forward pass, thereby causing the gradient
    memory to explode during evaluation."""

    def __init__(self, loss_fn) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs) -> T.Tensor:
        """Return the loss."""
        current_grad_state = T.is_grad_enabled()
        loss = self.loss_fn(*args, **kwargs)
        T.set_grad_enabled(current_grad_state)
        return loss


class MyBCEWithLogit(nn.Module):
    """A wrapper for the calculating BCE using logits in pytorch that makes the
    syntax consistant with pytorch's CrossEntropy loss.

    - Automatically squeezes out the batch dimension to ensure same shape
    - Automatically changes targets to floats

    Vanilla BCE wants identical shapes (batch x output)
    While CE loss wants targets just as indices (batch)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, outputs, targets):
        """Return the loss."""
        return self.loss_fn(outputs.squeeze(dim=-1), targets.float())


class ModifiedSinkhorn(nn.Module):
    def __init__(self) -> None:
        """Applies four different sinkhorn loss functions:

        1)  PT Weighted sinkhorn loss of eta/phi point cloud (main one)
        2)  Champfer loss on eta/phi point cloud (no weights) 3)
        Champfer loss on sinkhorn marginal 4)  Huber loss on all total
        Pt
        """
        super().__init__()
        self.snk = GeomWrapper(SamplesLoss(loss="sinkhorn"))
        self.w = 1

    def forward(self, a_mask, pc_a, b_mask, pc_b):
        a_pt = a_mask * pc_a[..., -1]
        b_pt = b_mask * pc_b[..., -1]
        a_etaphi = pc_a[..., :-1]
        b_etaphi = pc_b[..., :-1]
        loss = (
            self.snk(a_pt, a_etaphi.detach(), b_pt, b_etaphi.detach())
            + self.snk(a_mask, a_etaphi, b_mask, b_etaphi)
            + self.snk(a_mask, a_pt.unsqueeze(-1), b_mask, b_pt.unsqueeze(-1))
        )
        return loss


class EnergyMovers(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.loss_fn = EMDLoss(**kwargs)

    def forward(
        self, a_weights, pc_a, b_weights, pc_b
    ) -> Union[T.Tensor, Tuple[T.Tensor, T.Tensor]]:
        pc_a = T.masked_fill(pc_a, (a_weights == 0).unsqueeze(-1), 0)
        pc_b = T.masked_fill(pc_b, (b_weights == 0).unsqueeze(-1), 0)
        return self.loss_fn(pc_a, pc_b)


class ChampferLoss(nn.Module):
    """The champfer loss function for use on batched and weighted
    pointclouds."""

    def forward(self, o_weights, outputs, t_weights, targets):
        """Constructor method for ChampferLoss
        args:
            o_weights: The output point cloud weights
            outputs: The output point cloud features
            t_weights: The target point cloud weights
            targets: The target point cloud features
        returns:
            loss per sample (no batch reduction)
        """

        # Calculate the distance matrix (squared) between the outputs and targets
        return champfer_loss(o_weights, outputs, t_weights, targets)


def masked_dist_loss(
    loss_fn: nn.Module,
    pc_a: T.Tensor,
    pc_a_mask: T.BoolTensor,
    pc_b: T.Tensor,
    pc_b_mask: T.BoolTensor,
    reduce: str = "mean",
) -> T.Tensor:
    """Calculates the distribution loss between two masked pointclouds.

    - This is done by using the masks as weights (compatible with the geomloss package)
    - The loss function should be permutation invariant

    args:
        loss_fn: The loss function to apply, must have a forward method
        pc_a: The first point cloud
        pc_a_mask: The mask of the first point cloud
        pc_b: The second point cloud
        pc_b_mask: The mask of the second point cloud
    kwargs:
        reduce: If the loss should be reduced along the batch dimension
    """

    # Calculate the weights by normalising the mask for each sample
    a_weights = pc_a_mask.float()
    b_weights = pc_b_mask.float()

    # Calculate the loss using these weights
    loss = loss_fn(a_weights, pc_a, b_weights, pc_b)

    if reduce == "mean":
        return loss.mean()
    if reduce == "none":
        return loss
    raise ValueError("Unknown reduce option for masked_dist_loss")
