"""Functions and classes used to define invertible transformations."""

from copy import deepcopy
from functools import partial
from typing import Literal, Mapping

import numpy as np
import torch as T
import torch.nn as nn
from nflows.transforms import (
    ActNorm,
    AffineCouplingTransform,
    BatchNorm,
    CompositeTransform,
    LULinear,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.transforms.base import Transform
from nflows.transforms.splines.rational_quadratic import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)

from ..modules import DenseNetwork
from ..torch_utils import get_act
from ..utils import key_change

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def change_kwargs_for_made(old_kwargs):
    """Convert a dictionary of keyword arguments.

    Used for configuring kwargs made for a mattstools DenseNetwork to one that can
    initialise a MADE network for the nflows package with similar (not exactly the same)
    hyperparameters.
    """
    new_kwargs = deepcopy(old_kwargs)

    # Certain keys must be changed
    key_change(new_kwargs, "ctxt_dim", "context_features")
    key_change(new_kwargs, "drp", "dropout_probability")
    key_change(new_kwargs, "do_res", "use_residual_blocks")

    # Certain keys are changed and their values modified
    if "act_h" in new_kwargs:
        new_kwargs["activation"] = get_act(new_kwargs.pop("act_h"))
    if "nrm" in new_kwargs:  # MADE only supports batch norm!
        new_kwargs["use_batch_norm"] = new_kwargs.pop("nrm") is not None

    # Some options are missing
    missing = ["ctxt_in_inpt", "ctxt_in_hddn", "n_lyr_pbk", "act_o", "do_out"]
    for miss in missing:
        if miss in new_kwargs:
            del new_kwargs[miss]

    # The hidden dimension passed to MADE as an arg, not a kwarg
    if "hddn_dim" in new_kwargs:
        hddn_dim = new_kwargs.pop("hddn_dim")
    # Otherwise use the same default value for mattstools.modules.DenseNet
    else:
        hddn_dim = 32

    return new_kwargs, hddn_dim


def stacked_ctxt_flow(xz_dim: int, ctxt_dim: int, nstacks: int, transform: partial):
    """Return a composite transform given the config."""
    return CompositeTransform([transform(xz_dim, ctxt_dim) for _ in range(nstacks)])


def stacked_norm_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    nstacks: int = 3,
    param_func: Literal["made", "cplng"] = "cplng",
    invrt_func: Literal["rqs", "aff"] = "aff",
    do_lu: bool = True,
    nrm: str = "none",
    net_kwargs: dict | None = None,
    rqs_kwargs: dict | None = None,
) -> CompositeTransform:
    """Create a stacked flow using a either autoregressive or coupling layers.

    Transform can be either be a rational quadratic spline or an affine layer.

    After each of these transforms, there can be an extra invertible
    linear layer, followed by some normalisation.

    Parameters
    ----------
    xz_dim:
        The number of input X (and output Z) features
    ctxt_dim:
        The dimension of the context feature vector
    nstacks:
        The number of NSF+Perm layers to use in the overall transform
    param_func:
        To use either autoregressive or coupling layers
    invrt_func:
        To use either spline or affine transformations
    do_lu:
        Use an invertible linear layer inbetween splines to encourage mixing
    nrm:
        Do a scale shift normalisation inbetween splines (batch or act)
    net_kwargs:
        Kwargs for the network constructor (includes ctxt dim)
    rqs_kwargs:
        Keyword args for the invertible spline layers
    """

    # Dictionary default arguments (also protecting dict from chaning on save)
    net_kwargs = deepcopy(net_kwargs) or {}
    rqs_kwargs = deepcopy(rqs_kwargs) or {}

    # We add the context dimension to the list of network keyword arguments
    net_kwargs["ctxt_dim"] = ctxt_dim

    # For MADE netwoks change kwargs from mattstools to nflows format
    if param_func == "made":
        made_kwargs, hddn_dim = change_kwargs_for_made(net_kwargs)

    # For coupling layers we need to define a custom network maker function
    elif param_func == "cplng":

        def net_mkr(inpt, outp):
            return DenseNetwork(inpt, outp, **net_kwargs)

    # Start the list of transforms out as an empty list
    trans_list = []

    # Start with a mixing layer
    if do_lu:
        trans_list.append(LULinear(xz_dim))

    # Cycle through each stack
    for i in range(nstacks):
        # For autoregressive funcions
        if param_func == "made":
            if invrt_func == "aff":
                trans_list.append(
                    MaskedAffineAutoregressiveTransform(xz_dim, hddn_dim, **made_kwargs)
                )

            elif invrt_func == "rqs":
                trans_list.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        xz_dim, hddn_dim, **made_kwargs, **rqs_kwargs
                    )
                )

        # For coupling layers
        elif param_func == "cplng":
            # Alternate between masking first half and second half (rounded up)
            mask = T.abs(T.round(T.arange(xz_dim) / (xz_dim - 1)).int() - i % 2)

            if invrt_func == "aff":
                trans_list.append(AffineCouplingTransform(mask, net_mkr))

            elif param_func == "cplng" and invrt_func == "rqs":
                trans_list.append(
                    PiecewiseRationalQuadraticCouplingTransform(
                        mask, net_mkr, **rqs_kwargs
                    )
                )

        # Add the mixing layers
        if do_lu:
            trans_list.append(LULinear(xz_dim))

        # Normalising layers (never on last layer in stack)
        if i < nstacks - 1:
            if nrm == "batch":
                trans_list.append(BatchNorm(xz_dim))
            elif nrm == "act":
                trans_list.append(ActNorm(xz_dim))

    # Return the list of transforms combined
    return CompositeTransform(trans_list)


def make_repeated_transforms(transform: Transform, num_layers: int):
    """Duplicate a transform and return the stacked list."""
    return CompositeTransform([transform] * num_layers)


class ContextSplineTransform(Transform):
    """An invertible transform of a applied elementwise to a tensor."""

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int,
        num_bins: int = 10,
        init_identity: bool = False,
        tails: str | None = None,
        tail_bound: float = 1.0,
        dense_config: Mapping | None = None,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ) -> None:
        """
        Parameters
        ----------
        inpt_dim : int
            The input dimension.
        ctxt_dim : int
            The context dimension.
        num_bins : int, optional
            The number of bins, by default 10.
        init_identity : bool, optional
            Whether to initialize as identity, by default False.
        tails : str or None, optional
            The type of tails to use, either None or linear, by default None.
        tail_bound : float, optional
            The tail bound, by default 1.0.
        dense_config : Mapping or None, optional
            The dense network configuration, by default None.
        min_bin_width : float, optional
            The minimum bin width, by default DEFAULT_MIN_BIN_WIDTH.
        min_bin_height : float, optional
            The minimum bin height, by default DEFAULT_MIN_BIN_HEIGHT.
        min_derivative : float, optional
            The minimum derivative, by default DEFAULT_MIN_DERIVATIVE.
        """

        super().__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.init_identity = init_identity

        self.net = DenseNetwork(
            inpt_dim=ctxt_dim,
            outp_dim=inpt_dim * self._output_dim_multiplier(),
            **(dense_config or {})
        )

        # To be equally spaced with identity mapping
        if init_identity:
            # Cycle through the final dense block and pull out the last linear layer
            for layer in self.net.output_block.block[::-1]:
                if isinstance(layer, nn.Linear):
                    break

            # Set the weights to be zero and change the bias
            T.nn.init.constant_(layer.weight, 0.0)
            T.nn.init.constant_(layer.bias, np.log(np.exp(1 - min_derivative) - 1))

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _process(
        self, inputs: T.Tensor, context: T.Tensor | None = None, inverse: bool = False
    ) -> tuple:
        # Pass through the context extraction network
        spline_params = self.net(context)

        # Save some usefull shapes
        batch_size, features = inputs.shape[:2]

        # Reshape the outputs to be batch x dim x spline_params
        transform_params = spline_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        # Out of the parameters we get the widths, heights, and knot gradients
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        # Select the appropriate function transform
        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        # Apply the spline transform
        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, sum_except_batch(logabsdet)

    def forward(self, inputs: T.Tensor, context: T.Tensor) -> T.Tensor:
        return self._process(inputs, context, inverse=False)

    def inverse(self, inputs: T.Tensor, context: T.Tensor) -> T.Tensor:
        return self._process(inputs, context, inverse=True)


def sum_except_batch(x: T.Tensor, num_batch_dims: int = 1) -> T.Tensor:
    """Sum all elements of x except for the first num_batch_dims dimensions."""
    return T.sum(x, dim=list(range(num_batch_dims, x.ndim)))
