import logging
from copy import deepcopy
from typing import Mapping, Optional

import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

from .modules import DenseNetwork
from .torch_utils import get_act

log = logging.getLogger(__name__)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.data.zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ResNetBlock(nn.Module):
    """A residual convolutional block that can optionally change the number of
    channels. Can also take in some context tensor.

    Does not resize the spacial dimensions of the input image, all convolutions are
    3x3 kernals with stride 1, padding 1

    Forward pass applies the following:
        (GroupNorm->Conv->Act->Drop) x 2 + skip_connection

    Contextual information is added by taking a linear projection of the context tensor
    to the number of channels and using it to scale and shift the image
    (broadcasted across the spacial dimensions)
    This takes place after the convolutional layers
    """

    def __init__(
        self,
        inpt_channels: int,
        ctxt_dim: int = 0,
        outp_channels: int = None,
        dims: int = 2,
        act: str = "lrlu",
        drp: float = 0,
        nrm_groups: int = 1,
    ) -> None:
        """
        args:
            inpt_channels: the number of input channels.
        kwargs:
            ctxt_dim: The dimension of the conditioning tensor
            outp_channels: The number of output channels, if diff from input
            dims: The dimensions of the signal, 1D, 2D, or 3D
            act: The activation function to use
            drp: The dropout probability
            nrm_groups: Normalisation groups (1=LayerNorm, c=InstanceNorm)
        """
        super().__init__()

        # Class attributes
        self.inpt_channels = inpt_channels
        self.ctxt_dim = ctxt_dim
        self.outp_channels = outp_channels or inpt_channels
        self.dims = dims
        self.act = act
        self.drp = drp
        self.nrm_groups = nrm_groups
        self.has_ctxt = ctxt_dim > 0

        # Create the main layer structure of the network which is split into two parts
        self.first_layers = nn.Sequential(
            nn.GroupNorm(nrm_groups, inpt_channels),
            conv_nd(dims, inpt_channels, outp_channels, 3, padding=1),
            get_act(act),
            nn.GroupNorm(nrm_groups, outp_channels),
        )

        self.second_layers = nn.Sequential(
            get_act(act),
            nn.Dropout(drp),  # Dropout in second layer only
            zero_module(
                conv_nd(dims, outp_channels, outp_channels, 3, padding=1)
            ),  # Start as zeros
        )

        # Create the skip connection, using a 1x1 conv to change channel size
        self.skip_connection = (
            nn.Identity()
            if self.inpt_channels == self.outp_channels
            else conv_nd(dims, inpt_channels, outp_channels, 1)
        )

        # The embedding layers for the contextual information
        if self.has_ctxt:
            self.ctxt_layers = nn.Sequential(
                get_act(act), nn.Linear(ctxt_dim, 2 * self.outp_channels)
            )

    def forward(self, input: T.Tensor, ctxt: T.Tensor = None) -> T.Tensor:
        """Apply the block to a Tensor with optional conditioning
        args:
            input: The import tensor with shape [N x C x ...]
        kwargs:
            ctxt: The optional context tensor with shape [N x ctxt_dim]
        """

        # Pass through the first layers of the module
        output = self.first_layers(input)

        # Introduce the contextual information
        if self.has_ctxt:
            # Check that the context tensor is present
            if ctxt is None:
                raise ValueError("ResNet was expecting a ctxt input but none given")

            # Pass through the layers, size for broadcasting and split int scale/shift
            ctxt = self.ctxt_layers(ctxt)
            ctxt = ctxt.view(ctxt.size() + self.dims * (1,))
            scale, shift = ctxt.chunk(2, 1)
            output = (1 + scale) * output + shift  # scale+1 so doesnt kill on init

        # Pass through the second layers of the module
        output = self.second_layers(output)

        # Return with the skip connection
        return output + self.skip_connection(input)


class MultiHeadedAttentionBlock(nn.Module):
    """A multi-headed self attention block that allows spatial positions to
    attend to each other.

    This layer essentailly flattens the image's spacial dimensions, making it a
    sequence where the length equals the original resolution. The dimension of each
    element of the sequence is the number of each channels.

    Then the message passing occurs, which is permutation invariant, using the exact
    same operations as a standard transformer except we use 1x1 convolutions instead
    of linear projections (same maths, but optimised performance)
    """

    def __init__(self, inpt_channels: int, num_heads: int = 1, nrm_groups: int = 1):
        super().__init__()

        # Ensure that the number of channels is divisible by the number of heads
        assert inpt_channels % num_heads == 0

        # Class attributes
        self.inpt_channels = inpt_channels
        self.num_heads = num_heads
        self.channels_per_head = inpt_channels // num_heads

        # The normalisation layer
        self.norm = nn.GroupNorm(nrm_groups, inpt_channels)

        # QKV are calculated using a 1 dimensional convolution with a 1 kernel size
        # This is equivalent to a 2D conv with 1x1 kernel, but generalises to 3D
        self.qkv = conv_nd(1, inpt_channels, inpt_channels * 3, 1)

        # The final convolutional layer (initiliased with zeros for stability)
        self.out_conv = zero_module(conv_nd(1, inpt_channels, inpt_channels, 1))

    def forward(self, inpt: T.Tensor, _ctxt: T.Tensor = None) -> T.Tensor:
        """Apply the model the message passing, context tensor is not used."""

        # Break up the input image shape into the batch, channels, and spacial
        b, c, *spatial = inpt.shape

        # Flatten each image in each channel dimension
        inpt = inpt.reshape(b, c, -1)

        # Apply the normalisation and get the qkv embeddings for each channel
        qkv = self.qkv(self.norm(inpt))
        q, k, v = T.chunk(qkv, 3, dim=1)  # Split into the component sequences

        # Essentially the images have now all been flattened
        # This means that we now take the image to be a sequence
        # Currently these tensors are of shape [b, feat*heads, seq]
        # Intermediate shape will be [b, head, feat, seq]
        # For the attention operation we need them to be in [b, heads, seq, feat]
        shape = (b, self.num_heads, self.channels_per_head, -1)
        q = q.view(shape).transpose(-1, -2)
        k = k.view(shape).transpose(-1, -2)
        v = v.view(shape).transpose(-1, -2)

        # Now we can use the attention operation from the transformers package
        a_out = scaled_dot_product_attention(q, k, v)

        # Concatenate the all of the heads together to get shape: b,f,seq
        a_out = a_out.transpose(-1, -2).contiguous().view(b, self.inpt_channels, -1)

        # Pass through the final 1x1 convolution layer and
        a_out = self.out_conv(a_out)

        # Apply redidual update and bring back spacial dimensions
        return (a_out + inpt).view(b, -1, *spatial)


class DoublingConvNet(nn.Module):
    """A very simple convolutional neural network which halves the spacial
    dimension with each block while doubling the number of channels.

    Attention operations occur after a certain number of downsampling steps

    Downsampling is performed using 2x2 average pooling

    Ends with a dense network
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        min_size: int = 2,
        attn_below: int = 8,
        start_channels: int = 32,
        max_channels: int = 256,
        resnet_kwargs: dict = None,
        attn_kwargs: dict = None,
        dense_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_size: Spacial dimensions of the inputs
            inpt_channels: Number of channels in the inputs
            outp_dim: Number of output featurs
            ctxt_dim: Size of the contextual tensor
            min_size: The smallest spacial dimensions to reach before flattening
            attn_below: Include attention below this dimension
            resnet_kwargs: Kwargs for the ResNetBlocks
            attn_kwargs: Kwargs for the MultiHeadedAttention block
            dense_kwargs: Kwargs for the dense network
        """
        super().__init__()

        # Save dict defaults
        resnet_kwargs = resnet_kwargs or {}
        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        # Class attributes
        self.inpt_size = inpt_size
        self.inpt_channels = inpt_channels
        self.ctxt_dim = ctxt_dim
        self.outp_dim = outp_dim
        self.min_size = min_size
        self.attn_below = attn_below
        self.start_channels = start_channels
        self.max_channels = max_channels
        self.dims = len(inpt_size)

        # The downsampling layer (not learnable)
        stride = 2 if self.dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(self.dims, kernel_size=stride, stride=stride)

        # The first ResNet block changes to the starting channel dimension
        first_kwargs = deepcopy(resnet_kwargs)
        first_kwargs.nrm_groups = 1
        self.first_block = ResNetBlock(
            inpt_channels=inpt_channels,
            ctxt_dim=ctxt_dim,
            outp_channels=start_channels,
            **first_kwargs,
        )

        # Keep track of the spacial dimensions for each input and output layer
        inp_size = np.array(inpt_size)
        inp_c = start_channels
        out_c = start_channels * 2

        # Start creating the levels (should exit but max 100 for safety)
        resnet_blocks = []
        for _ in range(100):
            lvl_layers = []

            # Add the resnet block
            lvl_layers.append(
                ResNetBlock(
                    inpt_channels=inp_c,
                    ctxt_dim=ctxt_dim,
                    outp_channels=out_c,
                    **resnet_kwargs,
                )
            )

            # Add an optional attention block if we downsampled enough
            if max(inpt_size) <= attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(inpt_channels=out_c, **attn_kwargs)
                )

            # Add the level's layers to the block list
            resnet_blocks.append(nn.ModuleList(lvl_layers))

            # Exit if the next iteration would lead too small spacial dimensions
            if min(inp_size) // 2 <= min_size:
                break

            # Update the dimensions for the next iteration
            inp_size = inp_size // 2  # Halve the spacial dimensions
            inp_c = out_c
            out_c = min(out_c * 2, max_channels)  # Double the channels

        # Combine layers into a module list
        self.resnet_blocks = nn.ModuleList(resnet_blocks)

        # Create the dense network
        self.dense = DenseNetwork(
            inpt_dim=np.prod(inp_size // 2) * out_c,
            outp_dim=outp_dim,
            ctxt_dim=ctxt_dim,
            **dense_kwargs,
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor = None):
        """Forward pass of the network."""

        # Pass through the first convolution layer to embed the channel dimension
        inpt = self.first_block(inpt)

        # Pass through the ResNetBlocks and the downsampling
        for level in self.resnet_blocks:
            for layer in level:
                inpt = layer(inpt, ctxt)
            inpt = self.down_sample(inpt)

        # Flatten and pass through final dense network and return
        inpt = T.flatten(inpt, start_dim=1)

        return self.dense(inpt, ctxt)


class UNet(nn.Module):
    """A image to image mapping network which halves the spacial dimension with
    each block while doubling the number of channels, before building back up
    to the original resolution.

    Attention operations occur after a certain number of downsampling steps

    Downsampling is performed using 2x2 average pooling
    Upsampling is performed using nearest neighbour
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_channels: int,
        ctxt_dim: int = 0,
        min_size: int = 8,
        attn_below: int = 8,
        start_channels: int = 32,
        max_channels: int = 128,
        resnet_kwargs: Optional[Mapping] = None,
        attn_kwargs: Optional[Mapping] = None,
        ctxt_embed_kwargs: Optional[Mapping] = None,
    ) -> None:
        """
        Args:
            inpt_size: Spacial dimensions of the inputs
            inpt_channels: Number of channels in the inputs
            outp_channels: Number of channels in the output image
            ctxt_dim: Size of the contextual tensor
            min_size: The smallest spacial dimensions to reach before flattening
            attn_below: Include attention below this resolution
            resnet_kwargs: Kwargs for the ResNetBlocks
            attn_kwargs: Kwargs for the MultiHeadedAttention block
            ctxt_embed_kwargs: Kwargs for the Dense context embedder
        """
        super().__init__()

        # Save dict defaults (these are modified)
        resnet_kwargs = deepcopy(resnet_kwargs) or {}
        attn_kwargs = deepcopy(attn_kwargs) or {}
        ctxt_embed_kwargs = deepcopy(ctxt_embed_kwargs) or {}

        # Class attributes
        self.inpt_size = inpt_size
        self.inpt_channels = inpt_channels
        self.outp_channels = outp_channels
        self.ctxt_dim = ctxt_dim
        self.min_size = min_size
        self.attn_below = attn_below
        self.start_channels = start_channels
        self.max_channels = max_channels
        self.dims = len(inpt_size)
        self.has_ctxt = ctxt_dim > 0

        # Add the dimensions to the resnet_kwargs
        resnet_kwargs.dims = self.dims

        # If there is a context input, have a network to embed it
        if self.has_ctxt:
            self.context_embedder = DenseNetwork(
                inpt_dim=ctxt_dim,
                **ctxt_embed_kwargs,
            )
            emb_ctxt_size = self.context_embedder.outp_dim
        else:
            emb_ctxt_size = 0

        # The downsampling layer and upscaling layers (not learnable)
        stride = 2 if self.dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(self.dims, kernel_size=stride, stride=stride)
        self.up_sample = nn.Upsample(scale_factor=2)

        # The first ResNet block changes to the starting channel dimension
        first_kwargs = deepcopy(resnet_kwargs)
        first_kwargs.nrm_groups = 1
        self.first_block = ResNetBlock(
            inpt_channels=inpt_channels,
            ctxt_dim=emb_ctxt_size,
            outp_channels=start_channels,
            **first_kwargs,
        )

        # Keep track of the spacial dimensions for each input and output layer
        inp_size = [np.array(inpt_size)]
        inp_c = [start_channels]
        out_c = [start_channels * 2]

        # The encoder blocks are ResNet->(attn)->Downsample
        encoder_blocks = []
        for _ in range(100):
            lvl_layers = []

            # Add the resnet block
            lvl_layers.append(
                ResNetBlock(
                    inpt_channels=inp_c[-1],
                    ctxt_dim=emb_ctxt_size,
                    outp_channels=out_c[-1],
                    **resnet_kwargs,
                )
            )

            # Add an optional attention block if we downsampled enough
            if max(inp_size[-1]) <= attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(inpt_channels=out_c[-1], **attn_kwargs)
                )

            # Add the level's layers to the block list
            encoder_blocks.append(nn.ModuleList(lvl_layers))

            # Exit if the next iteration would lead an output with small spacial dimensions
            if min(inp_size[-1]) // 2 <= min_size:
                break

            # Update the dimensions for the NEXT iteration
            inp_size.append(inp_size[-1] // 2)  # Halve the spacial dimensions
            inp_c.append(out_c[-1])
            out_c.append(min(out_c[-1] * 2, max_channels))  # Double the channels

        # Combine layers into a module list
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # The middle part of the UNet
        self.middle_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    inpt_channels=out_c[-1],
                    ctxt_dim=emb_ctxt_size,
                    outp_channels=out_c[-1],
                    **resnet_kwargs,
                ),
                MultiHeadedAttentionBlock(inpt_channels=out_c[-1], **attn_kwargs),
                ResNetBlock(
                    inpt_channels=out_c[-1],
                    ctxt_dim=emb_ctxt_size,
                    outp_channels=out_c[-1],
                    **resnet_kwargs,
                ),
            ]
        )

        # Loop in reverse to create the decoder blocks
        decoder_blocks = []
        for i in range(1, len(out_c) + 1):
            lvl_layers = []

            # Add the resnet block
            lvl_layers.append(
                ResNetBlock(
                    inpt_channels=out_c[-i] * 2,  # Concatenates across the unet
                    ctxt_dim=emb_ctxt_size,
                    outp_channels=inp_c[-i],
                    **resnet_kwargs,
                )
            )

            # Add the attention layer at the appropriate levels
            if max(inp_size[-i]) <= attn_below:
                lvl_layers.append(
                    MultiHeadedAttentionBlock(inpt_channels=inp_c[-i], **attn_kwargs)
                )

            # Add the level's layers to the block list
            decoder_blocks.append(nn.ModuleList(lvl_layers))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        # Final block in maps to the number of output channels
        last_kwargs = deepcopy(resnet_kwargs)
        last_kwargs.nrm_groups = 1
        last_kwargs.drp = 0  # No dropout on final block! These are the outputs!
        self.last_block = ResNetBlock(
            inpt_channels=start_channels,
            ctxt_dim=emb_ctxt_size,
            outp_channels=outp_channels,
            **last_kwargs,
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor = None):
        """Forward pass of the network."""

        # Some context tensors come from labels and must match the same type as inpt
        if ctxt.dtype != inpt.dtype:
            ctxt = ctxt.type(inpt.dtype)

        # Make sure the input size is expected
        if inpt.shape[-1] != self.inpt_size[-1]:
            log.warning("Input image does not match the training sample!")

        # Embed the context tensor
        ctxt = self.context_embedder(ctxt)

        # Pass through the first convolution layer to embed the channel dimension
        inpt = self.first_block(inpt, ctxt)

        # Pass through the encoder
        dec_outs = []
        for level in self.encoder_blocks:
            for layer in level:
                inpt = layer(inpt, ctxt)
            dec_outs.append(inpt)  # Save the output to the buffer
            inpt = self.down_sample(inpt)  # Apply the downsampling

        # Pass through the middle blocks
        for block in self.middle_blocks:
            inpt = block(inpt, ctxt)

        # Pass through the decoder blocks
        for level in self.decoder_blocks:
            inpt = self.up_sample(inpt)  # Apply the upsampling
            inpt = T.cat([inpt, dec_outs.pop(-1)], dim=1)  # Concat with buffer
            for layer in level:
                inpt = layer(inpt, ctxt)

        # Pass through the final layer
        inpt = self.last_block(inpt, ctxt)

        return inpt
