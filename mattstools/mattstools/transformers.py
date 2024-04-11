"""Some classes to describe transformer architectures."""

import math
from typing import Mapping, Optional, Union

import torch as T
import torch.nn as nn
from torch.nn.functional import dropout, scaled_dot_product_attention, softmax

from .modules import DenseNetwork


def merge_masks(
    kv_mask: Union[T.BoolTensor, None],
    attn_mask: Union[T.BoolTensor, None],
    attn_bias: Union[T.Tensor, None],
    query: T.Size,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding information
    and the bias terms.

    New philosophy is just to define a kv_mask, and let the q_mask be
    ones. Let the padded nodes receive what they want! Their outputs
    dont matter and they don't add to computation anyway!!!
    """

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if kv_mask is not None:
        q_mask = T.full(query.shape[:-1], True, device=query.device)  # Always full
        merged_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

    # If attention mask exists, create
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    # If the attention bias exists, convert to a float and add
    if (attn_bias is not None) and (attn_bias.numel() > 0):
        merged_mask = (~merged_mask).type(query.dtype) * -1e9
        merged_mask = merged_mask + attn_bias.permute(0, 3, 1, 2)

    return merged_mask


def my_scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: Optional[T.BoolTensor] = None,
    attn_bias: Optional[T.Tensor] = None,
    dropout_p: float = 0.0,
) -> T.Tensor:
    """DEPRECATED! THE PYTORCH-2.0 IMPLEMENATION IS 25% FASTER AND HAS A
    REDUCED MEMORY OVERHEAD SO MY ATTENTION LAYERS HAVE SWITCHED OVER TO
    THAT!!!

    Apply the attention using the scaled dot product between the key query
    and key tensors, then matrix multiplied by the value.

    Note that the attention scores are ordered in recv x send, which is the opposite
    to how I usually do it for the graph network, which is send x recv

    We use masked fill -T.inf as this kills the padded key/values elements but
    introduces nans for padded query elements. We could used a very small number like
    -1e9 but this would need to scale with if we are using half precision.

    Args:
        query: Batched query sequence of tensors (b, h, s, f)
        key: Batched key sequence of tensors (b, h, s, f)
        value: Batched value sequence of tensors (b, h, s, f)
        attn_mask: The attention mask, used to blind certain combinations of k,q pairs
        attn_bias: Extra weights to combine with attention weights
        drp: Dropout probability
    """
    DeprecationWarning("Dont use this! Switch to pytorch 2.0 built in version!")

    # Perform the matrix multiplication
    scores = query @ key.transpose(-2, -1) / math.sqrt(key.shape[-1])

    # Add the bias terms if present
    if attn_bias is not None:  # Move the head dimension to the first
        scores = scores + attn_bias

    # Mask away the scores between invalid elements in sequence
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, -T.inf)

    # Apply the softmax function per head feature
    scores = softmax(scores, dim=-1)

    # Kill the nans introduced by the padded query elements
    if attn_mask is not None:
        scores = T.nan_to_num(scores)

    # Apply dropout to the attention scores
    scores = dropout(scores, p=dropout_p)

    # Finally multiply these scores by the output
    scores = scores @ value

    return scores


class MultiHeadedAttentionBlock(nn.Module):
    """Generic Multiheaded Attention.

    Takes in three sequences with dim: (batch, sqeuence, features)
    - q: The primary sequence queries (determines output sequence length)
    - k: The attending sequence keys (determines incoming information)
    - v: The attending sequence values

    In a message passing sense you can think of q as your receiver nodes, v and k
    are the information coming from the sender nodes.

    When q == k(and v) this is a SELF attention operation
    When q != k(and v) this is a CROSS attention operation

    ===

    Block operations:

    1) Uses three linear layers to project the sequences.
    - q = q_linear * q
    - k = k_linear * k
    - v = v_linear * v

    2) Outputs are reshaped to add a head dimension, and transposed for matmul.
    - features = model_dim = head_dim * num_heads
    - dim becomes: batch, num_heads, sequence, head_dim

    3) Passes these through to the attention module (message passing)
    - In standard transformers this is the scaled dot product attention
    - Also takes additional dropout layer to mask the attention

    4) Flatten out the head dimension and pass through final linear layer
    - results are same as if attention was done seperately for each head and concat
    - dim: batch, q_seq, head_dim * num_heads
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
        drp: float = 0,
        init_zeros: bool = False,
        do_casual: bool = False,
        do_layer_norm: bool = True,
    ) -> None:
        """
        Args:
            model_dim: The dimension of the model
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
            drp: The dropout probability used in the MHA operation
            init_zeros: If the final linear layer is initialised with zero weights
            do_casual: Casual attention should only be used if the q, k, v are the same
                Slightly faster matrix multiplication at the beginning
            do_layer_norm: If a layernorm is applied before the output final linear
                projection
        """
        super().__init__()

        # Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.do_casual = do_casual
        self.drp = drp
        self.do_layer_norm = do_layer_norm

        # Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        # Initialise the weight matrices (only 1 for do casual)
        if do_casual:
            self.all_linear = nn.Linear(model_dim, 3 * model_dim)
        else:
            self.q_linear = nn.Linear(model_dim, model_dim)
            self.k_linear = nn.Linear(model_dim, model_dim)
            self.v_linear = nn.Linear(model_dim, model_dim)

        # The optional (but advised) layer normalisation
        if do_layer_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

        # Set the output linear layer weights and bias terms to zero
        self.out_linear = nn.Linear(model_dim, model_dim)
        if init_zeros:
            self.out_linear.weight.data.fill_(0)
            self.out_linear.bias.data.fill_(0)

    def forward(
        self,
        q: T.Tensor,
        k: Optional[T.Tensor] = None,
        v: Optional[T.Tensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        attn_bias: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """
        Args:
            q: The main sequence queries (determines the output length)
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
            attn_bias: Extra bias term for the attention matrix (eg: edge features)
        """

        # Store the batch size, useful for reshaping
        b_size, seq, feat = q.shape

        # If only q and q_mask are provided then we automatically apply self attention
        if k is None:
            k = q
        if v is None:
            v = k

        # Work out the masking situation, with padding, no peaking etc
        attn_mask = merge_masks(kv_mask, attn_mask, attn_bias, q)

        # Generate the q, k, v projections
        if self.do_casual:
            q_out, k_out, v_out = self.all_linear(q).chunk(3, -1)
        else:
            q_out = self.q_linear(q)
            k_out = self.k_linear(k)
            v_out = self.v_linear(v)

        # Break final dim, transpose to get dimensions: B,H,Seq,Hdim
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q_out = q_out.view(shape).transpose(1, 2)
        k_out = k_out.view(shape).transpose(1, 2)
        v_out = v_out.view(shape).transpose(1, 2)

        # Calculate the new sequence values
        a_out = scaled_dot_product_attention(
            q_out,
            k_out,
            v_out,
            attn_mask=attn_mask,
            dropout_p=self.drp if self.training else 0,
        )

        # Concatenate the all of the heads together to get shape: B,Seq,F
        a_out = a_out.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        # Pass through the optional normalisation layer
        if self.do_layer_norm:
            a_out = self.layer_norm(a_out)

        # Pass through final linear layer
        return self.out_linear(a_out)


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style
    arcitecture.

    We choose Normformer as it has often proved to be the most stable to train
    https://arxiv.org/abs/2210.06423
    https://arxiv.org/abs/2110.09456

    It contains:
    - Multihead(self)Attention block
    - A dense network

    Layernorm is applied before each operation
    Residual connections are used to bypass each operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimension of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(
            model_dim, do_casual=True, **mha_config
        )
        self.dense = DenseNetwork(
            model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config
        )

        # The pre MHA and pre FFN layer normalisations
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.self_attn(
            self.norm1(x), kv_mask=mask, attn_mask=attn_mask, attn_bias=attn_bias
        )
        x = x + self.dense(self.norm2(x), ctxt=ctxt)
        return x


class TransformerDecoderLayer(nn.Module):
    """A transformer dencoder layer based on the GPT-2+Normformer style
    arcitecture.

    It contains:
    - self-attention-block
    - cross-attention block
    - dense network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer

    Attnention masks and biases are only applied to the self attention operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(
            model_dim, do_casual=True, **mha_config
        )
        self.cross_attn = MultiHeadedAttentionBlock(
            model_dim, do_casual=False, **mha_config
        )
        self.dense = DenseNetwork(
            model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config
        )

        # The pre_operation normalisation layers (lots from Foundation Transformers)
        self.norm_preSA = nn.LayerNorm(model_dim)
        self.norm_preC1 = nn.LayerNorm(model_dim)
        self.norm_preC2 = nn.LayerNorm(model_dim)
        self.norm_preNN = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        q_mask: Optional[T.BoolTensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"

        # Apply the self attention residual update
        q_seq = q_seq + self.self_attn(
            self.norm_preSA(q_seq),
            kv_mask=q_mask,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
        )

        # Apply the cross attention residual update
        q_seq = q_seq + self.cross_attn(
            q=self.norm_preC1(q_seq), k=self.norm_preC2(kv_seq)
        )

        # Apply the dense residual update
        q_seq = q_seq + self.dense(self.norm_preNN(q_seq), ctxt)

        return q_seq


class TransformerCrossAttentionLayer(nn.Module):
    """A transformer cross attention layer.

    It contains:
    - cross-attention-block
    - A feed forward network

    Does not allow for attn masks/biases
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimension of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.cross_attn = MultiHeadedAttentionBlock(
            model_dim, do_casual=False, **mha_config
        )
        self.dense = DenseNetwork(
            model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config
        )

        # The two pre MHA and pre FFN layer normalisations
        self.norm0 = nn.LayerNorm(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        kv_mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        q_seq = q_seq + self.cross_attn(
            self.norm1(q_seq), self.norm0(kv_seq), kv_mask=kv_mask
        )
        q_seq = q_seq + self.dense(self.norm2(q_seq), ctxt)

        return q_seq


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final
    normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """A stack of N transformer dencoder layers followed by a final
    normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context input
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, q_seq: T.Tensor, kv_seq: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            q_seq = layer(q_seq, kv_seq, **kwargs)
        return self.final_norm(q_seq)


class TransformerVectorEncoder(nn.Module):
    """A type of transformer encoder which procudes a single vector for the
    whole seq.

    Sequence -> Vector

    Then the sequence (and optionally edges) are passed through several MHSA layers.
    Then a learnable class token is updated using cross attention.
    This results in a single element sequence.
    Contains a final normalisation layer

    It is non resizing, so model_dim must be used for inputs and outputs
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_sa_layers: int = 2,
        num_ca_layers: int = 2,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature size for input, output, and all intermediate sequences
            num_sa_layers: Number of self attention encoder layers
            num_ca_layers: Number of cross/class attention encoder layers
            mha_config: Keyword arguments for all multiheaded attention layers
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_sa_layers = num_sa_layers
        self.num_ca_layers = num_ca_layers

        self.sa_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_sa_layers)
            ]
        )
        self.ca_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim, mha_config, dense_config, ctxt_dim
                )
                for _ in range(num_ca_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

        # Initialise the class token embedding as a learnable parameter
        self.class_token = nn.Parameter(T.randn((1, 1, self.model_dim)))

    def forward(
        self,
        seq: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        return_seq: bool = False,
    ) -> Union[T.Tensor, tuple]:
        """Pass the input through all layers sequentially."""

        # Pass through the self attention encoder
        for layer in self.sa_layers:
            seq = layer(seq, mask, attn_bias=attn_bias, attn_mask=attn_mask, ctxt=ctxt)

        # Get the learned class token and expand to the batch size
        # Use shape not len as it is ONNX safe!
        class_token = self.class_token.expand(seq.shape[0], 1, self.model_dim)

        # Pass through the class attention layers
        for layer in self.ca_layers:
            class_token = layer(class_token, seq, kv_mask=mask, ctxt=ctxt)

        # Pass through the final normalisation layer
        class_token = self.final_norm(class_token)

        # Pop out the unneeded sequence dimension of 1
        class_token = class_token.squeeze(1)

        # Return the class token and optionally the sequence as well
        if return_seq:
            return class_token, seq
        return class_token


class TransformerVectorDecoder(nn.Module):
    """A type of transformer decoder which creates a sequence given a starting
    vector and a desired mask.

    Vector -> Sequence

    Randomly initialises the q-sequence using the mask shape and a gaussian
    Uses the input vector as 1-long kv-sequence in decoder layers

    It is non resizing, so model_dim must be used for inputs and outputs
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 2,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of decoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(
        self, vec: T.Tensor, mask: T.BoolTensor, ctxt: Optional[T.Tensor] = None
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""

        # Initialise the q-sequence randomly (adhere to mask)
        q_seq = T.randn(
            (*mask.shape, self.model_dim), device=vec.device, dtype=vec.dtype
        ) * mask.unsqueeze(-1)

        # Reshape the vector from batch x features to batch x seq=1 x features
        vec = vec.unsqueeze(1)

        # Pass through the decoder
        for layer in self.layers:
            q_seq = layer(q_seq, vec, kv_mask=mask, ctxt=ctxt)
        return self.final_norm(q_seq)


class FullTransformerVectorEncoder(nn.Module):
    """A TVE with added input and output dense embedding networks.

    Sequence -> Vector

    1)  Embeds the squence into a higher dimensional space based on model_dim
        using a dense network.
    2)  If there are edge features these are projected into space = n_heads
            This is a very optional step which most will want to ignore but it is what
            ParT used! https://arxiv.org/abs/2202.03772
    3)  Then it passes these through a TVE to get a single vector output
    4)  Finally is passes the vector through an embedding network
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        edge_dim: int = 0,
        ctxt_dim: int = 0,
        tve_config: Optional[Mapping] = None,
        node_embd_config: Optional[Mapping] = None,
        outp_embd_config: Optional[Mapping] = None,
        edge_embd_config: Optional[Mapping] = None,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of of the final output vector
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            edge_dim: Dim. of the input edge features
            tve_config: Keyword arguments to pass to the TVE constructor
            node_embd_config: Keyword arguments for node dense embedder
            outp_embd_config: Keyword arguments for output dense embedder
            edge_embd_config: Keyword arguments for edge dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim
        tve_config = tve_config or {}
        node_embd_config = node_embd_config or {}
        outp_embd_config = outp_embd_config or {}
        edge_embd_config = edge_embd_config or {}

        # Initialise the TVE, the main part of this network
        self.tve = TransformerVectorEncoder(**tve_config, ctxt_dim=ctxt_dim)
        self.model_dim = self.tve.model_dim

        # Initialise all node (inpt) and vector (output) embedding network
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_dim,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **outp_embd_config,
        )

        # Initialise the edge embedding network (optional)
        if self.edge_dim:
            self.edge_embd = DenseNetwork(
                inpt_dim=self.edge_dim,
                outp_dim=self.tve.sa_layers[0].self_attn.num_heads,
                ctxt_dim=self.ctxt_dim,
                **edge_embd_config,
            )

    def forward(
        self,
        seq: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        return_seq: bool = False,
    ) -> Union[T.Tensor, tuple]:
        """Pass the input through all layers sequentially."""

        # Embed the sequence
        seq = self.node_embd(seq, ctxt)

        # Embed the attention bias (edges, optional)
        if attn_bias is not None:
            attn_bias = self.edge_embd(attn_bias, ctxt)

        # Pass throught the tve
        output = self.tve(
            seq,
            mask,
            ctxt=ctxt,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            return_seq=return_seq,
        )

        # If we had asked to return both, then split before embedding
        if return_seq:
            output, seq = output
        output = self.outp_embd(output, ctxt)

        # Embed the output vector and return
        if return_seq:
            return output, seq
        return output


class FullTransformerVectorDecoder(nn.Module):
    """A TVD with added input and output embedding networks.

    Vector -> Sequence

    1)  Embeds the input vector into a higher dimensional space based on model_dim
        using a dense network.
    2)  Passes this through a TVD to get a sequence output
    3)  Passes the sequence through an embedding dense network with vector as context
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        tvd_config: Optional[Mapping] = None,
        vect_embd_config: Optional[Mapping] = None,
        outp_embd_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of the input vector
            outp_dim: Dim. of each element of the output sequence
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            tvd_config: Keyword arguments to pass to the TVD constructor
            vec_embd_config: Keyword arguments for vector dense embedder
            out_embd_config: Keyword arguments for output node dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        tvd_config = tvd_config or {}
        vect_embd_config = vect_embd_config or {}
        outp_embd_config = outp_embd_config or {}

        # Initialise the TVE, the main part of this network
        self.tvd = TransformerVectorDecoder(**tvd_config)
        self.model_dim = self.tvd.model_dim

        # Initialise all embedding networks
        self.vec_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_dim,
            **vect_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **outp_embd_config,
        )

    def forward(
        self, vec: T.Tensor, mask: T.BoolTensor, ctxt: Optional[T.Tensor] = None
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        vec = self.vec_embd(vec, ctxt=ctxt)
        seq = self.tvd(vec, mask, ctxt=ctxt)
        seq = self.outp_embd(seq, ctxt)
        seq = T.masked_fill(seq, ~mask.unsqueeze(-1), 0)  # Force zero padding
        return seq


class FullTransformerEncoder(nn.Module):
    """A transformer encoder with added input and output embedding networks.

    Sequence -> Sequence
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        edge_dim: int = 0,
        ctxt_dim: int = 0,
        te_config: Optional[Mapping] = None,
        node_embd_config: Optional[Mapping] = None,
        outp_embd_config: Optional[Mapping] = None,
        edge_embd_config: Optional[Mapping] = None,
        ctxt_embd_config: Optional[Mapping] = None,
    ) -> None:
        """
        Args:
            inpt_dim: Dim. of each element of the sequence
            outp_dim: Dim. of of the final output vector
            edge_dim: Dim. of the input edge features
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            te_config: Keyword arguments to pass to the TVE constructor
            node_embd_config: Keyword arguments for node dense embedder
            outp_embd_config: Keyword arguments for output dense embedder
            edge_embd_config: Keyword arguments for edge dense embedder
            ctxt_embd_config: Keyword arguments for context dense embedder
        """
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.edge_dim = edge_dim
        te_config = te_config or {}
        node_embd_config = node_embd_config or {}
        outp_embd_config = outp_embd_config or {}
        edge_embd_config = edge_embd_config or {}
        ctxt_embd_config = ctxt_embd_config or {}

        # Initialise the context embedding network (optional)
        if self.ctxt_dim:
            self.ctxt_emdb = DenseNetwork(
                inpt_dim=self.ctxt_dim,
                **ctxt_embd_config,
            )
            self.ctxt_out = self.ctxt_emdb.outp_dim
        else:
            self.ctxt_out = 0

        # Initialise the TVE, the main part of this network
        self.te = TransformerEncoder(**te_config, ctxt_dim=self.ctxt_out)
        self.model_dim = self.te.model_dim

        # Initialise all embedding networks
        self.node_embd = DenseNetwork(
            inpt_dim=self.inpt_dim,
            outp_dim=self.model_dim,
            ctxt_dim=self.ctxt_out,
            **node_embd_config,
        )
        self.outp_embd = DenseNetwork(
            inpt_dim=self.model_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_out,
            **outp_embd_config,
        )

        # Initialise the edge embedding network (optional)
        if self.edge_dim:
            self.edge_embd = DenseNetwork(
                inpt_dim=self.edge_dim,
                outp_dim=self.te.layers[0].self_attn.num_heads,
                ctxt_dim=self.ctxt_out,
                **edge_embd_config,
            )

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        if self.ctxt_dim:
            ctxt = self.ctxt_emdb(ctxt)
        if self.edge_dim:
            attn_bias = self.edge_embd(attn_bias, ctxt)
        x = self.node_embd(x, ctxt=ctxt)
        x = self.te(x, mask=mask, ctxt=ctxt, attn_bias=attn_bias, attn_mask=attn_mask)
        x = self.outp_embd(x, ctxt=ctxt)
        return x


class ClassAttention(nn.Module):
    """Pooling operation for a sequence using class attention.

    Sequence -> Vector
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_ca_layers: int = 2,
        n_out: int = 1,
        mha_config: Mapping | None = None,
        dense_config: Mapping | None = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature size for input, output, and all intermediate sequences
            num_sa_layers: Number of self attention encoder layers
            num_ca_layers: Number of cross/class attention encoder layers
            n_out: Length of the output vector
            mha_config: Keyword arguments for all multiheaded attention layers
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_ca_layers = num_ca_layers

        self.ca_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim, mha_config, dense_config, ctxt_dim
                )
                for _ in range(num_ca_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

        # Initialise the class token embedding as a learnable parameter
        self.class_token = nn.Parameter(T.randn((1, 1, self.model_dim)))
        self.output_layer = nn.Linear(model_dim, n_out)

    def forward(
        self,
        seq: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass the input through all layers sequentially."""

        # Expand the class token and to the batch size
        class_token = self.class_token.expand(seq.shape[0], 1, self.model_dim)

        # Pass through the class attention layers
        for layer in self.ca_layers:
            class_token = layer(class_token, seq, kv_mask=mask, ctxt=ctxt)

        # Pass through the final normalisation layer
        class_token = self.final_norm(class_token)

        # Pop out the unneeded sequence dimension of 1
        return self.output_layer(class_token.squeeze(1))
