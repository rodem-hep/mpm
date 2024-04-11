import math

import torch as T
import torch.nn as nn
from torch.nn.functional import softmax


def merge_masks(
    q_mask: T.BoolTensor,
    kv_mask: T.BoolTensor,
    attn_mask: T.Tensor,
    q: T.Tensor,
    k: T.Tensor,
) -> T.Tensor:
    """Create a full attention mask which incoporates the padding
    information."""

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        q_mask = (
            q_mask
            if q_mask is not None
            else T.full(q.shape[:-1], True, dtype=T.bool, device=q.device)
        )
        kv_mask = (
            kv_mask
            if kv_mask is not None
            else T.full(k.shape[:-1], True, dtype=T.bool, device=k.device)
        )
        pad_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

        # Combine with the attention mask depending on the dtype
        if attn_mask is None:
            attn_mask = pad_mask
        elif attn_mask.dtype == T.bool:
            attn_mask = attn_mask.logical_and(pad_mask)
        else:
            attn_mask = attn_mask.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))

    # If attention mask exists, convert to float format
    if attn_mask is not None and attn_mask.dtype == T.bool:
        new_attn_mask = T.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(~attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    return attn_mask


def attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    dim_key: int,
    attn_mask: T.Tensor = None,
) -> T.Tensor:
    """Apply the attention using the scaled dot product between the key query
    and key tensors, then matrix multiplied by the value.

    Note that the attention scores are ordered in recv x send, which is the opposite
    to how I usually do it for the graph network, which is send x recv

    Args:
        query: Batched query sequence of tensors (b, h, s, f)
        key: Batched key sequence of tensors (b, h, s, f)
        value: Batched value sequence of tensors (b, h, s, f)
        dim_key: The dimension of the key features, used to scale the dot product
        attn_mask: The attention mask, used to blind certain combinations of k,q pairs
        attn_bias: Extra weights to combine with attention weights
    """

    # Perform the matrix multiplication (bmm is faster than matmul on gpus)
    query = query / math.sqrt(dim_key)
    if attn_mask is not None:
        scores = T.baddbmm(attn_mask, query, key.transpose(-2, -1))
    else:
        scores = T.bmm(query, key.transpose(-2, -1))

    # Apply the softmax function per head feature
    scores = softmax(scores, dim=-1)

    # Finally multiply these scores by the output
    scores = T.bmm(scores, value)

    return scores


class NewHeadedAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        # Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        # Initialise the weight matrices
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        q: T.Tensor,
        k: T.Tensor = None,
        v: T.Tensor = None,
        q_mask: T.BoolTensor = None,
        kv_mask: T.BoolTensor = None,
        attn_mask: T.BoolTensor = None,
    ) -> T.Tensor:
        """
        args:
            q: The main sequence queries (determines the output length)
        kwargs:
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
        """

        # If only q and q_mask are provided then we automatically apply self attention
        if k is None:
            k = q
            if kv_mask is None:
                kv_mask = q_mask
        v = v if v is not None else k

        # Store the batch size, useful for reshaping
        b_size, seq, feat = q.shape

        # Work out the masking situation, with padding, no peaking etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q, k)

        # Generate the q, k, v projections, break final head dimension in 2
        shape = (b_size, seq, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(shape)
        k = self.k_linear(k).view(shape)
        v = self.v_linear(v).view(shape)

        # Transpose to get dimensions and reshape: bxh,seq,hd (required for bmm)
        shape = (b_size * self.num_heads, seq, self.head_dim)
        q = q.transpose(1, 2).contiguous().view(shape)
        k = k.transpose(1, 2).contiguous().view(shape)
        v = v.transpose(1, 2).contiguous().view(shape)

        # Apply the same reshaping to the attention mask
        if attn_mask is not None:
            # Give the attention mask a head dimension if it currently does not have one
            if attn_mask.dim() < 4:
                attn_mask = attn_mask.unsqueeze(-3).expand(-1, self.num_heads, -1, -1)

            # If it has a head dimension move it to forward
            else:
                attn_mask = attn_mask.transpose(-3, -1)
            attn_mask = attn_mask.reshape(b_size * self.num_heads, *attn_mask.shape[2:])

        # Calculate the new sequence values, for memory reasons overwrite q
        q = attention(
            q,
            k,
            v,
            self.model_dim,
            attn_mask=attn_mask,
        )  # Returned shape is bxh,s,f

        # Concatenate the all of the heads together to get shape: b,seq,f
        q = q.view(b_size, self.num_heads, seq, self.head_dim)
        q = q.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        # Pass through final linear layer
        q = self.out_linear(q)

        # Keep zeropadding using nan to num
        q = T.nan_to_num(q)

        return q
