"""Defines the lightweight and streamlined graph object and operations."""
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from ..gnets.graphs import GraphBatch
from ..gnets.modules import DenseNetwork
from ..torch_utils import aggr_via_sparse, ctxt_from_mask, decompress, pass_with_mask


class EdgeBlockLite(nn.Module):
    """The edge updating and pooling step of a graph network block."""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        n_heads: int = 1,
        feat_kwargs: dict = None,
        attn_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            n_heads: Number of attention heads
            feat_kwargs: The dictionary of kwargs for the feature dense network
            attn_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        # Number of attention heads must divide dimension
        assert outp_dim[0] % n_heads == 0
        self.head_dim = outp_dim[0] // n_heads

        # Dict default kwargs
        feat_kwargs = feat_kwargs or {}
        attn_kwargs = attn_kwargs or {}

        # Useful dimensions
        edge_inpt_dim = inpt_dim[0] + 2 * inpt_dim[1]
        ctxt_inpt_dim = inpt_dim[2] + ctxt_dim
        self.same_size = inpt_dim[0] == outp_dim[0]

        # The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=edge_inpt_dim,
            outp_dim=outp_dim[0],
            ctxt_dim=ctxt_inpt_dim,
            **feat_kwargs,
        )

        # The attention network for pooling
        self.attn_net = DenseNetwork(
            inpt_dim=edge_inpt_dim,
            outp_dim=n_heads,
            ctxt_dim=ctxt_inpt_dim,
            **attn_kwargs,
        )

        # The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(edge_inpt_dim)

    def forward(self, graph: GraphBatch, ctxt: T.Tensor = None) -> T.Tensor:
        """
        args:
            graph: The batched graph object
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_edges: The new edge features of the graph
        """

        # Create the inputs for the edge networks
        ex_size = (*graph.adjmat.shape, -1)
        edges = T.cat(
            [
                graph.nodes.unsqueeze(-2).expand(ex_size)[graph.adjmat],
                graph.nodes.unsqueeze(-3).expand(ex_size)[graph.adjmat],
                graph.edges,
            ],
            dim=-1,
        )
        edges = self.pre_ln(edges)

        # Pass them through the attention network (first allows overwrite)
        edge_weights = self.attn_net(
            edges, ctxt_from_mask([graph.globs, ctxt], graph.adjmat)
        )
        edge_weights = aggr_via_sparse(
            edge_weights, graph.adjmat, reduction="softmax", dim=1
        )

        # Pass them through the feature network
        edges = self.feat_net(edges, ctxt_from_mask([graph.globs, ctxt], graph.adjmat))
        if self.same_size:
            edges = edges + graph.edges

        # Broadcast the attention to get the multiple poolings and sum
        edge_weights = (
            edge_weights.unsqueeze(-1).expand(-1, -1, self.head_dim).flatten(1)
        )
        edge_weights = edges * edge_weights
        edge_weights = aggr_via_sparse(
            edge_weights, graph.adjmat, reduction="sum", dim=1
        )
        edge_weights = edge_weights / math.sqrt(self.feat_net.outp_dim)
        edge_weights = decompress(edge_weights, graph.adjmat.any(1))

        return edges, edge_weights


class NodeBlockLite(nn.Module):
    """The node updating and pooling step of a graph network block."""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        n_heads: int = 1,
        feat_kwargs: dict = None,
        attn_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            n_heads: Number of attention heads
            feat_kwargs: The dictionary of kwargs for the feature dense network
            attn_kwargs: The dictionary of kwargs for the attention dense network
        """
        super().__init__()

        # Number of attention heads must divide dimension
        assert outp_dim[1] % n_heads == 0
        self.head_dim = outp_dim[1] // n_heads

        # Dict default kwargs
        feat_kwargs = feat_kwargs or {}
        attn_kwargs = attn_kwargs or {}

        # Useful dimensions
        node_inpt_dim = outp_dim[0] + inpt_dim[1]
        ctxt_inpt_dim = inpt_dim[2] + ctxt_dim
        self.same_size = inpt_dim[1] == outp_dim[1]

        # The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=node_inpt_dim,
            outp_dim=outp_dim[1],
            ctxt_dim=ctxt_inpt_dim,
            **feat_kwargs,
        )

        # The attention network for pooling
        self.attn_net = DenseNetwork(
            inpt_dim=node_inpt_dim,
            outp_dim=n_heads,
            ctxt_dim=ctxt_inpt_dim,
            **attn_kwargs,
        )

        # The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(node_inpt_dim)

    def forward(
        self, graph: GraphBatch, pooled_edges: T.Tensor, ctxt: T.Tensor = None
    ) -> T.Tensor:
        """
        args:
            graph: The batched graph object
            pooled_edges: The pooled information per receiver node
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_nodes: The new node features of the graph
        """

        # Create the inputs for the node networks
        nodes = T.cat([graph.nodes, pooled_edges], dim=-1)
        nodes = self.pre_ln(nodes)

        # Pass them through the attention network (first allows overwrite)
        node_weights = F.softmax(
            pass_with_mask(
                nodes,
                self.attn_net,
                graph.mask,
                high_level=[graph.globs, ctxt],
                padval=-T.inf,
            ),
            dim=-2,
        ) / math.sqrt(self.feat_net.outp_dim)
        node_weights[~graph.mask] = 0  # Get rid of nans for 0 node graphs (remove?)

        # Pass them through the feature network
        nodes = pass_with_mask(
            nodes, self.feat_net, graph.mask, high_level=[graph.globs, ctxt]
        )
        if self.same_size:
            nodes = nodes + graph.nodes

        # Broadcast the attention to get the multiple poolings and sum
        node_weights = (
            node_weights.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(-2)
        )
        node_weights = (nodes * node_weights).sum(dim=-2)

        return nodes, node_weights


class GlobBlockLite(nn.Module):
    """The global updating step of a graph network block."""

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        feat_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            outp_dim: The dimensions of the output graph [e,n,g] of the gn block
            ctxt_dim: The size of the contextual information
            feat_kwargs: The dictionary of kwargs for the feature dense network
        """
        super().__init__()

        # Dict default kwargs
        feat_kwargs = feat_kwargs or {}

        # Useful dimensions
        glob_inpt_dim = outp_dim[1] + inpt_dim[2]
        self.same_size = inpt_dim[2] == outp_dim[2]

        # The dense network to update messsages
        self.feat_net = DenseNetwork(
            inpt_dim=glob_inpt_dim,
            outp_dim=outp_dim[2],
            ctxt_dim=ctxt_dim,
            **feat_kwargs,
        )

        # The pre-post layernormalisation layer
        self.pre_ln = nn.LayerNorm(glob_inpt_dim)

    def forward(
        self, graph: GraphBatch, pooled_nodes: T.Tensor, ctxt: T.Tensor = None
    ) -> T.Tensor:
        """
        args:
            graph: The batched graph object
            pooled_nodes: The pooled information across the graph
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_globs: The new global features of the graph
        """
        globs = T.cat([graph.globs, pooled_nodes], dim=-1)
        globs = self.pre_ln(globs)
        globs = self.feat_net(globs, ctxt=ctxt)
        if self.same_size:
            globs = globs + graph.globs
        return globs


class GNBlockLite(nn.Module):
    """A message passing Graph Network Block.

    - Lite implies that the coding and variability between models is minimal
    - Always applied additive residual connections if available
    - Always uses attention pooling
    - Always produces output edges
    - Always produces output globals
    - Always applies a pre and post LayerNormalisation
    """

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: list,
        ctxt_dim: int = 0,
        edge_block_kwargs: dict = None,
        node_block_kwargs: dict = None,
        glob_block_kwargs: dict = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
            outp_dim: The dimensions of the output graph [e,n,g]
        kwargs:
            edge_block_kwargs: kwargs for the edge block
            node_block_kwargs: kwargs for the node block
            glob_block_kwargs: kwargs for the glob block
        """
        super().__init__()

        # Dict default kwargs
        edge_block_kwargs = edge_block_kwargs or {}
        node_block_kwargs = node_block_kwargs or {}
        glob_block_kwargs = glob_block_kwargs or {}

        # Store the input dimensions
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # Define the update blocks
        self.edge_block = EdgeBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **edge_block_kwargs
        )
        self.node_block = NodeBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **node_block_kwargs
        )
        self.glob_block = GlobBlockLite(
            inpt_dim, outp_dim, ctxt_dim, **glob_block_kwargs
        )

    def forward(self, graph: GraphBatch, ctxt: T.Tensor = None) -> GraphBatch:
        """Return an updated graph with the same structure, but new
        features."""
        graph.edges, pooled_edges = self.edge_block(graph, ctxt)
        graph.nodes, pooled_nodes = self.node_block(graph, pooled_edges, ctxt)
        del pooled_edges  # Saves alot of memory if we delete right away
        graph.globs = self.glob_block(graph, pooled_nodes, ctxt)
        return graph

    def __repr__(self):
        """A way to print the block config on one line for quick review."""
        string = str(self.inpt_dim)
        string += f"->EdgeNet[{self.edge_block.feat_net.one_line_string()}]"
        if self.edge_block.same_size:
            string += "(+)"
        string += f"->EdgePool[{self.edge_block.attn_net.one_line_string()}]"
        string += f"->NodeNet[{self.node_block.feat_net.one_line_string()}]"
        if self.node_block.same_size:
            string += "(+)"
        string += f"->NodePool[{self.node_block.attn_net.one_line_string()}]"
        string += f"->GlobNet[{self.glob_block.feat_net.one_line_string()}]"
        if self.glob_block.same_size:
            string += "(+)"
        return string


class GNBStack(nn.Module):
    """A stack of N many identical GNBlockLite(s) Graph to Graph."""

    def __init__(
        self,
        inpt_dim: list,
        model_dim: list,
        num_blocks: int,
        ctxt_dim: int = 0,
        edge_block_kwargs: dict = None,
        node_block_kwargs: dict = None,
        glob_block_kwargs: dict = None,
    ) -> None:
        """
        args:
            num_blocks: The number of blocks in the stack
            inpt_dim: The dimensions of the input graph [e,n,g] (unchanging)
        kwargs:
            edge_block_kwargs: kwargs for the edge block
            node_block_kwargs: kwargs for the node block
            glob_block_kwargs: kwargs for the glob block
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.inpt_dim = inpt_dim
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim
        self.blocks = nn.ModuleList(
            [
                GNBlockLite(
                    inpt_dim if i == 0 else model_dim,
                    model_dim,
                    ctxt_dim,
                    edge_block_kwargs,
                    node_block_kwargs,
                    glob_block_kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, graph: GraphBatch, ctxt: T.Tensor = None) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        graph._clone()
        for blocks in self.blocks:
            graph = blocks(graph, ctxt)
        return graph
