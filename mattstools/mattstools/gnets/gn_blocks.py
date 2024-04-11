"""Defines the lightweight and streamlined graph object and operations."""
import math
from typing import Optional, Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap

from ..modules import DenseNetwork
from ..torch_utils import (
    aggr_via_sparse,
    ctxt_from_mask,
    decompress,
    empty_0dim_like,
    masked_pool,
    pass_with_mask,
    smart_cat,
)
from .graphs import GraphBatch


class EdgeBlock(nn.Module):
    """The edge updating and pooling step of a graph network block.

    Generates input edge features:
    - Existing edges (if present)
    - Combined send, recv node features
    - Applies a pre-LN step if desired (reccomended)

    (Optional) Calculates new edges using a neural network
    - Uses globals and ctxt tensor as conditional information

    Updates the old edges with the new ones
    - Can use a concatenation or additive residual layer
    - Additive residual layer will only work if dim does not change

    Pools the new edges by removing the sender dimension
    - Can use average pooling, sum pooling, or attention pooling
    """

    def __init__(
        self,
        inpt_dim: list,
        ctxt_dim: int = 0,
        msg_type: str = "sr",
        do_ln: bool = True,
        use_net: bool = True,
        pool_type: str = "attn",
        rsdl_type: str = "add",
        feat_kwargs: DotMap = None,
        attn_kwargs: DotMap = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
        kwargs:
            ctxt_dim: The size of the contextual information
            msg_type: How the send/receive messages are combined
            do_ln: If the input features are going to be layer-normed
            use_net: If the features are updated with a neural network
            pool_type: The type of pooling operation to apply
            feat_kwargs: The DotMapionary of kwargs for the feature dense network
            attn_kwargs: The DotMapionary of kwargs for the attention dense network
        """
        super().__init__()

        # Check the configuration
        assert all(m in "srd" for m in msg_type)
        assert pool_type in ["sum", "mean", "attn"]
        assert rsdl_type in ["none", "add", "cat"]

        # DotMap default kwargs
        self.feat_kwargs = feat_kwargs or DotMap({"outp_dim": 4})
        self.attn_kwargs = attn_kwargs or DotMap({"outp_dim": 1})

        # Set the attributes from the parameters
        self.inpt_dim = inpt_dim
        self.msg_type = msg_type
        self.do_ln = do_ln
        self.use_net = use_net
        self.pool_type = pool_type
        self.rsdl_type = rsdl_type
        self.ctxt_dim = ctxt_dim
        self.do_rsdl = rsdl_type != "none" and inpt_dim[0]

        # Useful dimensions
        self.feat_inpt_dim, self.feat_outp_dim, self.ctxt_inpt_dim = self._get_dims()

        # Values dependant on the above dimensions
        if pool_type == "attn":
            self.n_heads = self.attn_kwargs.outp_dim
            assert self.feat_outp_dim % self.n_heads == 0
            self.head_dim = self.feat_outp_dim // self.n_heads

        # The pre layer normalisation layer
        if do_ln:
            self.pre_ln = nn.LayerNorm(self.feat_inpt_dim)

        # The dense network to update messsages
        if use_net:
            self.feat_net = DenseNetwork(
                inpt_dim=self.feat_inpt_dim,
                ctxt_dim=self.ctxt_inpt_dim,
                **self.feat_kwargs,
            )

        # The attention network for pooling
        if pool_type == "attn":
            self.attn_net = DenseNetwork(
                inpt_dim=self.feat_inpt_dim,
                ctxt_dim=self.ctxt_inpt_dim,
                **self.attn_kwargs,
            )

        # Turn off residual additive connections if the sizes dont line up
        if rsdl_type == "add" and inpt_dim[0] != self.feat_outp_dim:
            self.do_rsdl = False

    def _get_dims(self):
        """Set some of the important dimensions needed for the block."""

        # Without a network the messages are made up of only pooled intormation
        feat_inpt_dim = len(self.msg_type) * self.inpt_dim[1]
        feat_outp_dim = feat_inpt_dim

        # With a network output can be anything and the inputs also take old edges
        if self.use_net:
            feat_inpt_dim += self.inpt_dim[0]
            feat_outp_dim = self.feat_kwargs.outp_dim

        # If using concat residual connections
        if self.do_rsdl and self.rsdl_type == "cat":
            feat_outp_dim += self.inpt_dim[0]

        # Full context dimension comes from global and ctxt information
        ctxt_inpt_dim = self.inpt_dim[2] + self.ctxt_dim

        return feat_inpt_dim, feat_outp_dim, ctxt_inpt_dim

    def _build_messages(self, graph: GraphBatch) -> T.Tensor:
        """Create the messages passed between the two nodes."""

        # Expand the node contributions (No mem is allocated in expand!)
        ex_size = (*graph.adjmat.shape, -1)
        if "s" in self.msg_type or "d" in self.msg_type:
            send_info = graph.nodes.unsqueeze(-2).expand(ex_size)[graph.adjmat]
        if "r" in self.msg_type or "d" in self.msg_type:
            recv_info = graph.nodes.unsqueeze(-3).expand(ex_size)[graph.adjmat]
        if "d" in self.msg_type:
            diff_info = send_info - recv_info

        # Start the messages as empty and add each tensor as needed
        pooled_mssgs = []
        if "s" in self.msg_type:
            pooled_mssgs.append(send_info)
        if "r" in self.msg_type:
            pooled_mssgs.append(recv_info)
        if "d" in self.msg_type:
            pooled_mssgs.append(diff_info)

        # Return all messages
        return smart_cat(pooled_mssgs, -1)

    def forward(
        self, graph: GraphBatch, ctxt: T.Tensor = None
    ) -> Tuple[T.Tensor, T.Tensor]:
        """
        args:
            graph: The batched graph object
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_edges: The new edge features of the graph
            pooled_edges: The pooled edges of the graph for the node update
        """

        # Create the inputs for the edge networks
        if self.use_net:
            edges = smart_cat([graph.edges, self._build_messages(graph)])
        else:
            edges = self._build_messages(graph)

        # Apply the pre_layer normalisation layer
        if self.do_ln:
            edges = self.pre_ln(edges)

        # Pass them through the attention network
        # (doing this first allows overrite and saves memory)
        if self.pool_type == "attn":
            pooled_edges = self.attn_net(
                edges, ctxt_from_mask([graph.globs, ctxt], graph.adjmat)
            ) / math.sqrt(self.feat_outp_dim)
            pooled_edges = aggr_via_sparse(
                pooled_edges, graph.adjmat, reduction="softmax", dim=1
            )
            pooled_edges = (
                pooled_edges.unsqueeze(-1).expand(-1, -1, self.head_dim).flatten(1)
            )

        # Pass the edges through the feature network
        if self.use_net:
            edges = self.feat_net(
                edges, ctxt_from_mask([graph.globs, ctxt], graph.adjmat)
            )

        # Apply the residual update
        if self.do_rsdl:
            if self.rsdl_type == "cat":
                edges = smart_cat([edges, graph.edges], dim=-1)
            if self.rsdl_type == "add":
                edges = edges + graph.edges

        # Apply the pooling per node,
        if self.pool_type == "attn":
            pooled_edges = aggr_via_sparse(
                edges * pooled_edges, graph.adjmat, reduction="sum", dim=1
            )
        else:
            pooled_edges = aggr_via_sparse(
                edges, graph.adjmat, reduction=self.pool_type, dim=1
            )

        # Decompress the pooled information as nodes are not comp!
        pooled_edges = decompress(pooled_edges, graph.adjmat.any(1))

        # Return the new edge features and the pooled edges
        return edges, pooled_edges

    def __repr__(self):
        """Sort string representation of the block."""
        string = f"EdgeBlock[{self.msg_type}"
        if self.do_ln:
            string += "-LN"
        if self.use_net:
            string += f"-FF({self.feat_net.one_line_string()})"
        if self.do_rsdl:
            string += f"-{self.rsdl_type}"
        string += f"-{self.pool_type}"
        if self.pool_type == "attn":
            string += f"({self.n_heads})"
        string += "]"
        return string


class NodeBlock(nn.Module):
    """The node updating and pooling step of a graph network block.

    Generates input node features:
    - Existing nodes
    - Pooled edge information
    - Applies a pre-LN step if desired (reccomended)

    (Optional) Calculates new edges using a neural network
    - Uses globals and ctxt tensor as conditional information

    Updates the old nodes with the new ones
    - Can use a concatenation or additive residual layer
    - Additive residual layer will only work if dim does not change

    Pools the new nodes over the entire graph
    - Only happens if the graph is producing a global layer
    - Can use average pooling, sum pooling, or attention pooling
    """

    def __init__(
        self,
        inpt_dim: list,
        pooled_edge_dim: int,
        ctxt_dim: int = 0,
        do_globs: bool = True,
        do_ln: bool = True,
        use_net: bool = True,
        pool_type: str = "attn",
        rsdl_type: str = "add",
        feat_kwargs: DotMap = None,
        attn_kwargs: DotMap = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            pooled_edge_dim: The size of the pooled edge tensors per receiver node
        kwargs:
            ctxt_dim: The size of the contextual information
            do_globs: If the output graph has global features (if pooling is required)
            do_ln: If the input features are going to be layer-normed
            use_net: If the features are updated with a neural network
            pool_type: The type of pooling operation to apply
            rsdl_type: Type of residuel connection to apply
            feat_kwargs: The DotMapionary of kwargs for the feature dense network
            attn_kwargs: The DotMapionary of kwargs for the attention dense network
        """
        super().__init__()

        # Check the configuration
        assert pool_type in ["sum", "mean", "attn"]
        assert rsdl_type in ["none", "add", "cat"]

        # DotMap default kwargs
        self.feat_kwargs = feat_kwargs or DotMap({"outp_dim": 4})
        self.attn_kwargs = attn_kwargs or DotMap({"outp_dim": 1})

        # Set the attributes from the parameters
        self.inpt_dim = inpt_dim
        self.pooled_edge_dim = pooled_edge_dim
        self.do_globs = do_globs
        self.do_ln = do_ln
        self.use_net = use_net
        self.pool_type = pool_type
        self.rsdl_type = rsdl_type
        self.ctxt_dim = ctxt_dim
        self.do_rsdl = rsdl_type != "none" and inpt_dim[1]

        # Useful dimensions
        self.feat_inpt_dim, self.feat_outp_dim, self.ctxt_inpt_dim = self._get_dims()

        # Values dependant on the above dimensions
        if pool_type == "attn" and self.do_globs:
            self.n_heads = self.attn_kwargs.outp_dim
            assert self.feat_outp_dim % self.n_heads == 0
            self.head_dim = self.feat_outp_dim // self.n_heads

        # The pre layer normalisation layer
        if do_ln:
            self.pre_ln = nn.LayerNorm(self.feat_inpt_dim)

        # The dense network to update features
        if use_net:
            self.feat_net = DenseNetwork(
                inpt_dim=self.feat_inpt_dim,
                ctxt_dim=self.ctxt_inpt_dim,
                **self.feat_kwargs,
            )

        # The attention network for pooling
        if pool_type == "attn" and do_globs:
            self.attn_net = DenseNetwork(
                inpt_dim=self.feat_inpt_dim,
                ctxt_dim=self.ctxt_inpt_dim,
                **self.attn_kwargs,
            )

        # Turn off residual additive connections if the sizes dont line up
        if rsdl_type == "add" and inpt_dim[1] != self.feat_outp_dim:
            self.do_rsdl = False

    def _get_dims(self):
        """Set some of the important dimensions needed for the block."""

        # Without a network the messages are made up of only pooled intormation
        feat_inpt_dim = self.pooled_edge_dim
        feat_outp_dim = feat_inpt_dim

        # With a network output can be anything and the inputs also take the old nodes
        if self.use_net:
            feat_inpt_dim += self.inpt_dim[1]
            feat_outp_dim = self.feat_kwargs.outp_dim

        # If using concat residual connections
        if self.do_rsdl and self.rsdl_type == "cat":
            feat_outp_dim += self.inpt_dim[1]

        # Full context dimension comes from global and ctxt information
        ctxt_inpt_dim = self.inpt_dim[2] + self.ctxt_dim

        return feat_inpt_dim, feat_outp_dim, ctxt_inpt_dim

    def forward(
        self, graph: GraphBatch, pooled_edges: T.Tensor, ctxt: T.Tensor = None
    ) -> Tuple[T.Tensor, T.Tensor]:
        """
        args:
            graph: The batched graph object
            pooled_edges: The pooled information per receiver node
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_nodes: The new node features of the graph
            pooled_nodes: The pooled nodes across the graph
        """

        # Create the inputs for the node networks
        if self.use_net:
            nodes = T.cat([graph.nodes, pooled_edges], dim=-1)
        else:
            nodes = pooled_edges

        # Apply the pre_layer normalisation layer
        if self.do_ln:
            nodes = self.pre_ln(nodes)

        # Pass them through the attention network
        # (doing this first allows overrite and saves memory)
        if self.pool_type == "attn" and self.do_globs:
            pooled_nodes = F.softmax(
                pass_with_mask(
                    nodes,
                    self.attn_net,
                    graph.mask,
                    high_level=[graph.globs, ctxt],
                    padval=-T.inf,
                )
                / math.sqrt(self.feat_outp_dim),
                dim=-2,
            )
            pooled_nodes = T.nan_to_num(pooled_nodes)  # Prevents Nans in 0 node graphs
            pooled_nodes = (
                pooled_nodes.unsqueeze(-1).expand(-1, -1, -1, self.head_dim).flatten(-2)
            )

        # Pass the nodes through the feature network
        if self.use_net:
            nodes = pass_with_mask(
                nodes, self.feat_net, graph.mask, high_level=[graph.globs, ctxt]
            )

        # Apply the residual update
        if self.do_rsdl:
            if self.rsdl_type == "cat":
                nodes = smart_cat([nodes, graph.nodes], dim=-1)
            if self.rsdl_type == "add":
                nodes = nodes + graph.nodes

        # Apply the pooling across the graph
        if self.do_globs:
            if self.pool_type == "attn":
                pooled_nodes = (nodes * pooled_nodes).sum(dim=-2)
            else:
                pooled_nodes = masked_pool(self.pool_type, nodes, graph.mask)
        else:
            pooled_nodes = None

        return nodes, pooled_nodes

    def __repr__(self):
        """Sort string representation of the block."""
        string = "NodeBlock["
        if self.do_ln:
            string += "LN-"
        if self.use_net:
            string += f"FF({self.feat_net.one_line_string()})-"
        if self.do_rsdl:
            string += f"{self.rsdl_type}-"
        if self.do_globs:
            string += f"{self.pool_type}"
            if self.pool_type == "attn":
                string += f"({self.n_heads})"
        if string[-1] == "-":
            string = string[:-1]
        string += "]"
        return string


class GlobBlock(nn.Module):
    """The global updating step of a graph network block.

    Generates input global features:
    - Existing globals (if present)
    - Pooled node informatio
    - Applies a pre-LN step if desired (reccomended)

    (Optional) Calculates new globals using a neural network
    - Uses ctxt tensor as conditional information

    Updates the old globals with the new ones
    - Can use a concatenation or additive residual layer
    - Additive residual layer will only work if dim does not change
    """

    def __init__(
        self,
        inpt_dim: list,
        pooled_node_dim: int,
        ctxt_dim: int = 0,
        do_ln: bool = True,
        use_net: bool = True,
        rsdl_type: str = "add",
        feat_kwargs: DotMap = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g] of the gn block
            pooled_edge_dim: The size of the pooled node tensors over the graph
        kwargs:
            ctxt_dim: The size of the contextual information
            do_ln: If the input features are going to be layer-normed
            use_net: If the features are updated with a neural network
            rsdl_type: Type of residuel connection to apply
            feat_kwargs: The DotMapionary of kwargs for the feature dense network
        """
        super().__init__()

        # Check the configuration
        assert rsdl_type in ["none", "add", "cat"]

        # DotMap default kwargs
        self.feat_kwargs = feat_kwargs or DotMap({"outp_dim": 4})

        # Useful dimensions
        self.inpt_dim = inpt_dim
        self.pooled_node_dim = pooled_node_dim
        self.ctxt_dim = ctxt_dim
        self.do_ln = do_ln
        self.use_net = use_net
        self.rsdl_type = rsdl_type
        self.do_rsdl = rsdl_type != "none" and inpt_dim[2]

        # Useful dimensions
        self.feat_inpt_dim, self.feat_outp_dim = self._get_dims()

        # The pre layer normalisation layer
        if do_ln:
            self.pre_ln = nn.LayerNorm(self.feat_inpt_dim)

        # The dense network to update features
        if use_net:
            self.feat_net = DenseNetwork(
                inpt_dim=self.feat_inpt_dim, ctxt_dim=self.ctxt_dim, **self.feat_kwargs
            )

        # Turn off residual additive connections if the sizes dont line up
        if rsdl_type == "add" and inpt_dim[2] != self.feat_outp_dim:
            self.do_rsdl = False

    def _get_dims(self):
        """Set some of the important dimensions needed for the block."""

        # Without a network the messages are made up of only pooled intormation
        feat_inpt_dim = self.pooled_node_dim
        feat_outp_dim = feat_inpt_dim

        # With a network output can be anything and the inputs also take the old globs
        if self.use_net:
            feat_inpt_dim += self.inpt_dim[2]
            feat_outp_dim = self.feat_kwargs.outp_dim

        # If using concat residual connections
        if self.do_rsdl and self.rsdl_type == "cat":
            feat_outp_dim += self.inpt_dim[2]

        return feat_inpt_dim, feat_outp_dim

    def forward(
        self, graph: GraphBatch, pooled_nodes: T.Tensor, ctxt: T.Tensor = None
    ) -> Tuple[T.Tensor, T.Tensor]:
        """
        args:
            graph: The batched graph object
            pooled_nodes: The pooled information across the graph
        kwargs:
            ctxt: The extra context tensor
        returns:
            new_globs: The new global features of the graph
        """

        # Create the inputs for the global networks
        if self.use_net:
            globs = T.cat([graph.globs, pooled_nodes], dim=-1)
        else:
            globs = pooled_nodes

        # Apply the pre_layer normalisation layer
        if self.do_ln:
            globs = self.pre_ln(globs)

        # Pass through the feature network
        if self.use_net:
            globs = self.feat_net(globs, ctxt=ctxt)

        # Apply the residual update
        if self.do_rsdl:
            if self.rsdl_type == "cat":
                globs = smart_cat([globs, graph.globs], dim=-1)
            if self.rsdl_type == "add":
                globs = globs + graph.globs

        return globs

    def __repr__(self):
        """Short string representation of the block."""
        string = []
        if self.do_ln:
            string.append("LN")
        if self.use_net:
            string.append(f"FF({self.feat_net.one_line_string()})")
        if self.do_rsdl:
            string.append(f"{self.rsdl_type}")
        string = f"GlobBlock[{'-'.join(string)}]"
        return string


class GNBlock(nn.Module):
    """A message passing Graph Network Block Updates the edges, nodes and
    globals in turn and returns a new graph batch."""

    def __init__(
        self,
        inpt_dim: list,
        ctxt_dim: int = 0,
        do_globs: bool = True,
        pers_edges: bool = True,
        edge_block_kwargs: Optional[DotMap] = None,
        node_block_kwargs: Optional[DotMap] = None,
        glob_block_kwargs: Optional[DotMap] = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
            ctxt_dim: The dimension of the conditioning tensor
        kwargs:
            do_globs: If this block will produce new global values
            pers_edges: If this block keeps edge features, or if they are dropped
            edge_block_kwargs: kwargs for the edge block
            node_block_kwargs: kwargs for the node block
            glob_block_kwargs: kwargs for the glob block
        """
        super().__init__()

        # DotMap default kwargs
        edge_block_kwargs = edge_block_kwargs or {}
        node_block_kwargs = node_block_kwargs or {}
        glob_block_kwargs = glob_block_kwargs or {}

        # Store the params
        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.do_globs = do_globs
        self.pers_edges = pers_edges

        # The module's edge block
        self.edge_block = EdgeBlock(
            inpt_dim=inpt_dim, ctxt_dim=ctxt_dim, **edge_block_kwargs
        )

        # The module's node block
        self.node_block = NodeBlock(
            inpt_dim=inpt_dim,
            pooled_edge_dim=self.edge_block.feat_outp_dim,
            ctxt_dim=ctxt_dim,
            do_globs=do_globs,
            **node_block_kwargs,
        )

        # (optional) The module's global block
        if do_globs:
            self.glob_block = GlobBlock(
                inpt_dim=inpt_dim,
                pooled_node_dim=self.node_block.feat_outp_dim,
                ctxt_dim=ctxt_dim,
                **glob_block_kwargs,
            )

        # Calculate the output size of this graph
        self.outp_dim = [
            self.edge_block.feat_outp_dim if pers_edges else 0,
            self.node_block.feat_outp_dim,
            self.glob_block.feat_outp_dim if do_globs else 0,
        ]

    def forward(self, graph: GraphBatch, ctxt: T.Tensor = None) -> GraphBatch:
        """Return an updated graph with the same structure, but new
        features."""
        graph.edges, pooled_edges = self.edge_block(graph, ctxt)
        graph.nodes, pooled_nodes = self.node_block(graph, pooled_edges, ctxt)
        del pooled_edges  # Saves alot of memory if we delete right away
        if not self.pers_edges:
            graph.edges = empty_0dim_like(graph.edges)
        if self.do_globs:
            graph.globs = self.glob_block(graph, pooled_nodes, ctxt)
        else:
            graph.globs = empty_0dim_like(graph.globs)
        return graph

    def __repr__(self):
        """A way to print the block config on one line for quick review."""
        string = str(self.inpt_dim)
        string += "->" + repr(self.edge_block)
        string += "->" + repr(self.node_block)
        if self.do_globs:
            string += "->" + repr(self.glob_block)
        string += "->" + str(self.outp_dim)
        return string
