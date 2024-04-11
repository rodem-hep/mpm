"""Defines the graph object type and other operations specific to handing
them."""
from __future__ import annotations

from typing import Iterable, _type_repr

import torch as T
from torch.utils.data._utils.collate import default_collate

from ..torch_utils import sel_device


class Graph:
    """The base class for the custom graph object.

    A collection of 5 tensors:
    - 3 describe the attributes of the edges, nodes, globals
    - 2 describe the structure, providing masks for the edges (adjmat) and nodes (mask)
    """

    def __init__(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        globs: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
        dev: str = "same",
    ) -> None:
        """
        args:
            edges: Compressed edge features (num_edges x Ef)
            nodes: Node features (N x Nf)
            globs: Global features (Gf)
            adjmat: The adjacency matrix, a mask for the edge features (N x N)
            mask: The node mask (N)
        kwargs:
            dev: A string indicating the device on which to store the tensors
        """

        # Check if the edges must be compressed manually
        if edges.dim() > nodes.dim():
            edges = edges[adjmat]

        # Save each of the component tensors onto the correct device
        self.edges = edges
        self.nodes = nodes
        self.globs = globs
        self.adjmat = adjmat
        self.mask = mask

        # Save the device where the graph will be stored
        if dev == "same":
            dev = nodes.device
        else:
            self.to(dev)

    @property
    def dtype(self) -> _type_repr:
        """Inherits the dtype from the node tensor."""
        return self.nodes.dtype

    def dim(self) -> list[int]:
        """Return the dimensions of the graph."""
        return [self.edges.shape[-1], self.nodes.shape[-1], self.globs.shape[-1]]

    def __len__(self) -> int:
        """Return the masking length of the graph."""
        return len(self.mask)

    def max_n(self) -> int:
        """Return the number of nodes that the graph can hold."""
        return self.mask.shape[-1]

    def to(self, dev: str) -> Graph:
        """Move the graph to a selected device."""
        self.edges = self.edges.to(dev)
        self.nodes = self.nodes.to(dev)
        self.globs = self.globs.to(dev)
        self.adjmat = self.adjmat.to(dev)
        self.mask = self.mask.to(dev)
        return self  # Needed for the pytorch move dev

    @property
    def device(self) -> T.device:
        return self.nodes.device

    def _clone(self) -> None:
        """Create an inplace clone of its tensors."""
        self.edges = self.edges.clone()
        self.nodes = self.nodes.clone()
        self.globs = self.globs.clone()
        self.adjmat = self.adjmat.clone()
        self.mask = self.mask.clone()


class GraphBatch(Graph):
    """A batch of graph objects.

    Batching the nodes, globs, adjmat and mask are simple as they just
    receive an extra batch dimension.

    Batching the edges however requires more steps as the edges are in compressed form
    This means that only the nonzero edges in the graph are stored such that
    full_edges[adjmat] = edges
    """

    @property
    def dtype(self):
        """Inherits the dtype from the node tensor."""
        return self.nodes.dtype

    def __getitem__(self, idx: int) -> Graph:
        """Retrieve a particular graph from within the graph batch using an
        index."""

        # Work out the indexes of the edge tensor which is compressed
        start = self.adjmat[:idx].sum()
        size = self.adjmat[idx].sum()

        return Graph(
            self.edges[start : start + size],
            self.nodes[idx],
            self.globs[idx],
            self.adjmat[idx],
            self.mask[idx],
        )

    def dim(self) -> tuple[int, list]:
        """Return the dimensions of the graph object starting with the batch
        length."""
        return (
            len(self),
            [self.edges.shape[-1], self.nodes.shape[-1], self.globs.shape[-1]],
        )

    def has_nan(self) -> list[bool]:
        """Check if there is any nan values in the graph's tensors."""
        result = [
            T.isnan(self.edges).any().item(),
            T.isnan(self.nodes).any().item(),
            T.isnan(self.globs).any().item(),
            T.isnan(self.adjmat).any().item(),
            T.isnan(self.mask).any().item(),
        ]
        return result

    def batch_select(self, b_mask: T.BoolTensor) -> GraphBatch:
        """Returns a batched graph object made from the subset of another
        batched graph This function needs to exist to account for the edges
        which have no batch dimension.

        Operation returns a new graph batch
        """

        assert self.adjmat.sum() == len(self.edges)

        return GraphBatch(
            self.edges[b_mask.repeat_interleave(self.adjmat.sum((-1, -2))).bool()],
            self.nodes[b_mask],
            self.globs[b_mask],
            self.adjmat[b_mask],
            self.mask[b_mask],
        )

    def batch_replace(self, graph_2: GraphBatch, b_mask: T.BoolTensor) -> None:
        """Replace samples with those from graph_2 following a mask Number of
        graphs in graph_2 must be smaller!

        Operation modifies the current graph batch
        """
        self.adjmat[b_mask] = graph_2.adjmat
        self.nodes[b_mask] = graph_2.nodes
        self.globs[b_mask] = graph_2.globs
        self.mask[b_mask] = graph_2.mask

        # This step kills all persistant edges until I work out how to do this
        # TODO Work out how to batch replace without killing edges!
        self.edges = T.zeros((self.adjmat.sum(), 0), dtype=T.float, device=self.device)

    def __repr__(self):
        """Return the name of the graph and its dimension for printing."""
        return f"GraphBatch({self.dim()})"

    def clone(self) -> GraphBatch:
        """Return an out of place clone of its tensors."""
        return GraphBatch(
            self.edges.clone(),
            self.nodes.clone(),
            self.globs.clone(),
            self.adjmat.clone(),
            self.mask.clone(),
        )


def gcoll(batch: Iterable) -> GraphBatch | tuple:
    """A custom collation function which allows us to batch together multiple
    graphs.

    - Wraps the pytorch default collation function to allow for all the memory tricks
    - Looks at the first element of the batch for instructions on what to do

    args:
        batch: An iterable list/tuple containing graphs or other iterables of graphs
    returns:
        Batched graph object
    """

    # Get the first element object type
    elem = batch[0]

    # If we are dealing with a graph object then we apply the customised collation
    if isinstance(elem, Graph):
        edges = T.cat([g.edges for g in batch])  # Input edges should be compressed
        nodes = default_collate([g.nodes for g in batch])
        globs = default_collate([g.globs for g in batch])
        adjmat = default_collate([g.adjmat for g in batch])
        mask = default_collate([g.mask for g in batch])
        return GraphBatch(edges, nodes, globs, adjmat, mask, dev=mask.device)

    # If we have a tuple, we must run the function for each object
    elif isinstance(elem, tuple):
        return tuple(gcoll(samples) for samples in zip(*batch))

    # If we are dealing with any other type we must run the normal collation function
    else:
        return default_collate(batch)


def blank_graph_batch(
    dim: list, max_nodes: int, b_size: int = 1, dev: str = "cpu"
) -> GraphBatch:
    """Create an empty graph of a certain shape.

    - All attributes are zeros
    - All masks/adjmats are false

    args:
        dim: The dimensions of the desired graph [e,n,g]
        max_nodes: The max number of nodes to allow for
        b_size: The batch dimension
    kwargs:
        dev: The device on which to store the graph
    returns:
        Empty graph object
    """
    dev = sel_device(dev)
    edges = T.zeros((0, dim[0]), device=dev)
    nodes = T.zeros((b_size, max_nodes, dim[1]), device=dev)
    globs = T.zeros((b_size, dim[2]), device=dev)
    adjmat = T.zeros((b_size, max_nodes, max_nodes), device=dev).bool()
    mask = T.zeros((b_size, max_nodes), device=dev).bool()

    return GraphBatch(edges, nodes, globs, adjmat, mask, dev=dev)
