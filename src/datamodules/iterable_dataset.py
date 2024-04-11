"""Pytorch Dataset definitions of various collections training samples."""

from copy import deepcopy
from pathlib import Path
from typing import List, Union
from functools import partial

import numpy as np
from torch.utils.data import IterableDataset
from src.datamodules.loading import JetClassIterator
from src.jet_utils import boost_jet_mopt, build_jet_edges, graph_coordinates


class IteratorWrapper(object):
    def __init__(
        self,
        base_iterator: JetClassIterator,
        augmentation_list: list,
        augmentation_prob: float,
        coordinates: dict,
        del_r_edges: float,
        boost_mopt: float,
    ) -> None:
        self.base_iterator = base_iterator
        self.n_classes = self.base_iterator.get_nclasses()
        self.augmentation_list = augmentation_list
        self.augmentation_prob = augmentation_prob
        self.coordinates = coordinates
        self.del_r_edges = del_r_edges
        self.boost_mopt = boost_mopt

    def __next__(self):
        # Load the particle constituent and high level jet information from the data
        high, nodes, mask, label = next(self.base_iterator)
        if self.base_iterator.features is not None:
            add_nodes = nodes[:, 3:]
            nodes = nodes[:, :3]

        # Build jet edges (will return empty if del_r is set to 0)
        # Edges are also compressed to save memory
        edges, adjmat = build_jet_edges(
            nodes, mask, self.coordinates["edge"], self.del_r_edges
        )

        # Apply boost pre-processing after the jet edges
        if self.boost_mopt != 0:
            nodes, high = boost_jet_mopt(nodes, high, self.boost_mopt)

        # Convert to the specified selection of local variables and extract edges
        nodes, high = graph_coordinates(nodes, high, mask, self.coordinates)

        # Add the additional features to the nodes
        if self.base_iterator.features is not None:
            nodes = np.concatenate((nodes, add_nodes), axis=1)

        # Clip the coordinates to their radiues
        nodes[:, :2] = np.clip(nodes[:, :2], -0.8, 0.8)
        return edges, nodes, high, adjmat, mask, label


class ModelClassifyIterator(IteratorWrapper):
    """Load a model and use this to label data, this can be done on the fly"""

    def __init__(self, *args, model_path: Path) -> None:
        super().__init__(*args)
        # Load a model from the specified path
        # This has to be a src.models.vq_vae.TransformerVqVae
        # orig_cfg = reload_original_config(path_spec=model_path, get_best=False)
        # model_class = hydra.utils.get_class(orig_cfg.model._target_)
        # self.model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)
        # self.remap_label = torch.arange(self.model.vq_layer.num_codes)
        self.n_classes = self.get_nclasses()

    # This works but is much much too slow, especially with a single work.
    # def __next__(self):
    #     return self.process_sample(super().__next__())

    # def process_sample(self, sample):
    #     # This actually isn't used
    #     sample_ = [torch.tensor(s).unsqueeze(0) for s in sample]
    #     with torch.no_grad():
    #         label = self.model.predict_step(sample_, 0)["code_labels"]
    #     label = self.remap_label.to(label)[label]
    #     sample_[-1] = label
    #     return [s.squeeze(0) for s in sample_]

    def get_nclasses(self):
        # labels = torch.Tensor([])
        # for i in range(int(5e3)):
        #     proposed_labels = self.process_sample(next(self))[-1]
        #     labels = torch.unique(torch.concatenate((labels, proposed_labels.view(-1))))

        # # Set overflow labels to zero
        # self.remap_label = torch.zeros(self.model.vq_layer.num_codes, dtype=torch.int32)
        # # Remap the labels
        # self.remap_label[labels.to(torch.int32)] = torch.arange(len(labels), dtype=torch.int32)

        # return len(labels) + 1
        return 427


class JetData(IterableDataset):
    """A pytorch dataset object containing high and low level jet
    information."""

    def __init__(
        self,
        *,
        dset: str,
        iterator: partial,
        path: Path,
        datasets: Union[dict, list],
        n_jets: int,
        n_csts: int,
        n_steps: int,
        coordinates: dict,
        min_n_csts: int = 0,
        leading: bool = True,
        recalculate_jet_from_pc: bool = False,
        incl_substruc: bool = False,
        del_r_edges: float = 0,
        boost_mopt: float = 0,
        augmentation_list: Union[str, List[str]] = "none",
        augmentation_prob: float = 1.0,
        iterator_wrapper: partial = IteratorWrapper,
        n_classes: int = None,
    ) -> None:
        """
        args:
            numpy_accessor: A hydra instantiable class that will read data from disk. Must return
            a numpy arrays of jets, constituents, mask, labels.
            See graph_coordinates function to see how these should be formatted in terms of features.
            # TODO format this better!
            dset: Either train or test
            path: Path to find the jet datasets, must contain either rodem or toptag
            datasets: Which physics processes will be loaded with which labels
            n_jets: How many jets to load in the entire dataset
            n_csts: The number of constituents to load per jet
            n_steps: number of
            coordinates: Dict of keys for which features to use in the graph
            min_n_csts: The minimum number of constituents in each jet
                - This filter is applied after data is loaded from file so it may
                result in less jets being returned than specified.
            leading: If the leading jet should be loaded, if False subleading is loaded
            recalculate_jet_from_pc: Redo jet eta, phi, pt, M using point cloud
            incl_substruc: If the substructure vars should be included (rodem only)
            del_r_edges: Build and attribute graph edges using the delta R of the nodes
            boost_mopt: Boost the jet along its axis until m/pt = X
            augmentation_list: List of order of augmentations to apply during get item
            augmentation_prob: Probability of each aug in list taking effect
        """

        # Check arguments
        if boost_mopt == -1:  # When boosting into the reference frame of the jet
            if any(tst in sr for tst in ["pt", "del"] for sr in coordinates["node"]):
                raise ValueError("Should only use xyz when boosting into jet frame!")

        # Check if the augmentation list is a string, and split using commas
        if isinstance(augmentation_list, str):
            if augmentation_list == "none":
                augmentation_list = []
            elif augmentation_list == "all":
                augmentation_list = [
                    "rotate",
                    "crop-10",
                    "merge-0.05",
                    "split-10",
                    "smear",
                    "boost-0.05",
                ]
            else:
                augmentation_list = [x for x in augmentation_list.split(",") if x]
        augmentation_list = augmentation_list.copy() or []

        # Class attributes
        self.path = path
        self.dset = dset
        self.n_nodes = n_csts
        self.datasets = datasets.copy()
        self.coordinates = deepcopy(coordinates)
        # TODO should use this
        self.min_n_csts = min_n_csts
        self.leading = leading
        self.recalculate_jet_from_pc = recalculate_jet_from_pc
        self.incl_substruc = incl_substruc
        self.del_r_edges = del_r_edges
        self.boost_mopt = boost_mopt
        self.augmentation_list = augmentation_list
        self.augmentation_prob = augmentation_prob
        self.do_augment = bool(augmentation_list)
        self.n_steps = n_steps

        # Load jets and constituents as pt, eta, phi, (M for jets)
        self.base_iterator = iterator(dset)
        self.iterator = iterator_wrapper(
            self.base_iterator,
            augmentation_list,
            augmentation_prob,
            coordinates,
            del_r_edges,
            boost_mopt,
        )
        self.n_classes = n_classes or self.iterator.n_classes

        # TODO how to do this?
        # # Check for Nan's (happens sometimes...)
        # if np.isnan(self.high_data).any():
        #     raise ValueError("Detected NaNs in the jet data!")
        # if np.isnan(self.node_data).any():
        #     raise ValueError("Detected NaNs in the constituent data!")

    def plot(self, max_events: int = 10_000) -> None:
        """Plot the collection of inputs
        Args:
            max_events: Max number of events to plot
        """
        # Create empty lists to hold the data used to fit the scalers
        # Create the plotting folder
        plot_path = Path("train_dist")
        plot_path.mkdir(parents=True, exist_ok=True)
        # TODO could call a sub numpy object

    def __iter__(self) -> tuple:
        """Retrives a jet from the dataset and returns it as a graph object
        along with the class label.

        Args:
            idx: The index of the jet to pull from the dataset
        """
        return self.iterator

    def __len__(self) -> int:
        if self.dset == "train":
            return min(self.n_steps, self.base_iterator.n_samples)
        else:
            return self.base_iterator.n_samples
