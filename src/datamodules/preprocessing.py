"""Classes that define the pre-processing of the jets for the models."""

import abc
import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data._utils.collate import default_collate

from mattstools.mattstools.plotting import plot_multi_hists_2
from src.datamodules.preproc_functions import preprocess_as_graphs

log = logging.getLogger(__name__)


def group_collate(x: iter, max_n: int) -> tuple:
    """Collate the first elements of an iterable Here we use the very versitle
    pytorch collation, then just convert back to numpy Very quick!"""
    samples = [s for _, s in zip(range(max_n), x)]
    samples = default_collate(samples)
    samples = [s.numpy() for s in samples]
    return samples


class PreProccessBase(abc.ABC):
    """Base class for preprocessing, should contain all common methods used by
    preprocessing.

    - call: Takes in jet information and returns the data as needed by the network
    - get_output_shape: Needed as network arcitectures require knowing dimensions
    - get_output_labels: Needed for plotting purposes by the network
    - get_n_csts: Needed to determine the number of consituents used
    - plot_N_jets (optional): Called at the start of training by datamodule
        Usefull for plotting the jets for debugging purposes
    """

    @abc.abstractmethod
    def __call__(
        self, csts: np.ndarray, high: np.ndarray, label: int, mask: np.ndarray
    ) -> tuple:
        """Takes in a single jet and applies the specific preprocessing.

        Parameters
        ----------
        csts : np.ndarray
            Must have shape (num_csts, 3) containing pt, eta, phi
        high : np.ndarray
            Must have shape 4 or 11 containing pt, eta, phi, mass, (substruc)
        label : int
            Single int for the class idx
        mask : np.ndarray
            Must have shape (num_csts)

        Returns
        -------
        Tuple of information containing preprecessed info
        """
        pass

    @abc.abstractmethod
    def get_output_shape(self) -> tuple:
        """Returns a tuple of shapes describing the output of __call__ Only
        need to return shapes that help with model setup."""
        pass

    @abc.abstractmethod
    def get_output_labels(self) -> tuple:
        """Returns a tuple of strings describing the output of __call__
        Primarily for diagnostics and plotting."""
        pass

    @abc.abstractmethod
    def get_n_csts(self) -> int:
        """Return the number of constituents used in producing a sample Must be
        defined!"""
        pass

    def plot_N_jets(self, train_set: iter, max_n: int) -> None:
        """Should plot and save the first N jets.

        This is called at the start of training to visualise the data
        given to the network.
        """
        log.info("No plotting method found for this pre-processing function!")


class NoPreProccessing(PreProccessBase):
    def __init__(self, n_csts: int = 64) -> None:
        super().__init__()
        self.n_csts = n_csts

    def __call__(
        self, csts: np.ndarray, high: np.ndarray, label: int, mask: np.ndarray
    ) -> tuple:
        return csts[: self.n_csts], high[:4], label, mask[: self.n_csts]

    def get_output_shape(self) -> tuple:
        return 3, 4, 1, 1

    def get_output_labels(self) -> tuple:
        return ["pt", "eta", "phi"], ["pt", "eta", "phi", "mass"], "label", "mask"

    def get_n_csts(self) -> int:
        return self.n_csts

    def plot_N_jets(self, train_set: iter, max_n: int, path: str = ".") -> None:
        csts, high, label, mask = group_collate(train_set, max_n)

        # Seperate it all out per class
        classes = np.unique(label)
        c_data = [[], []]
        for c in classes:
            class_mask = label == c
            c_data[0].append(csts[class_mask][mask[class_mask]])
            c_data[1].append(high[class_mask])

        # Plot per class
        for t, data, labels in zip(["csts", "high"], c_data, self.get_output_labels()):
            plot_multi_hists_2(
                data_list=data,
                data_labels=classes,
                col_labels=labels,
                logy=True,
                path=Path(path, t + ".png"),
            )


class PreProccessGraphs(PreProccessBase):
    def __init__(
        self,
        n_csts: int = 64,
        del_r_edges: float = 0.0,
        edge_coords: list | None = None,
        csts_coords: list | None = None,
        high_coords: list | None = None,
        transformation_list: list | None = None,
        transformation_prob: list | None = None,
    ) -> None:
        super().__init__()
        self.n_csts = n_csts
        self.del_r_edges = del_r_edges
        self.edge_coords = edge_coords
        self.csts_coords = csts_coords
        self.high_coords = high_coords
        self.transformation_list = transformation_list
        self.transformation_prob = transformation_prob

        if self.edge_coords and self.del_r_edges == 0:
            log.warning("Requesting edge features but with no del_r!")
            log.warning("Setting edge features to zero!")
            self.edge_coords = []

    def __call__(
        self, csts: np.ndarray, high: np.ndarray, label: int, mask: np.ndarray
    ) -> tuple:
        return preprocess_as_graphs(
            csts,
            high,
            label,
            mask,
            n_csts=self.n_csts,
            del_r_edges=self.del_r_edges,
            edge_coords=self.edge_coords,
            csts_coords=self.csts_coords,
            high_coords=self.high_coords,
            transformation_list=self.transformation_list,
            transformation_prob=self.transformation_prob,
        )

    def get_output_shape(self) -> tuple:
        return (
            len(self.edge_coords),
            len(self.csts_coords),
            len(self.high_coords),
            1,
            1,
            1,
        )

    def get_output_labels(self) -> tuple:
        return self.edge_coords, self.csts_coords, self.high_coords

    def get_n_csts(self) -> int:
        return self.n_csts

    def plot_N_jets(self, train_set: iter, max_n: int, path: str = ".") -> None:
        edges, csts, high, adjmat, mask, label = group_collate(train_set, max_n)

        # Seperate it all out per class
        classes = np.unique(label)
        c_data = [[], [], []]
        for c in classes:
            class_mask = label == c
            c_data[0].append(edges[class_mask][adjmat[class_mask]][:100_000])
            c_data[1].append(csts[class_mask][mask[class_mask]])
            c_data[2].append(high[class_mask])

        # Plot per class
        for t, data, labels in zip(
            ["edges", "csts", "high"], c_data, self.get_output_labels()
        ):
            # Dont plot if empty
            if np.prod(data[0].shape) == 0:
                continue

            plot_multi_hists_2(
                data_list=data,
                data_labels=classes,
                col_labels=labels,
                logy=True,
                path=Path(path, t + ".png"),
            )
