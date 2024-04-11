"""Pytorch Dataset definitions of various collections training samples."""

import logging
import random
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from pyparsing import Generator
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from mattstools.mattstools.numpy_utils import unison_shuffled_copies
from src.datamodules.preprocessing import NoPreProccessing, PreProccessBase

log = logging.getLogger(__name__)


def get_file_list(
    processes: list,
    path: Path,
    n_files: int | list | None = None,
    n_start: int | list | None = 0,
) -> list:
    """Load the list of files for each process. If n_files is not specified,
    will always makes sure that the number of files for each process is
    balanced.

    Parameters
    ----------
    processes : list
        List of processes to load files for.
    path : Path
        Path to the directory containing the files.
    n_files : int | list | None, optional
        Number of files to load for each process.
        If a single integer is provided, it will be used for all processes.
        If a list is provided, it must have the same length as `processes`.
        If `None`, the number of files will be automatically balanced to limit.
        Default is `None`.
    n_start : int | list | None, optional
        Starting index of the files to load for each process.
        If a single integer is provided, it will be used for all processes.
        If a list is provided, it must have the same length as `processes`.
        Default is 0.

    Returns
    -------
    list
        List of lists, where each sublist contains the file paths for one process.
    """

    # Turn start into list for generality
    n_proc = len(processes)
    if isinstance(n_start, int):
        n_start = n_proc * [n_start]

    # Autobalance the dataset
    if n_files is None:
        log.info("Checking file list to ensure balanced dataset")
        n_files = 9999999
        for i in range(n_proc):
            proc_files = list((path / processes[i]).glob("*.h5"))
            nfs = len(proc_files)
            n_files = min(n_files, nfs - n_start[i])

    # Turn into a list for generality
    if isinstance(n_files, int):
        n_files = n_proc * [n_files]

    # Fill in the file list such that each has a seperate sub list
    file_list = [[] for _ in processes]
    for i in range(n_proc):
        log.info(f"Loading {n_files[i]-n_start[i]} files for {processes[i]}")
        for j in range(n_start[i], n_files[i]):
            file_list[i].append(path / processes[i] / f"{processes[i]}_{j}.h5")
    return file_list


def load_filedata_into_mem(
    file_list: list,
    n_csts: int | None = None,
    n_jets: int | None = None,
    disable: bool = True,
) -> tuple:
    """Load data from files into memory.

    Parameters
    ----------
    n_csts : int | None
        Number of constituents to load.
    file_list : list
        List of files for each process to load
    n_jets : int | None
        Number of jets to load from each file
    disable : bool
        Prevents tqdm outputs

    Returns
    -------
    tuple
        A tuple containing the constituents, high level variables and labels.
    """

    # Initiate empty lists for loading
    csts = []
    high = []
    labels = []

    # Cycle through all of the processes and their file_lists
    for p, files in enumerate(file_list):
        # Load each one into memory
        for file in tqdm(
            files, f"loading files from process {p} into memory", disable=disable
        ):
            with h5py.File(file, mode="r") as ifile:
                csts.append(ifile["csts"][:n_jets, :n_csts].astype(np.float32))
                high.append(ifile["hlvs"][:n_jets].astype(np.float32))
                labels += [p] * len(high[-1])

    # Stack them together
    csts = np.vstack(csts)
    high = np.vstack(high)
    labels = np.hstack(labels)

    return csts, high, labels


class RODEMBase:
    """The base class for configuring the datasets."""

    def __init__(
        self,
        *,
        dset: str,
        path: str,
        processes: list,
        n_files: int | list | None = None,
        n_csts: int | None = None,
        n_jets: int | None = None,
        n_start: int | list = 0,
        preproc_fn: PreProccessBase = NoPreProccessing(),
    ) -> None:
        """
        Parameters
        ----------
        dset : str
            Either train or test.
        path : Path
            Path to the rodem datasets (folder contains train/ and test/).
        processes : list
            Which physics processes will be loaded.
        n_files : int | list | None
            How many files to load from each processes, each file has 100k jets.
        n_csts : int
            The number of constituents to load per jet.
        n_jets : int
            The number of jets to load per file.
        preproc_fn : partial
            Preprocessing function to apply in the getitem method.
        n_start : int | list
            The idx of the file to start from when loading multiple files
            Useful for defining orthogonal train/val splits
        incl_substruc : bool, optional
            If the substructure vars should be included (rodem only), by default False.
        """

        # Class attributes
        self.dset = dset
        self.path = Path(path)
        self.processes = processes
        self.n_files = n_files
        self.n_start = n_start
        self.n_csts = n_csts
        self.n_jets = n_jets
        self.preproc_fn = preproc_fn
        self.n_classes = len(processes)

        # Get the full paths of every file that makes up this dataset
        if dset == "train":
            self.file_list = get_file_list(
                processes, self.path / "train", n_files=n_files, n_start=n_start
            )
        else:
            self.file_list = [[self.path / "test" / f"{p}_test.h5"] for p in processes]

    def get_n_classes(self) -> int:
        """Count the number of classes used in this dataset."""
        return len(self.processes)


class RODEMMappable(Dataset, RODEMBase):
    """A pytorch mappable dataset object for the rodem jets dataset."""

    def __init__(self, *args, n_samples: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load the data from the hdf files
        self.csts, self.high, self.labels = load_filedata_into_mem(
            self.file_list, self.n_csts, disable=False
        )
        # Always shuffle this data manually
        self.shuffle_samples()
        if n_samples is not None:
            self.csts, self.high, self.labels = (
                self.csts[:n_samples],
                self.high[:n_samples],
                self.labels[:n_samples],
            )
        self.mask = np.any(self.csts != 0, axis=-1)
        log.info(f"Loaded {len(self)} jets from {len(self.file_list)} files")

    def shuffle_samples(self):
        # Shuffle the data randomly
        indx = np.random.permutation(len(self.csts))
        self.csts, self.high, self.labels = (
            self.csts[indx],
            self.high[indx],
            self.labels[indx],
        )
        self.mask = np.any(self.csts != 0, axis=-1)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves an item and applies the pre-processing function."""

        # Load the particle constituent and high level jet information from the data
        csts = self.csts[idx]
        high = self.high[idx]
        label = self.labels[idx]
        mask = self.mask[idx]

        # Pass through and return the pre-processing function
        return self.preproc_fn(csts, high, label, mask)


class RODEMIdealCwola(RODEMMappable):
    def __init__(self, *args, n_signal: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This class will have mixed samples in the training set
        if self.dset == "train":
            self.true_labels = np.copy(self.labels)
            bkg_mx = self.true_labels == 0
            n_split = sum(bkg_mx) // 2
            self.bkg = self.csts[bkg_mx], self.high[bkg_mx]
            half_0 = [bb[:n_split] for bb in self.bkg]
            half_1 = [bb[n_split:] for bb in self.bkg]
            self.signal = self.csts[~bkg_mx][:n_signal], self.high[~bkg_mx][:n_signal]
            half_1 = [np.vstack((c1, c0)) for c1, c0 in zip(half_1, self.signal)]
            self.csts, self.high = [
                np.vstack((c1, c0)) for c1, c0 in zip(half_1, half_0)
            ]
            self.labels = np.ones(len(self.csts), dtype=np.int64)
            self.labels[:n_split] = 0
            self.shuffle_samples()

    def reset(self):
        # Reset the labels to the correct labels
        self.csts, self.high = [
            np.vstack((c1, c0)) for c1, c0 in zip(self.bkg, self.signal)
        ]
        self.labels = np.concatenate(
            (np.zeros(len(self.bkg[0])), np.ones(len(self.signal[0])))
        )
        self.shuffle_samples()


class RODEMIterable(IterableDataset, RODEMBase):
    """A pytorch iterable dataset object for the rodem jets dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_total = self._count_samples()
        log.info(f"Streaming {len(self)} jets from {np.size(self.file_list)} files")

    def _count_samples(self) -> int:
        """Counts the total number of samples in the entire dataset.

        Relatively quick as there is no big I/O
        """
        total = 0
        for p, file_list in enumerate(self.file_list):
            for file in file_list:
                with h5py.File(file, mode="r") as ifile:
                    total += len(ifile["hlvs"])
        return total

    def __len__(self) -> int:
        return self.n_total

    def __iter__(self) -> Generator:
        """Called seperately for each worker (thread)

        - Divides up the file for each worker
        - Loads 1 file per process at a time into a buffer
        - Shuffles the buffer
        - Divides up the buffers into batches
        - Returns each batch
        """

        # Check if we are using single-process vs multi process data loading
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Calculate which files this worker is responsible for
        file_list = deepcopy(self.file_list)  # Make a copy so suffling can be done
        worker_files = [np.array_split(fl, num_workers)[worker_id] for fl in file_list]

        # Cycle through the files for a buffer
        for b_files in zip(*worker_files):
            b_files = [[b] for b in b_files]

            # Load the data into a buffer and shuffle
            b_csts, b_high, b_labels = load_filedata_into_mem(b_files, self.n_csts)
            b_csts, b_high, b_labels = unison_shuffled_copies(b_csts, b_high, b_labels)

            # Yeild each batch from the buffer
            for csts, high, labels in zip(b_csts, b_high, b_labels):
                mask = np.any(csts != 0, axis=-1)

                yield self.preproc_fn(csts, high, labels, mask)

        # Final task of the first worker is to shuffle the dataset for the next epoch
        if worker_id == 0:
            for fl in self.file_list:
                random.shuffle(fl)
