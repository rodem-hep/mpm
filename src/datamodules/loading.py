from abc import abstractmethod
from copy import deepcopy
from itertools import cycle, islice
from pathlib import Path

# TODO this first function doesn't work for some reason
from typing import Dict, Mapping, Tuple, Union

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from torch.utils.data import Dataset
from src.datamodules.preprocessing import NoPreProccessing, PreProccessBase


from src.jet_utils import cstptetaphi_to_jet, cstpxpypz_to_jet, pxpypz_to_ptetaphi


def process_csts(cst_data, n_csts=-1):
    """Given cst_data of shape [n_samples, n_csts, 4] Where the last axis has
    e, px, py, pz, convert to pt, eta, phi and calculate the jet level
    information."""
    # Splitting in this way does not result in any memory copy
    cst_e = cst_data[..., 0:1]
    cst_px = cst_data[..., 1:2]
    cst_py = cst_data[..., 2:3]
    cst_pz = cst_data[..., 3:4]

    # Calculate the overall jet kinematics from the constituents
    jet_px, jet_py, jet_pz, jet_m, _ = cstpxpypz_to_jet(cst_px, cst_py, cst_pz, cst_e)

    # Limit constituent data to the number of requested nodes
    cst_px = cst_px[:, :n_csts]
    cst_py = cst_py[:, :n_csts]
    cst_pz = cst_pz[:, :n_csts]

    # Convert both sets of values to spherical
    cst_pt, cst_eta, cst_phi = pxpypz_to_ptetaphi(cst_px, cst_py, cst_pz)
    jet_pt, jet_eta, jet_phi = pxpypz_to_ptetaphi(jet_px, jet_py, jet_pz)

    # Combine the information and return
    cst_data = np.concatenate([cst_pt, cst_eta, cst_phi], axis=-1)
    jet_data = np.vstack([jet_pt, jet_eta, jet_phi, jet_m]).T
    return cst_data, jet_data


def load_jetclass(filepath, treename=None, n_csts: int = 64, features: list = None):
    """Load a file from the JetClass dataset in a way that is consistent with
    RODEM loading. Available features (accessed using branches):

    ['part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta',
    'part_dphi', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr',
    'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon', 'label_QCD',
    'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql',
    'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl', 'jet_pt',
    'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles', 'jet_sdmass',
    'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
    'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
    'aux_truth_match']
    """

    branches = ["part_energy", "part_px", "part_py", "part_pz"]
    if features is not None:
        branches += features
    all_labels = [
        "label_QCD",
        "label_Tbl",
        "label_Tbqq",
        "label_Wqq",
        "label_Zqq",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
    ]

    with uproot.open(filepath) as f:
        if treename is None:
            treenames = {
                k.split(";")[0]
                for k, v in f.items()
                if getattr(v, "classname", "") == "TTree"
            }
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    "Need to specify `treename` as more than one trees are found in file %s: %s"
                    % (filepath, str(treenames))
                )
        tree = f[treename]
        outputs = tree.arrays(filter_name=branches, library="ak")
        labels = tree.arrays(filter_name=all_labels, library="pd")

    # awk_arr = ak.fill_none(ak.pad_none(outputs, 64, clip=True), 0)
    # Note, unless the clip value is set to be at or above the maximum number of nodes
    # awkward will set some masks all to zero
    awk_arr = ak.pad_none(outputs, n_csts, clip=True)
    part_data = np.stack(
        [ak.to_numpy(awk_arr[n]).astype("float32").data for n in branches], axis=1
    ).transpose(0, 2, 1)

    nan_mx = np.isnan(part_data)
    # TODO check that there aren't any that don't all have zeros?
    mask = ~np.any(nan_mx, axis=-1)
    part_data[nan_mx] = 0

    trans_data, jet_data = process_csts(part_data, n_csts)
    if features is not None:
        part_data = np.concatenate([trans_data, part_data[..., 4:]], axis=-1)
    else:
        part_data = trans_data
    mask = mask[:, :n_csts]

    # Just use the index as the label, not dealing with one hots here
    labels = labels[all_labels].to_numpy().astype(np.float32).argmax(axis=1)

    return jet_data, part_data, mask, labels


# In newer versions of itertools this is in the package.
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def load_files(file_paths, load_function, n_csts, shuffle=True, features=None):
    data_list = []
    for file in file_paths:
        data_list += [load_function(file, n_csts=n_csts, features=features)]
    data = [np.vstack([d[i] for d in data_list]) for i in range(len(data_list[0]))]
    # Reshape the labels
    data[-1] = data[-1].reshape(-1, 1)
    if shuffle:
        indx = np.random.permutation(len(data[0]))
        data = [d[indx] for d in data]
    return data


class IteratorBase:
    @abstractmethod
    def __next__(self):
        # return jet_data, part_data, mask, labels
        raise NotImplemented()

    @abstractmethod
    def get_nclasses(self):
        # Return number of classes in the dataset
        raise NotImplemented()


class JetClassIterator(IteratorBase):
    def __init__(
        self,
        dset: str,
        n_load: int,
        path: str,
        n_nodes: int = 64,
        processes: list = None,
        max_files: int = None,
        features: list = None,
    ) -> None:
        """
        dset: string [train, test, val]
        n_load: Number of files to load. When using an iterator data is stored in many different files, the data in each file often isn't shuffled (containing samples all from one class for example). So load several files and shuffle the loaded samples.
        """
        self.n_load = n_load
        self.dset = dset
        self.features = features
        # Get the path to the set of files to load
        # TODO unhardcode
        data_path = Path(path)
        if dset == "train":
            direct = data_path / "train_100M"
            self.n_samples = 100_000_000
        elif dset == "test":
            direct = data_path / "test_20M"
            # While debugging don't evaluate on everything
            self.n_samples = 10_000_000
        else:
            direct = data_path / "val_5M"
            self.n_samples = 1_000_000
        proc_dict = {
            "QCD": ["ZJets"],
            "WZ": ["ZTo", "WTo"],
            "ttbar": ["TTBar_", "TTBarLep"],
            "higgs": ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q"],
        }
        if isinstance(processes, str):
            processes = [processes]
        elif processes == None:
            processes = proc_dict.keys()
        self.file_list = []
        self.processes = processes
        for process in processes:
            proc = proc_dict[process]
            # Load the files
            for pr in proc:
                proc_files = np.array(list(direct.glob(f"{pr}*.root")))
                if dset == "test":
                    # Order the files by number
                    proc_files = proc_files[
                        np.argsort(
                            [int(file.stem.split("_")[-1]) for file in proc_files]
                        )
                    ]
                self.file_list += [proc_files]
        # Make an interleaved shuffled (glob grabs randomly) list of the files
        stacked_files = np.array(self.file_list).transpose()
        if (max_files is not None) and (dset == "train"):
            # Select max files files per process
            stacked_files = stacked_files[:max_files]
        # stacked_files = stacked_files[:1]
        self.n_samples = int(1e5) * np.prod(stacked_files.shape)
        self.file_list = stacked_files.flatten().tolist()
        # Build an infinite iterator over the file list
        # TODO using cycle slightly minimising the amount of shuffling that is done, can you do better?
        self.file_iterator = batched(cycle(self.file_list), self.n_load)
        # TODO can this class be generalised?
        self.n_nodes = n_nodes
        self.load_func = load_jetclass
        # Set the index
        self.data_i = 0
        self.load_data()

    def load_data(self):
        # Reset the counting index
        self.data_i = 0
        files = next(self.file_iterator)
        # This returns: self.data = (jet_data, part_data, mask, labels)
        self.data = load_files(
            files,
            self.load_func,
            self.n_nodes,
            shuffle=self.dset != "test",
            features=self.features,
        )

        # TODO this is a hacky shit thing to do
        if len(self.processes) == 1:
            label = self.data[-1]
        elif len(self.processes) == 4:
            label = self.data[-1]
        else:
            label = np.copy(self.data[-1])
            label -= 1
            # This only happens in pretraining
            label[label < 0] = 0
        self.data[-1] = label

    def get_sample(self):
        sample = [d[self.data_i] for d in self.data]
        self.data_i += 1
        return sample

    def get_nclasses(self):
        # TODO this is also hacky and stupid
        return 4

    def __next__(self):
        try:
            data = self.get_sample()
        except IndexError:
            self.load_data()
            data = self.get_sample()
        return data


class JetClassMappable(Dataset):
    def __init__(
        self,
        dset: str,
        n_load: int,
        path: str,
        n_nodes: int = 64,
        n_samples: int = None,
        processes: list = None,
        max_files: int = None,
        features: list = None,
        n_csts: int = None,
        preproc_fn: PreProccessBase = NoPreProccessing(),
    ) -> None:
        """
        dset: string [train, test, val]
        n_load: Number of files to load. When using an iterator data is stored in many different files, the data in each file often isn't shuffled (containing samples all from one class for example). So load several files and shuffle the loaded samples.
        max_files: Maximum number of files to load per process
        """
        self.n_load = n_load
        self.dset = dset
        self.features = features
        self.preproc_fn = preproc_fn
        # Get the path to the set of files to load
        # TODO unhardcode
        data_path = Path(path)
        if dset == "train":
            direct = data_path / "train_100M"
        elif dset == "test":
            direct = data_path / "test_20M"
            # While debugging don't evaluate on everything
        else:
            direct = data_path / "val_5M"
        proc_dict = {
            "QCD": ["ZJets"],
            "WZ": ["ZTo", "WTo"],
            "ttbar": ["TTBar_", "TTBarLep"],
            "higgs": ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q"],
        }
        if isinstance(processes, str):
            processes = [processes]
        elif (processes == None) or (processes[0] == None):
            processes = proc_dict.keys()
        self.file_list = []
        self.processes = processes
        self.n_classes = 0
        for process in processes:
            proc = proc_dict[process]
            # Load the files
            for pr in proc:
                self.n_classes += 1
                proc_files = np.array(list(direct.glob(f"{pr}*.root")))
                if dset == "test":
                    # Order the files by number
                    proc_files = proc_files[
                        np.argsort(
                            [int(file.stem.split("_")[-1]) for file in proc_files]
                        )
                    ]
                self.file_list += [proc_files]
        # Make an interleaved shuffled (glob grabs randomly) list of the files
        stacked_files = np.array(self.file_list).transpose()
        if (max_files is not None) and (dset == "train"):
            # Select max files files per process
            stacked_files = stacked_files[:max_files]
        self.file_list = stacked_files.flatten().tolist()
        if n_samples is not None:
            self.n_samples = int(n_samples // self.n_classes)
        else:
            self.n_samples = None
        self.n_nodes = n_nodes
        self.load_func = load_jetclass
        self.load_data()

    def load_data(self):
        # This returns: self.data = (jet_data, part_data, mask, labels)
        self.data = load_files(
            self.file_list,
            self.load_func,
            self.n_nodes,
            shuffle=self.dset != "test",
            features=self.features,
        )

        # Select self.n_samples sample from each of the classes
        if self.n_samples is not None:
            label = self.data[-1]
            indxs = []
            for lbl in np.unique(label):
                mask = label == lbl
                # Randomly select self.n_samples where mask is True
                indxs += [
                    np.random.choice(np.where(mask)[0], self.n_samples, replace=False)
                ]
            indx = np.hstack(indxs)
            # Randomly shuffle the samples
            np.random.shuffle(indx)
            self.data = [d[indx] for d in self.data]

        # TODO this is a hacky shit thing to do specific that doesn't even really work!
        if len(self.processes) == 1:
            process = self.processes[0]
            if process == "QCD":
                label = np.zeros_like(self.data[-1])
            elif process == "ttbar":
                label = np.ones_like(self.data[-1])
            elif process == "WZ":
                label = 2 * np.ones_like(self.data[-1])
        elif len(self.processes) == 4:
            label = self.data[-1]
        else:
            label = np.copy(self.data[-1])
            label -= 1
            # This only happens in pretraining
            label[label < 0] = 0
        self.data[-1] = label.reshape(-1)

    def __len__(self) -> int:
        return len(self.data[-1])

    def __getitem__(self, idx: int) -> tuple:
        high, nodes, mask, label = [d[idx] for d in self.data]
        return self.preproc_fn(nodes, high, label, mask)

    def get_nclasses(self):
        return self.n_classes


if __name__ == "__main__":
    # This is a test on baobab
    load_jetclass(
        "/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/JetClass/Pythia/train_100M/HToGG_022.root"
    )
    # # A simple plot maker
    # import matplotlib.pyplot as plt
    # jet_data, part_data, mask, labels = self.data
    # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    # all_csts = part_data[mask]

    # ax[0].hist(np.log(all_csts[:, 0] + 1), bins=100)
    # ax[0].set_yscale('log')
    # ax[1].hist(all_csts[:, 1], bins=100)
    # ax[2].hist(all_csts[:, 2], bins=100)
    # # plt.xscale('log')
    # plt.savefig(files[0].parent.parent / 'constituents.png')
    # plt.close(fig)
