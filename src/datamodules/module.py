import logging
from copy import deepcopy
from typing import Mapping, Optional
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mattstools.mattstools.torch_utils import train_valid_split
from src.datamodules.dataset import JetData

log = logging.getLogger(__name__)


class PointCloudDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        val_frac: float = 0.1,
        data_conf: Optional[Mapping] = None,
        loader_kwargs: Optional[Mapping] = None,
        cwola: bool = False,
        cwola_frac: bool = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cwola = cwola
        self.cwola_frac = cwola_frac

        # Load a mini dataset to infer the dimensions
        mini_conf = deepcopy(self.hparams.data_conf)
        mini_conf["n_jets"] = 5
        self.mini_set = JetData(dset="test", **mini_conf)
        self.inpt_dim = self.get_dims()

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets depending on the stage of
        training/eval."""

        if stage in ["fit", "validate"]:
            dataset = JetData(dset="train", **self.hparams.data_conf)
            dataset.plot()
            self.train_set, self.valid_set = train_valid_split(
                dataset, self.hparams.val_frac
            )
            log.info(
                f"Loaded: {len(self.train_set)} train, {len(self.valid_set)} valid"
            )

            if self.cwola:
                all_indx = self.train_set.indices
                all_labels = self.train_set.dataset.labels[all_indx].reshape(-1)
                if len(np.unique(all_labels)) != 2:
                    raise Exception("Cwola can only be run with two datasets.")

                # Split labels and shuffle
                bkg_indx = all_indx[all_labels == 0]
                sig_indx = all_indx[all_labels == 1]
                np.random.shuffle(sig_indx)
                np.random.shuffle(bkg_indx)

                # Split and relabel some of the background
                n_bkg = len(bkg_indx)
                n_relabel = n_bkg // 2
                self.train_set.dataset.labels[bkg_indx[:n_relabel]] = 1

                # Select a subset of the signal
                n_sig = int(self.cwola_frac * n_bkg)
                sig_indx = sig_indx[:n_sig]

                # Reset the training index
                self.train_set.indices = np.concatenate((sig_indx, bkg_indx))

        if stage in ["test", "predict"]:
            test_conf = deepcopy(self.hparams.data_conf)
            test_conf["n_jets"] = -1
            test_conf["min_n_csts"] = 0
            test_conf["leading"] = True
            self.test_set = JetData(dset="test", **test_conf)
            log.info(f"Loaded: {len(self.test_set)} test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        edges, nodes, high, adjmat, mask, label = self.mini_set[0]
        return edges.shape[-1], nodes.shape[-1], high.shape[0]

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the input dataset."""
        return self.mini_set[0][1].shape[-2]

    @property
    def n_classes(self) -> int:
        """Return the number of jet types/classes used in training."""
        return self.mini_set.n_classes
