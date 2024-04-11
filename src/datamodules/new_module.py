import logging
from copy import deepcopy
from functools import partial
from typing import Mapping

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.preprocessing import PreProccessBase

log = logging.getLogger(__name__)


class RodemDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_set: partial,
        valid_set: partial,
        test_set: partial,
        preproc_fn: PreProccessBase,
        loader_kwargs: Mapping | None = None,
        plot_first_N_jets: int = 5000
    ) -> None:
        super().__init__()
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.preproc_fn = preproc_fn
        self.loader_kwargs = loader_kwargs
        self.plot_first_N_jets = plot_first_N_jets

        # Save the additional outputs that allow us to determine the data shape
        self.n_csts = self.preproc_fn.get_n_csts()
        self.n_nodes = self.n_csts
        self.dims = self.preproc_fn.get_output_shape()
        self.setup_complete = False
        self.setup("fit")

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        return self.dims[:3]

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets depending on the stage of
        training/eval."""

        if not self.setup_complete:
            if stage in ["fit", "validate"]:
                self.train_set = self.train_set(
                    n_csts=self.n_csts, preproc_fn=self.preproc_fn
                )
                self.n_classes = self.train_set.n_classes
                if hasattr(self.train_set, "reset"):
                    # This is mainly used for CWoLa style training to look at in sample performance
                    # Note that there can't be any early stopping here
                    self.valid_set = deepcopy(self.train_set)
                    self.valid_set.reset()
                else:
                    self.valid_set = self.valid_set(
                        n_csts=self.n_csts, preproc_fn=self.preproc_fn
                    )
                self.preproc_fn.plot_N_jets(self.train_set, self.plot_first_N_jets)
                self.setup_complete = True

        if stage in ["test", "predict"]:
            self.test_set = self.test_set(
                n_csts=self.n_csts, preproc_fn=self.preproc_fn
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        val_kwargs = deepcopy(self.loader_kwargs)
        val_kwargs["drop_last"] = False
        val_kwargs["shuffle"] = False
        return DataLoader(self.valid_set, **val_kwargs)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.loader_kwargs)
        test_kwargs["drop_last"] = False
        test_kwargs["shuffle"] = False
        return DataLoader(self.test_set, **test_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
