import logging
from copy import deepcopy
from typing import Mapping, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datamodules.iterable_dataset import JetData

log = logging.getLogger(__name__)


class PointCloudDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        val_frac: float = 0.1,
        data_conf: Optional[Mapping] = None,
        loader_kwargs: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load a mini dataset to infer the dimensions
        mini_conf = deepcopy(self.hparams.data_conf)
        mini_conf["n_jets"] = 5
        self.mini_set = JetData(dset="test", **mini_conf)
        self.inpt_dim = self.get_dims()

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets depending on the stage of
        training/eval."""

        if stage in ["fit", "validate"]:
            self.train_set = JetData(dset="train", **self.hparams.data_conf)
            self.valid_set = JetData(dset="val", **self.hparams.data_conf)
            self.train_set.plot()
            log.info(
                f"Loaded: {len(self.train_set)} train, {len(self.valid_set)} valid"
            )

        if stage in ["test", "predict"]:
            test_conf = deepcopy(self.hparams.data_conf)
            test_conf["n_jets"] = -1
            test_conf["min_n_csts"] = 0
            test_conf["leading"] = True
            self.test_set = JetData(dset="test", **test_conf)
            log.info(f"Loaded: {len(self.test_set)} test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        edges, nodes, high, adjmat, mask, label = next(self.mini_set.iterator)
        return edges.shape[-1], nodes.shape[-1], high.shape[0]

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the input dataset."""
        return self.mini_set.n_nodes

    @property
    def n_classes(self) -> int:
        """Return the number of jet types/classes used in training."""
        return self.mini_set.n_classes
