from pathlib import Path
from pytorch_lightning.callbacks import Callback

import torch as T

import h5py

from mattstools.mattstools.torch_utils import to_np


class EvalEnd(Callback):
    def __init__(self, dirpath: Path) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        pl_module.eval()
        with T.no_grad():
            output = []
            label = []
            for idx, batch in enumerate(trainer.val_dataloaders):
                batch = pl_module.transfer_batch_to_device(
                    batch, pl_module.device, dataloader_idx=0
                )
                out_dict = pl_module.predict_step(batch, idx)
                output += [out_dict["output"].cpu()]
                label += [out_dict["label"].cpu()]
        score_dict = {
            "output": T.concat(output),
            "label": T.concat(label),
        }
        with h5py.File(self.dirpath / f"end.h5", mode="w") as file:
            for score in score_dict.keys():
                file.create_dataset(score, data=to_np(score_dict[score]))
