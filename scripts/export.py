from copy import deepcopy

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from pathlib import Path

import h5py
import hydra
import torch as T
from omegaconf import DictConfig

from mattstools.mattstools.hydra_utils import reload_original_config
from mattstools.mattstools.torch_utils import to_np

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best)

    log.info("Loading best checkpoint")
    device = "cuda" if T.cuda.is_available() else "cpu"
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)

    log.info("Instantiating original trainer")
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    # Allow running on single process
    if isinstance(cfg.processes, str):
        cfg.processes = [cfg.processes]

    # Cycle through the datasets and create the dataloader
    for proc in cfg.processes:
        log.info(f"Instantiating the test data module for {proc}")
        dataconf = deepcopy(orig_cfg.datamodule)
        dataconf.update(cfg.datamodule)
        dataconf["test_set"]["processes"] = [proc]
        datamodule = hydra.utils.instantiate(dataconf)

        log.info("Running inference")
        outputs = trainer.predict(model=model, datamodule=datamodule)

        log.info("Combining predictions across dataset")
        scores = list(outputs[0].keys())
        score_dict = {score: T.vstack([o[score] for o in outputs]) for score in scores}

        log.info("Saving outputs")
        output_dir = Path(orig_cfg.paths.full_path, "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_dir / f"{proc}_test.h5", mode="w") as file:
            for score in scores:
                file.create_dataset(score, data=to_np(score_dict[score]))


if __name__ == "__main__":
    main()
