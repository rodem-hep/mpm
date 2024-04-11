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
    version_base=None,
    config_path=str(root / "configs"),
    config_name="jetclass_export.yaml",
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    # TODO should load best but isn't saved for transformer!
    # orig_cfg = reload_original_config(cfg, get_best=False)
    orig_cfg = reload_original_config(cfg, get_best=True)

    # Evaluate on a different datset if passed
    reset = hasattr(cfg, "datamodule")
    if reset:
        # TODO should be able to do this smarter and point to saved config somewhere
        orig_cfg.datamodule = cfg.datamodule

    log.info("Loading best checkpoint")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    # Cycle through the datasets and create the dataloader
    for dataset in cfg.datasets:
        output_dir = Path(orig_cfg.paths.full_path, "output_jetclass")
        output_dir.mkdir(parents=True, exist_ok=True)
        if dataset == None:
            data_nm = "None"
        elif len(dataset) > 1:
            data_nm = "_".join(dataset)
        else:
            data_nm = dataset
        data_file = output_dir / f"{data_nm}_test{cfg.tag}.h5"
        if cfg.resume:
            if data_file.is_file():
                continue
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.iterator.processes = dataset
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

        log.info("Instantiating the trainer")
        trainer = hydra.utils.instantiate(orig_cfg.trainer)

        log.info("Running the prediction loop")
        outputs = trainer.predict(model=model, datamodule=datamodule)

        log.info("Combining predictions across dataset")
        scores = list(outputs[0].keys())
        score_dict = {score: T.vstack([o[score] for o in outputs]) for score in scores}

        log.info("Saving outputs")
        with h5py.File(data_file, mode="w") as file:
            for score in scores:
                file.create_dataset(score, data=to_np(score_dict[score]))


if __name__ == "__main__":
    main()
