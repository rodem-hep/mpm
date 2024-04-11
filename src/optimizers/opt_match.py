from pytorch_lightning.callbacks.callback import Callback


class MatchOptimizers(Callback):
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx
    ) -> None:
        if batch_idx == 0:
            pl_module.optimizers().param_groups[0]["lr"] = trainer.optimizers[
                0
            ].param_groups[0]["lr"]
        else:
            trainer.optimizers[0].param_groups[0][
                "lr"
            ] = pl_module.optimizers().param_groups[0]["lr"]
