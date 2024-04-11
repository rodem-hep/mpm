from torch.optim import Adam
from pytorch_lightning.callbacks import BaseFinetuning


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if (current_epoch == self._unfreeze_at_epoch) and pl_module.train_backbone:
            self.unfreeze_and_add_param_group(
                modules=pl_module.backbone,
                optimizer=optimizer,
                train_bn=True,
            )


class UnFreezeIterable(FeatureExtractorFreezeUnfreeze):
    def __init__(self, unfreeze_at_step=int(1e6)):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_step

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Called when the epoch begins."""
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            num_param_groups = len(optimizer.param_groups)
            super(UnFreezeIterable, self).finetune_function(
                pl_module, batch_idx, optimizer
            )
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)
