import math
from functools import partial
from pathlib import Path
from typing import Mapping

import hydra
import numpy as np
import torch as T
import wandb
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy, Perplexity

from mattstools.mattstools.hydra_utils import reload_original_config
from mattstools.mattstools.modules import NewIterativeNormLayer, SingleLinear
from mattstools.mattstools.torch_utils import get_sched
from mattstools.mattstools.transformers import ClassAttention, FullTransformerEncoder


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", T.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return T.cat([f.cos(), f.sin()], dim=-1)


class WrPerplexity(Perplexity):
    def forward(self, pred, label, *args, **kwargs):
        return super().forward(pred.unsqueeze(1), label.view(-1, 1), *args, **kwargs)


class Bert(LightningModule):
    """A class for implementing the Bert pretraining model."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        encoder: partial,
        max_mask: float,
        normaliser_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        positional_encoder: partial = None,
        order_inputs: bool = False,
        use_class_weights: int = 0,
        output_head_config: Mapping = None,
        linear_track: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]

        # self.n_pos_enc = 3 * 64
        self.n_pos_enc = self.node_dim

        self.n_classes = n_classes
        self.max_mask = max_mask
        self.use_class_weights = use_class_weights
        # TODO this isn't learnable unless you can backprob through the iterative norm layer
        self.masked_token = nn.Parameter(T.ones(1, self.node_dim))
        self.register_buffer("class_weights", T.ones(self.n_classes))
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", weight=self.class_weights)

        # Define a look up table for the number of nodes to mask for each padding length
        self.register_buffer("count_mask", T.zeros((n_nodes, n_nodes), dtype=T.bool))
        for i in range(n_nodes):
            self.count_mask[i, : int(self.max_mask * (i + 1))] = 1

        # The layers which normalise the input data
        self.node_norm = NewIterativeNormLayer(self.node_dim, **normaliser_config)
        if self.edge_dim:
            self.edge_norm = NewIterativeNormLayer(self.edge_dim, **normaliser_config)
        if self.high_dim:
            self.high_norm = NewIterativeNormLayer(self.high_dim, **normaliser_config)

        # self.positional_encoder = FourierFeatures(self.node_dim, self.n_pos_enc)
        self.encoder = encoder(
            inpt_dim=self.n_pos_enc,
            outp_dim=self.n_classes,
            edge_dim=self.edge_dim,
            ctxt_dim=self.high_dim,
        )
        self.order_inputs = order_inputs
        if self.order_inputs:
            self.pos_encoding = nn.Parameter(T.zeros((1, n_nodes, self.n_pos_enc)))
        if positional_encoder is not None and isinstance(positional_encoder, partial):
            # Don't order the inputs if we are using the positional encoder
            self.order_inputs = False
            self.pos_encode = True
            self.pos_encoding = nn.Parameter(T.zeros((1, n_nodes, self.n_classes)))
            self.positional_encoder = positional_encoder(
                inpt_dim=self.n_classes,
                outp_dim=self.n_classes,
            )
        else:
            self.pos_encode = False

        # TODO should we use the .compute method of these metrics to track them?
        self.metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=self.n_classes),
                "w_accuracy": Accuracy(
                    task="multiclass", num_classes=self.n_classes, average="weighted"
                ),
                "top_5": Accuracy(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=min(5, self.n_classes),
                ),
                "top_10": Accuracy(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=min(10, self.n_classes),
                ),
                "perplexity": WrPerplexity(),
            }
        )
        n_c = 10
        self.track_metrics = nn.ModuleDict(
            {
                "accuracy_track": Accuracy(task="multiclass", num_classes=n_c),
                "top_2_track": Accuracy(
                    task="multiclass",
                    num_classes=n_c,
                    top_k=5,
                ),
            }
        )

        if linear_track:
            self.cf_monitor = SingleLinear(self.n_classes, n_c)
        else:
            self.cf_monitor = ClassAttention(
                model_dim=self.n_classes,
                num_ca_layers=2,
                n_out=n_c,
                dense_config={"hddn_dim": 2 * self.n_classes, "act_h": "silu"},
                mha_config={"num_heads": 7, "do_layer_norm": True},
            )

        self.track_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        edges: T.Tensor,
        inp_nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> tuple:
        """Pass inputs through the full autoencoder.

        Returns:
            reconstructions, latent samples, latent means, latent_stds
        """
        # Pass through the normalisation layers
        nodes = self.node_norm(inp_nodes, mask)
        if self.high_dim:
            high = self.high_norm(high)
        return self.encoder(nodes, mask, ctxt=high)

    def output_head(self, output):
        """Take the output of forward and process to finish taks."""
        return output

    def get_loss(self, output, inp_nodes, label, mask, false_mask):
        # Calculate the loss
        lbl = label.view(-1)
        # Can't pass -1 labels
        lbl[lbl < 0] = 0
        loss = self.loss_fn(output.view(-1, self.n_classes), lbl)
        # Only use the loss at masked tokens
        loss = loss.view(mask.shape) * false_mask
        # Take the mean across masked tokens in each jet
        n_constituents = false_mask.sum(-1) + 1e-5
        # # Sometimes no nodes are masked
        # n_constituents[n_constituents == 0] = 1
        loss = loss.sum(-1) / n_constituents
        return loss

    def sample_mask(self, edges, nodes, mask):
        """Sample a mask for the real nodes in the sample."""
        # TODO should push this to the dataloader to speed up training
        # Mask self.max_mask of the real nodes
        with T.no_grad():
            # Sample random numbers and use order to dictate which is replaced
            indx = T.randn(mask.shape).to(edges)
            # Assign padded nodes a low number so they won't be selected
            indx[~mask] = -100
            # Use the largest random numbers as the masked tokens
            indx = T.topk(indx, int(self.max_mask * nodes.shape[1])).indices
            # Convert to index for flattened tensor
            odindx = indx + T.arange(
                0, nodes.shape[1] * nodes.shape[0], nodes.shape[1]
            ).view(-1, 1).to(indx)
            # Mask out indexes above self.max_mask percet
            count_mask = F.embedding(mask.sum(-1) - 1, self.count_mask)
            one_d_indx = odindx[count_mask[:, : indx.shape[1]]]
            false_mask = T.zeros(np.prod(list(mask.size())))
            false_mask[one_d_indx] = 1
            false_mask = false_mask.view(*mask.shape).to(nodes)
            # # REplace all tokens with mask, other options commented below
            false_matrix = false_mask.unsqueeze(-1).tile(self.node_dim)
            # Don't mask 80% of the nodes as in original bert
            # false_matrix = (
            #     (T.where(T.rand_like(false_mask) > 0.8, 0, false_mask))
            #     .unsqueeze(-1)
            #     .tile(self.node_dim)
            # )
            # # Don't change 10% of the nodes
            # decider = T.rand_like(false_mask)
            # false_matrix = (0.9 > T.where(decider > 0.8, 0, false_mask)).unsqueeze(-1).tile(self.node_dim)
            # # Randomly rescale 10% of the nodes
            # false_matrix = (T.where(decider > 0.9, T.rand_like(false_mask), false_mask)).unsqueeze(-1).tile(self.node_dim)
        return false_mask, false_matrix

    def preprocess_inputs(self, sample: tuple):
        return sample

    def get_metrics(self, sample: tuple, false_mask, output, label):
        false_indx = false_mask.to(T.bool)
        preds = output[false_indx]
        lbls = label[false_indx]
        metrics = {}
        for key, func in self.metrics.items():
            metrics[key] = func(preds, lbls)
        return metrics

    def monitor_quality(self, output, mask, og_labels):
        # prediction = self.cf_monitor(output, mask)
        prediction = self.cf_monitor(output, mask)
        loss = self.track_loss(prediction, og_labels.view(-1))
        metrics = {}
        for key, func in self.track_metrics.items():
            metrics[key] = func(prediction, og_labels.view(-1).long())
        return loss, metrics

    def _shared_step(self, sample: tuple) -> T.Tensor:
        sample = self.preprocess_inputs(sample)
        # Pass through the network
        edges, nodes, high, adjmat, mask, label = sample

        false_mask, false_matrix = self.sample_mask(edges, nodes, mask)

        # Set the masked nodes to the special token and make it differentiable
        masked_nodes = (1 - false_matrix) * nodes + false_matrix * self.masked_token
        if self.order_inputs:
            # If ordering the inputs, use the same strategy as BEiT
            masked_nodes = masked_nodes + self.pos_encoding
        output = self.forward(edges, masked_nodes, high, adjmat, mask)
        output = self.output_head(output)

        if self.pos_encode:
            output = self.positional_encoder(output + self.pos_encoding, mask)
        loss = self.get_loss(output, nodes, label, mask, false_mask)

        # Calculate some metrics
        metrics = self.get_metrics(sample, false_mask, output, label)
        l1, m2 = self.monitor_quality(output.detach(), mask, edges)
        metrics.update(m2)
        return loss.mean() + l1.mean(), metrics

    def log_wandb(self, loss: T.Tensor, metrics: dict, tag: str):
        self.log(f"{tag}/total_loss", loss)
        for key, value in metrics.items():
            self.log(f"{tag}/{key}", value)

    def preprocess_inputs(self, sample: tuple):
        # Always pass an empty tensor for the edges and high
        # Edges is no longer used, and this is an easy if hacky way to get at the label
        sample[0] = sample[-1]
        return sample

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        # Get the learning rate for the optimizer
        # Sometimes these two get out of sync for some reason
        if (
            self.optimizers().param_groups[0]["lr"]
            != self.lr_schedulers().down.optimizer.param_groups[0]["lr"]
        ):
            if batch_idx == 0:
                self.lr_schedulers().down.optimizer.param_groups = (
                    self.optimizers().param_groups
                )
            else:
                self.optimizers().param_groups = (
                    self.lr_schedulers().down.optimizer.param_groups
                )
        self.optimizers().param_groups[0]["lr"]
        loss, metrics = self._shared_step(sample)
        self.log_wandb(loss, metrics, "train")
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss, metrics = self._shared_step(sample)
        self.log_wandb(loss, metrics, "valid")
        return loss

    def predict_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        # Pass through the network
        edges, nodes, high, adjmat, mask, label = sample
        output = self.forward(edges, nodes, high, adjmat, mask)
        return {"output": output.mean(1), "label": label.view(-1, 1)}

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        if self.use_class_weights:
            bin_counts = T.zeros_like(self.class_weights)
            n_labels = 0
            max_batches = int(1e3)
            for i, sample in enumerate(self.trainer.datamodule.train_dataloader()):
                edges, nodes, high, adjmat, mask, label = sample
                lbl = label[mask].to(self.device)
                n_labels += len(lbl)
                bin_counts += T.bincount(lbl, minlength=self.n_classes)
                if i > max_batches:
                    break

            class_weight = n_labels / (self.n_classes * bin_counts)
            class_weight[~class_weight.isfinite()] = 1
            self.class_weights = class_weight
            self.loss_fn = nn.CrossEntropyLoss(
                reduction="none", weight=self.class_weights
            )

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            for key in list(self.metrics.keys()) + list(self.track_metrics.keys()):
                wandb.define_metric(f"train/{key}", summary="mean")
                wandb.define_metric(f"valid/{key}", summary="mean")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.scheduler.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
        }

    def on_validation_epoch_end(self) -> None:
        pass


class IterableBert(Bert):
    """A subclass is required to infer the number of labels that are present in
    the dataset, and to find the labels in the forward pass.

    Iterable datasets are too large to create a saved and labelled copy.
    The other option would be to load a separate file of labels. This
    probably makes more sense.
    """

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        encoder: partial,
        max_mask: float,
        normaliser_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        model_path: Path,
        use_class_weights: int = 0,
        output_head_config: Mapping = None,
        quantized_dim: int = None,
        **kwargs,
    ) -> None:
        if (quantized_dim is not None) and (quantized_dim != 0):
            inpt_dim[1] = quantized_dim
            self.quantize_inpt = True
        else:
            self.quantize_inpt = False
        super().__init__(
            inpt_dim=inpt_dim,
            n_nodes=n_nodes,
            n_classes=n_classes,
            encoder=encoder,
            max_mask=max_mask,
            normaliser_config=normaliser_config,
            optimizer=optimizer,
            scheduler=scheduler,
            use_class_weights=use_class_weights,
            output_head_config=output_head_config,
            **kwargs,
        )
        # Load the VQ VAE
        orig_cfg = reload_original_config(path_spec=model_path, get_best=False)
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        self.model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)
        # # Not all of the outputs of the loaded model will be used, so we need to remap them
        self.register_buffer(
            "remap_label", T.zeros(self.model.quantizer.num_codes, dtype=T.long)
        )
        self.unique_labels = T.zeros([427]).to(dtype=T.long)
        # self.register_buffer('unique_labels', nn.parameter.UninitializedBuffer(dtype=T.long))
        # Don't want to include super rare classes
        self.max_iter = 100
        self.lab_iter = 0

    def preprocess_inputs(self, sample: tuple):
        with T.no_grad():
            edges, nodes, high, adjmat, mask, label = sample
            # Always pass an empty tensor for the edges and high
            # Edges is no longer used, and this is an easy if hacky way to get at the label
            sample[0] = label
            et = T.empty(*edges.shape).to(edges)
            ht = T.empty(*high.shape).to(high)
            self.model.eval()
            self.model.node_norm.frozen = True
            encoder_pred = self.model.predict_step(
                (et, nodes, ht, adjmat, mask, label), 0
            )
            label = encoder_pred["code_labels"]
            if self.quantize_inpt:
                sample[1] = encoder_pred["quant_nodes"]
        sample[-1] = self.set_labels(label)
        return sample

    def set_labels(self, proposed_labels):
        in_shape = proposed_labels.shape
        proposed_labels = proposed_labels.view(-1).to(T.long)
        if self.lab_iter == 0:
            self.unique_labels = T.Tensor([]).to(self.unique_labels)
        if self.lab_iter < self.max_iter:
            self.lab_iter += 1
            if len(self.unique_labels) < self.n_classes:
                self.unique_labels = T.unique(
                    T.concatenate((self.unique_labels, proposed_labels))
                ).to(T.long)
                # Remap the labels
                self.remap_label[self.unique_labels] = T.arange(
                    len(self.unique_labels), dtype=T.long
                ).to(self.remap_label)
        return self.remap_label[proposed_labels].view(*in_shape)


class IterableBertFixedEncoding(Bert):
    """
    A subclass is required to infer the number of labels that are present in
    the dataset, and to find the labels in the forward pass.

    Iterable datasets are too large to create a saved and labelled copy.
    The other option would be to load a separate file of labels. This
    probably makes more sense.
    """

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        encoder: partial,
        labeller: partial,
        max_mask: float,
        normaliser_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        use_class_weights: int = 0,
        output_head_config: Mapping = None,
        quantized_dim: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            inpt_dim=inpt_dim,
            n_nodes=n_nodes,
            n_classes=n_classes,
            encoder=encoder,
            max_mask=max_mask,
            normaliser_config=normaliser_config,
            optimizer=optimizer,
            scheduler=scheduler,
            use_class_weights=use_class_weights,
            output_head_config=output_head_config,
            **kwargs,
        )
        self.quantize_inpt = quantized_dim is not None
        self.labeller = labeller(inpt_dim[1], num_labels=n_classes)

    def preprocess_inputs(self, sample: tuple):
        with T.no_grad():
            # TODO set the inputs to their centroid values as an option
            sample[0] = sample[-1]
            sample[-1], code = self.labeller(sample[1])
            if self.quantize_inpt:
                sample[1] = code
        return sample

    def on_train_start(self):
        n_store = []
        m_store = []
        for i, sample in enumerate(self.trainer.datamodule.train_dataloader()):
            edges, nodes, high, adjmat, mask, label = sample
            n_store += [nodes]
            m_store += [mask]
            if i > 20:
                break
        nodes = T.vstack(n_store)
        mask = T.vstack(m_store)
        self.labeller.fit(nodes, mask)


class RegressBert(Bert):
    """A class for implementing the Bert pretraining model."""

    def __init__(
        self,
        *args,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        model_path: Path = None,
        quantized_dim: int = None,
        **kwargs,
    ) -> None:
        n_classes = inpt_dim[1]
        if "linear_track" in kwargs:
            kwargs.pop("linear_track")
        super().__init__(
            inpt_dim=inpt_dim,
            n_nodes=n_nodes,
            n_classes=n_classes,
            linear_track=True,
            **kwargs,
        )
        self.loss_fn = nn.MSELoss(reduction="none")

    def get_loss(self, output, inp_nodes, label, mask, false_mask):
        # Calculate the loss
        nodes = self.node_norm(inp_nodes, mask)
        loss = self.loss_fn(output, nodes).sum(-1)
        # Only use the loss at masked tokens
        loss = loss * false_mask
        # Take the mean across masked tokens in each jet
        n_constituents = false_mask.sum(-1) + 1e-5
        # # Sometimes no nodes are masked
        # n_constituents[n_constituents == 0] = 1
        loss = loss.sum(-1) / n_constituents
        return loss

    def get_metrics(self, sample: tuple, false_mask, output, label):
        return {}


class FineTuner(LightningModule):
    """A class for fine tuning existing models."""

    def __init__(
        self,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        path_spec: Path,
        finaliser: partial,
        optimizer: partial,
        scheduler: Mapping,
        reinstantiate: bool = False,
        train_backbone: bool = True,
        get_best: bool = True,
        finaliser_out=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        n_classes = 1 if n_classes == 2 else n_classes
        self.multi_class = n_classes >= 2
        self.n_classes = n_classes

        # Load the model to be fine tuned
        orig_cfg = reload_original_config(path_spec=path_spec, get_best=get_best)
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        backbone = model_class.load_from_checkpoint(orig_cfg.ckpt_path, strict=False)
        if reinstantiate:
            backbone = hydra.utils.instantiate(
                orig_cfg.model,
                inpt_dim=(
                    backbone.edge_dim,
                    backbone.node_dim,
                    backbone.high_dim,
                ),
                n_nodes=backbone.count_mask.shape[0],
                n_classes=backbone.n_classes,
            )
        if hasattr(backbone, "trans_enc"):
            backbone.encoder = backbone.trans_enc
        if hasattr(backbone, "quantize_inpt"):
            self.quantize_inpt = backbone.quantize_inpt
        else:
            self.quantize_inpt = False
        self.backbone = backbone
        self.train_backbone = train_backbone
        # self.final = finaliser(
        #     inpt_dim=self.backbone.n_classes,
        #     outp_dim=self.n_classes
        # )
        self.is_full_t = isinstance(self.backbone.encoder, FullTransformerEncoder)
        if self.is_full_t:
            enc_dim = self.backbone.encoder.model_dim
        elif hasattr(backbone, "trans_enc"):
            enc_dim = backbone.trans_enc.model_dim
        else:
            enc_dim = self.backbone.n_classes
        final_out_size = finaliser_out or self.n_classes
        self.final = finaliser(
            inpt_dim=enc_dim,
            outp_dim=final_out_size,
            edge_dim=self.edge_dim,
            ctxt_dim=self.high_dim,
        )
        self.lin_out = (
            nn.Identity()
            if finaliser_out is None
            else nn.Linear(final_out_size, self.n_classes)
        )

        self.loss_fn = (
            nn.CrossEntropyLoss(reduction="none")
            if self.multi_class
            else nn.BCEWithLogitsLoss(reduction="none")
        )

        task = "multiclass" if n_classes >= 2 else "binary"
        self.metrics = nn.ModuleDict(
            {
                "accuracy": Accuracy(task=task, num_classes=self.n_classes),
                "au_roc": AUROC(num_classes=self.n_classes, task=task),
            }
        )

    def log_wandb(self, loss: T.Tensor, metrics: dict, tag: str):
        # TODO code duplication with Bert, build subclass
        self.log(f"{tag}/total_loss", loss)
        for key, value in metrics.items():
            self.log(f"{tag}/{key}", value)

    def backbone_forward(
        self,
        attn_bias: T.Tensor,
        inp_nodes: T.Tensor,
        ctxt: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> tuple:
        if self.quantize_inpt:
            sample = [attn_bias, inp_nodes, ctxt, adjmat, mask, mask]
            inp_nodes = self.backbone.preprocess_inputs(sample)[1]
        if self.is_full_t:
            # This is a hacky way to exclude the output embedding
            encoder = self.backbone.encoder
            x = self.backbone.node_norm(inp_nodes, mask)
            if encoder.ctxt_dim:
                ctxt = encoder.ctxt_emdb(ctxt)
            if encoder.edge_dim:
                attn_bias = encoder.edge_embd(attn_bias, ctxt)
            x = encoder.node_embd(x, ctxt=ctxt)
            output_backbone = encoder.te(x, mask=mask, ctxt=ctxt)
        else:
            output_backbone = self.backbone(attn_bias, inp_nodes, ctxt, adjmat, mask)
        return output_backbone

    def forward(
        self,
        edges: T.Tensor,
        inp_nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
    ) -> tuple:
        output_backbone = self.backbone_forward(edges, inp_nodes, high, adjmat, mask)
        # output = self.final(edges, output_backbone, high, adjmat, mask)
        output = self.final(output_backbone, mask, ctxt=high)
        return self.lin_out(output)

    def _shared_step(self, sample: tuple) -> T.Tensor:
        # Pass through the network
        edges, nodes, high, adjmat, mask, label = sample

        # Pass through the models
        output = self(edges, nodes, high, adjmat, mask)

        # Calculate the loss
        label = label.view(-1)
        if self.multi_class:
            loss = self.loss_fn(output, label)
        else:
            output = output.view(-1)
            loss = self.loss_fn(output, label.to(output))

        # Calculate some metrics
        metrics = {}
        for key, func in self.metrics.items():
            metrics[key] = func(output, label)
        return loss.mean(), metrics

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss, metrics = self._shared_step(sample)
        self.log_wandb(loss, metrics, "train")
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss, metrics = self._shared_step(sample)
        self.val_loss += [loss]
        return loss

    def on_validation_epoch_start(self):
        self.val_loss = []
        for key, func in self.metrics.items():
            func.reset()

    def on_validation_epoch_end(self):
        # For some metrics like the AU_ROC this is the only way it can actuallybe computed
        metrics = {}
        for key, func in self.metrics.items():
            metrics[key] = func.compute()
            func.reset()
        self.log_wandb(T.mean(T.Tensor(self.val_loss)), metrics, "valid")

    def predict_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        # Pass through the network
        edges, nodes, high, adjmat, mask, label = sample
        output = self.forward(edges, nodes, high, adjmat, mask)
        # return T.argmax(output, axis=-1)
        return {"output": output, "label": label.view(-1, 1)}

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            for key in self.metrics.keys():
                wandb.define_metric(f"train/{key}", summary="min")
                wandb.define_metric(f"valid/{key}", summary="min")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        # if self.train_backbone:
        #     opt = self.hparams.optimizer(params=self.parameters())
        # else:
        #     opt = self.hparams.optimizer(params=list(self.final.parameters()) + list(self.lin_out.parameters()))
        opt = self.hparams.optimizer(
            params=list(self.final.parameters()) + list(self.lin_out.parameters())
        )

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.scheduler.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
        }

    def on_predict_epoch_start(self):
        pass
