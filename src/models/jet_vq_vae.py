from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import torch as T
from pytorch_lightning import LightningModule
from src.jet_utils import locals_to_jet_pt_mass

import wandb
from mattstools.mattstools.modules import IterativeNormLayer
from mattstools.mattstools.plotting import plot_multi_hists_2
from mattstools.mattstools.torch_utils import get_sched, to_np

from vqtorch.nn import VectorQuant


class VqVae(LightningModule):
    """
    An transformer based vector quantised autoencoder for point cloud data using
    transformers.
    TODO should write a base class for autoencoders to reduce code duplication...
    """

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_classes: int,
        n_nodes: int,
        lat_dim: int,
        alpha: int,
        loss_fn: partial,
        normaliser_config: Mapping,
        encoder: partial,
        decoder: partial,
        vq_kwargs: Mapping,
        optimizer: partial,
        scheduler: Mapping,
    ) -> None:
        """ """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.alpha = alpha
        self.loss_fn = loss_fn
        self.lat_dim = lat_dim
        self.val_step_outs = []

        # The layers which normalise the input data
        self.node_norm = IterativeNormLayer(self.node_dim, **normaliser_config)
        if self.high_dim:
            self.high_norm = IterativeNormLayer(self.high_dim, **normaliser_config)

        # The transformer encoding model
        self.encoder = encoder(
            inpt_dim=self.node_dim,
            outp_dim=self.lat_dim,
            ctxt_dim=self.high_dim,
        )

        # Initialise the transformer generator model
        self.generator = decoder(
            inpt_dim=self.lat_dim,
            outp_dim=self.node_dim,
        )

        # make the VQ-VAE layer
        self.quantizer = VectorQuant(**vq_kwargs)

    def forward(
        self,
        nodes: T.Tensor,
        high: T.Tensor,
        mask: T.BoolTensor,
    ) -> tuple:
        """Pass inputs through the full autoencoder.

        Returns:
            reconstructions, latent samples, latent means, latent_stds
        """

        # Pass the inputs through the normalisation layers
        # TODO might need to skip
        # self.node_norm.frozen.item()
        nodes = self.node_norm(nodes, mask)
        if self.high_dim:
            high = self.high_norm(high)

        # Pass through the encoder
        latents = self.encoder(nodes, mask, ctxt=high)

        # For debugging skip this step
        # This quantizer expects image inputs, so give it a channel dim to match this
        z_q_flat, vq_dict = self.quantizer(latents[mask].unsqueeze(-1))
        # Replace the quantized latents with the original latents
        z_q = T.zeros_like(latents)
        # TODO quantizer is returning float32, should be float16
        # This breaks the backward step...
        z_q[mask] = z_q_flat.squeeze(-1).to(latents)
        # z_q = latents
        # vq_dict = {"loss": 0.}

        # Pass through the generator
        rec_nodes = self.generator(z_q, mask, ctxt=high)

        # # Should probably actually calculate the loss in the scaled
        # rec_nodes = self.node_norm.reverse(rec_nodes, mask)

        return rec_nodes, z_q, vq_dict

    def _shared_step(self, sample: tuple, loss_reduction: str = "mean") -> T.Tensor:
        # Unpack the tuple
        # csts, csts_label, high, label, mask, null_mask = sample
        edges, csts, high, adjmat, mask, label = sample

        # Get the reconstructions from the vae
        rec_nodes, _latents, vq_dict = self.forward(csts, high, mask)

        # Distribution matching loss for reconstructed nodes
        rec_loss = self.loss_fn(self.node_norm(csts, mask), rec_nodes)
        rec_loss[~mask] = 0.0
        rec_loss = rec_loss.sum((1, 2)) / mask.sum(1)
        rec_loss = rec_loss.mean()

        # Latent space loss for encodings
        lat_loss_flat = vq_dict["loss"]
        # Reshape the loss to match the mask
        lat_loss = T.zeros_like(csts[..., 0])
        lat_loss[mask] = lat_loss_flat
        # Now take a weighted sum of the loss, accounting for the variable length of the jets
        lat_loss = (lat_loss.sum(1) / mask.sum(1)).mean()

        # Combine the losses with their respective terms
        total_loss = rec_loss + lat_loss * self._get_alpha_weight()
        return rec_nodes, rec_loss, lat_loss, total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        if self.node_norm.frozen:
            rec_nodes, rec_loss, lat_loss, total_loss = self._shared_step(sample)
            self.log("train/rec_loss", rec_loss)
            self.log("train/lat_loss", lat_loss)
            self.log("train/total_loss", total_loss)
            self.log("alpha_weight", self._get_alpha_weight())
        else:
            # Don't pass through the networks unless this has been fit
            edges, csts, high, adjmat, mask, label = sample
            nodes = self.node_norm(csts, mask)
            if self.high_dim:
                high = self.high_norm(high)
            total_loss = None
        return total_loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        rec_nodes, rec_loss, lat_loss, total_loss = self._shared_step(sample)
        self.log("valid/rec_loss", rec_loss)
        self.log("valid/lat_loss", lat_loss)
        self.log("valid/total_loss", total_loss)

        # Save some data for plotting
        if batch_idx < 100:
            _, nodes, _, _, mask, _ = sample
            # Add the variables required for plotting
            self.val_step_outs.append(
                to_np((nodes, self.node_norm.reverse(rec_nodes, mask), mask))
            )

        return total_loss

    def predict_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        # Pass through the network
        edges, csts, high, adjmat, mask, label = sample
        rec_nodes, latents, vq_loss = self.forward(csts, high, mask)
        # TODO this should be added internally to the vq_layer
        dist = T.cdist(latents.view(-1, self.lat_dim), self.quantizer.codebook.weight)
        code_label = T.argmin(dist, dim=-1).view(csts.shape[:2])
        return {
            "data_edges": edges,
            "data_nodes": csts,
            "data_high": high,
            "mask": mask,
            "data_label": label.view(-1, 1),
            "quant_nodes": latents,
            "code_labels": code_label,
            "rec_nodes": rec_nodes,
        }

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""
        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("train/rec_loss", summary="min")
            wandb.define_metric("train/lat_loss", summary="min")
            wandb.define_metric("train/alpha_weight", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            wandb.define_metric("valid/rec_loss", summary="min")
            wandb.define_metric("valid/lat_loss", summary="min")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mltools to initialise the scheduler (cyclic-epoch sync)
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

    def _get_alpha_weight(self):
        # TODO maybe should schedule this?
        return self.alpha

    def on_validation_epoch_end(self) -> None:
        """Makes several plots of the jets and how they are reconstructed.

        Assumes that the nodes are of the format: del_eta, del_phi,
        log_pt
        """
        # Supress all "invalid value encountered in divide" Runtime errors in the plotting
        with np.errstate(divide="ignore", invalid="ignore"):
            # Unpack the list
            nodes = np.vstack([v[0] for v in self.val_step_outs])
            rec_nodes = np.vstack([v[1] for v in self.val_step_outs])
            # Invert the preprocessing on the reconstructed nodes
            mask = np.vstack([v[2] for v in self.val_step_outs])

            # Create the plotting dir
            plot_dir = Path("./plots/")
            plot_dir.mkdir(parents=False, exist_ok=True)

            # Save the above to disk
            np.savez(
                f"./plots/val_step_{self.trainer.current_epoch}.npz",
                nodes=nodes,
                rec_nodes=rec_nodes,
                mask=mask,
            )

            # Clear the outputs
            self.val_step_outs.clear()
            n_add_subs = nodes.shape[-1] - 3

            # Plot histograms for the constituent marginals
            cst_img = plot_multi_hists_2(
                data_list=[nodes[mask], rec_nodes[mask]],
                data_labels=["Original", "Reconstructed"],
                col_labels=["del_eta", "del_phi", "log_pt"]
                + [f"add_sub_{i}" for i in range(n_add_subs)],
                return_img=True,
                do_ratio_to_first=True,
                path=f"./plots/csts_{self.trainer.current_epoch}",
                # bins=[
                #     np.linspace(-1, 1, 50),
                #     np.linspace(-1, 1, 50),
                #     np.linspace(-3, 6, 50),
                # ],
                bins=[50] * (3 + n_add_subs),
                logy=True,
            )

            # Convert to total jet mass and pt
            jets = locals_to_jet_pt_mass(nodes, mask)
            rec_jets = locals_to_jet_pt_mass(rec_nodes, mask)

            # Image for the total jet variables
            jet_img = plot_multi_hists_2(
                data_list=[jets, rec_jets],
                data_labels=["Original", "Reconstructed"],
                col_labels=["pt", "mass"],
                bins="quant",
                do_ratio_to_first=True,
                return_img=True,
                path=f"./plots/jets_{self.trainer.current_epoch}",
            )

            # Create the wandb table and add the data
            if wandb.run is not None:
                table = wandb.Table(columns=["csts", "jets"])
                table.add_data(wandb.Image(cst_img), wandb.Image(jet_img))
                wandb.run.log({"table": table}, commit=False)
