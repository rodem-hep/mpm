from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F
import wandb
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans


class Labeller(LightningModule):
    def __init__(self, inpt_dim, num_labels: int = 512):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.inpt_dim = inpt_dim
        self.num_labels = num_labels

    def fit(nodes, mask):
        raise NotImplementedError


class KnnLabeller(Labeller):
    """Fit an unconditional knn to the input representation."""

    def __init__(self, inpt_dim: int, num_labels: int = 512):
        super().__init__(inpt_dim, num_labels)
        self.kmeans = KMeans(
            n_clusters=num_labels,
            random_state=0,
            init="k-means++",
            n_init="auto",
        )
        self.register_buffer("codes", T.zeros(num_labels, inpt_dim))
        self.register_buffer("initialised", T.zeros(1))

    @T.no_grad()
    def fit(self, nodes, mask):
        data = nodes[mask].detach().cpu().numpy()
        self.kmeans.fit(data)
        self.codes = T.Tensor(self.kmeans.cluster_centers_).to(self.device)

    @T.no_grad()
    def forward(self, nodes):
        node_shape = nodes.shape
        dist = T.cdist(nodes.view(-1, node_shape[-1]), self.codes)
        indx = T.argmin(dist, dim=-1)
        label = indx.view(node_shape[:2])
        return label, self.codes[indx].view(node_shape)
