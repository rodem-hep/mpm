from collections import defaultdict
import pathlib
import h5py
import numpy as np
from typing import Union


def quantized_latents(
    path: pathlib.Path,
    debug: bool,
    hand_sliced: bool,
    use_quantised: bool,
    datasets: Union[dict, list],
):
    if len(datasets) == 1:
        return quantized_data(
            path=pathlib.Path(path) / f"{datasets[list(datasets.keys())[0]]}_test.h5",
            debug=debug,
            hand_sliced=hand_sliced,
            use_quantised=use_quantised,
        )
    else:
        datastore = defaultdict(list)
        for i, file in enumerate(datasets.values()):
            (
                high_data,
                nodes,
                mask,
                new_labels,
                rec_nodes,
                data_nodes,
                _,
            ) = quantized_data(
                path=pathlib.Path(path) / f"{file}_test.h5",
                debug=debug,
                hand_sliced=hand_sliced,
                use_quantised=use_quantised,
            )
            datastore["hd_all"] += [high_data]
            datastore["ndes"] += [nodes]
            datastore["masks"] += [mask]
            datastore["labels"] += [np.ones((mask.shape[0], 1), dtype=np.int64) * i]
            datastore["rec_node"] += [rec_nodes]
            datastore["data_node"] += [data_nodes]
        to_return = {}
        return_keys = [
            "hd_all",
            "ndes",
            "masks",
            "labels",
            "rec_node",
            "data_node",
            "unique_labels",
        ]
        for key, value in datastore.items():
            to_return[key] = np.concatenate(value)
        to_return["unique_labels"] = i + 1
        return [to_return[key] for key in return_keys]


def quantized_data(
    path: pathlib.Path, debug: bool, hand_sliced: bool, use_quantised: bool
):
    # Read the data, return the necessary shit
    n_take = int(1e4) if debug else -1
    with h5py.File(path, mode="r") as h:
        high_data = h["data_high"][:n_take]
        mask = h["mask"][:n_take]
        nodes = h["quant_nodes"][:n_take]
        labels = h["code_labels"][:n_take]
        rec_nodes = h["rec_nodes"][:n_take]
        data_nodes = h["data_nodes"][:n_take]
    # If the codebook wasn't at full occupancy we need to filter labels
    # Set all masked labels to an arbitrary label, only want unique real labels
    labels[~mask] = labels[0, 0]
    # Assume all labels are in the N to save time + memory
    unique_labels = np.unique(labels.reshape(-1)[: int(1e5)])
    # Lazy solution, TODO do something faster and memory efficient
    new_labels = np.zeros_like(labels)
    for i in range(len(unique_labels)):
        new_labels[labels == i] = i

    # Hacky hand based solution
    if hand_sliced:
        dt = data_nodes[mask]
        pt = dt[:, 2]
        n_quants = 8
        steps = np.linspace(1 / n_quants, 1, n_quants)
        pt_quants = np.quantile(pt, steps)
        radius_quants = []
        prev_bound = pt_quants[0] - 1e6
        for i, pt_bound in enumerate(pt_quants):
            to_hist = dt[(pt > prev_bound) & (pt < pt_bound)]
            eta, phi = to_hist[:, 0], to_hist[:, 1]
            radius = (eta**2 + phi**2) ** 0.5
            radius_quants += [np.quantile(radius, steps)]
            prev_bound = pt_bound
        radius = np.stack(radius_quants)

        angles = 2 * np.pi / n_quants
        angle_quants = np.linspace(angles, 2 * np.pi, n_quants) - np.pi

        # Bin all of the data
        def make_mask(data, previous, current):
            return (data >= previous) & (data < current)

        all_data = data_nodes.reshape(-1, 3)
        nodes = np.zeros_like(all_data)
        pt_data = all_data[:, -1]
        eta, phi, pt_data = all_data.transpose()
        radius_data = (eta**2 + phi**2) ** 0.5
        angle_data = np.arctan2(eta, phi)
        prev_pt = -1e6
        cnt = 0
        new_labels = np.zeros(all_data.shape[0])
        for i, pt_bound in enumerate(pt_quants):
            prev_radius = 0
            for rad_bound in radius[i]:
                prev_angle = -np.pi
                for angle_bound in angle_quants:
                    condition = (
                        make_mask(pt_data, prev_pt, pt_bound)
                        & make_mask(radius_data, prev_radius, rad_bound)
                        & make_mask(angle_data, prev_angle, angle_bound)
                    )
                    new_labels[condition] = cnt
                    nodes[condition] = all_data[condition].mean(0)
                    cnt += 1
                    prev_angle = angle_bound
                prev_radius = rad_bound
            prev_pt = pt_bound
        new_labels = new_labels.reshape(*labels.shape).astype(labels.dtype)
        nodes = nodes.reshape(*data_nodes.shape)
        unique_labels = np.arange(cnt)

    if not use_quantised:
        # TODO if you want to use this you will need the scaling
        nodes = data_nodes
        nodes[..., -1] /= 6

    # Set all training labels to -1
    new_labels[~mask] = -1
    return high_data, nodes, mask, new_labels, rec_nodes, data_nodes, len(unique_labels)
