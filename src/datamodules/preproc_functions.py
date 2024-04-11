import numpy as np

from src.new_jutils import build_jet_edges, change_coordinates
from src.pc_transforms import jet_transforms


def preprocess_as_graphs(
    csts: np.ndarray,
    high: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    n_csts: int = 64,
    del_r_edges: float = 0.0,
    edge_coords: list | None = None,
    csts_coords: list | None = None,
    high_coords: list | None = None,
    transformation_list: list | None = None,
    transformation_prob: list | None = None,
) -> tuple:
    """Pre-processes jet constituents as graphs.

    Parameters
    ----------
    csts : np.ndarray
        The constituents of the jet.
    high : np.ndarray
        The kinematics and other high level features of the jet
    label : np.ndarray
        The label of the jet.
    mask : np.ndarray
        The mask for the constituents in the jet.
    n_csts : int, optional
        The number of constituents to keep, by default 64.
    del_r_edges:
        The delta_r radius for building edges between constituents
    edge_coords : list | None, optional
        The coordinates to use for the graph edges, by default None.
    csts_coords : list | None, optional
        The coordinates to use for the constituents, by default None.
    high_coords : list | None, optional
        The coordinates to use for the high level features, by default None.
    transformation_list : list | None, optional
        The list of transformations to apply to the jet constituents, by default None.
    transformation_prob : list | None, optional
        The probabilities of applying each transformation in `transformation_list`,
        by default None.

    Returns
    -------
    tuple
        A tuple containing the pre-processed:
            edges features,
            constituents,
            high level features,
            adjacency matrix, constituent mask,
            and label
    """

    # Trim the constituents
    csts = csts[:n_csts]
    mask = mask[:n_csts]

    # Apply transformations to the base pointcloud
    csts, high, mask = jet_transforms(
        csts, high, mask, transformation_list, transformation_prob
    )

    # Build jet edges (will return empty if del_r is set to 0)
    edges, adjmat = build_jet_edges(csts, mask, edge_coords, del_r_edges)

    # Convert to the specified selection of local variables and extract edges
    csts_coords = csts_coords or ["del_eta", "del_phi", "log_squash_pt"]
    csts, high = change_coordinates(csts, high, mask, csts_coords, high_coords)

    return edges, csts, high, adjmat, mask, label
