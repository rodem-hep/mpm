"""Functions relating to the loading and manipulation of jet tensors."""

from typing import Tuple

import numpy as np

from mattstools.mattstools.numpy_utils import empty_0dim_like, log_clip, log_squash
from mattstools.mattstools.utils import signed_angle_diff

EPS = 1e-8  # Small value to prevent division of zero


def pxpypz_to_ptetaphi(
    px: np.ndarray, py: np.ndarray = None, pz: np.ndarray = None
) -> np.ndarray:
    """Convert from cartesian to ATLAS co-ordinates."""

    # If only one argument is given then do manual splitting
    if py is None and pz is None:
        py = px[..., 1]
        pz = px[..., 2]
        px = px[..., 0]

    pt = np.sqrt(px**2 + py**2)
    mtm = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(np.clip(pz / (mtm + EPS), -1 + EPS, 1 - EPS))
    phi = np.arctan2(py, px)

    return pt, eta, phi


def ptetaphi_to_pxpypz(
    pt: np.ndarray, eta: np.ndarray = None, phi: np.ndarray = None
) -> np.ndarray:
    """Convert from ATLAS to cartesian co-ordinates."""

    # If only one argument is given then do manual splitting
    if eta is None and phi is None:
        eta = pt[..., 1]
        phi = pt[..., 2]
        pt = pt[..., 0]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    return px, py, pz


def get_eng_from_ptetaphiM(particle) -> np.ndarray:
    """Given a 3 or 4 vector (with mass) in ptetaphi space, calculate the
    energy."""
    pt = particle[..., 0]
    eta = particle[..., 1]

    # Extra step if mass is present
    if particle.shape[-1] > 3:
        mass = particle[..., 3]
        return np.sqrt(mass**2 + (pt * np.cosh(eta)) ** 2)

    return pt * np.cosh(eta)


def cstpxpypz_to_jet(
    cst_px: np.ndarray, cst_py: np.ndarray, cst_pz: np.ndarray, cst_e: np.ndarray = None
) -> tuple:
    """Calculate high level jet variables using only the constituents.

    Args:
        cst_px: The constituent px
        cst_py: The constituent py
        cst_pz: The constituent pz
        cst_e: The constituent E to calculate total jet energy
               If none then cst are assumed to be massless and energy = momentum
    """

    # Calculate the total jet momenta
    jet_px = np.squeeze(cst_px).sum(axis=-1)
    jet_py = np.squeeze(cst_py).sum(axis=-1)
    jet_pz = np.squeeze(cst_pz).sum(axis=-1)

    # Calculate the total jet energy
    if cst_e is None:
        cst_e = np.sqrt(cst_px**2 + cst_py**2 + cst_pz**2)
    jet_e = np.squeeze(cst_e).sum(axis=-1)

    # Calculate the total jet mass
    jet_m = np.sqrt(np.maximum(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0))

    return jet_px, jet_py, jet_pz, jet_m, jet_e


def cstptetaphi_to_jet(
    csts: np.ndarray, mask: np.ndarray, jets: np.ndarray
) -> np.ndarray:
    """Inplace! Overwrite the jet info (pt, eta, phi, M) with values rederived
    from the point cloud.

    Does not touch the additional entries of the jet (like subjettiness)
    """
    jets = jets.copy()

    # Calculate the constituent pt, eta and phi
    pt = csts[..., 0]
    eta = csts[..., 1]
    phi = csts[..., 2]

    # Calculate the total jet values (always include the mask when summing!)
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # jet pt
    jets[..., 0] = np.sqrt(jet_px**2 + jet_py**2)

    # jet eta
    jets[..., 1] = 0.5 * log_clip((jet_e + jet_pz) / (jet_e - jet_pz))

    # jet_phi
    jets[..., 2] = np.arctan2(jet_py, jet_px)

    # jet mass
    jets[..., 3] = np.sqrt(
        np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, EPS, None)
    )

    return jets


def locals_to_jet_pt_mass(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculate the overall jet kinematics using only the local info:

    - del_eta
    - del_phi
    - log_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = nodes[..., 0]
    phi = nodes[..., 1]
    pt = np.exp(nodes[..., 2])

    # Calculate the total jet values (always include the mask when summing!)
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_pt**2 - jet_pz**2, EPS, None))

    return np.vstack([jet_pt, jet_m]).T


def cstpxpypz_to_jet_mass(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculate the overall jet kinematics using the constituent cartesian
    info."""

    # Break up the constituent coordinates
    px = nodes[..., 0]
    py = nodes[..., 1]
    pz = nodes[..., 2]
    e = np.sqrt(np.clip(px**2 + py**2 + pz**2, EPS, None))

    # Calculate the total jet values (always include the mask when summing!)
    jet_px = (px * mask).sum(axis=-1)
    jet_py = (py * mask).sum(axis=-1)
    jet_pz = (pz * mask).sum(axis=-1)
    jet_e = (e * mask).sum(axis=-1)

    # Calculate the jet mass from its 4 momenta
    jet_p2 = jet_px**2 + jet_py**2 + jet_pz**2
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_p2, EPS, None))

    return jet_m


def cstetaphipt_to_jet_mass(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculate the overall jet kinematics using the constituent cartesian
    info.

    - eta, phi, pt
    """

    # Break up the coordinates
    eta = nodes[..., 0].copy()
    phi = nodes[..., 1].copy()
    pt = nodes[..., 2].copy()

    # Calculate the constituent px, py, pz
    pxpypz = np.zeros_like(nodes)
    pxpypz[..., 0] = pt * np.cos(phi)
    pxpypz[..., 1] = pt * np.sin(phi)
    pxpypz[..., 2] = pt * np.sinh(eta)

    return cstpxpypz_to_jet_mass(pxpypz, mask)


def inv_mass_all_pairs(cnsts: np.ndarray) -> np.ndarray:
    """Return the invariant mass matrix of all pairs of constuents Expects the
    constituents to be described by pt,eta,phi."""

    # We need the cartesian representations for the constituent momenta
    cst_px, cst_py, cst_pz = ptetaphi_to_pxpypz(cnsts)
    cst_e = cnsts[..., 0] * np.cosh(cnsts[..., 1])  # From pt and eta

    # Get the expanded sum of all pairs
    cst_px = np.expand_dims(cst_px, -1) + np.expand_dims(cst_px, -2)
    cst_py = np.expand_dims(cst_py, -1) + np.expand_dims(cst_py, -2)
    cst_pz = np.expand_dims(cst_pz, -1) + np.expand_dims(cst_pz, -2)
    cst_e = np.expand_dims(cst_e, -1) + np.expand_dims(cst_e, -2)

    # Return the mass from p_mu^2
    return np.sqrt(
        np.clip(cst_e**2 - cst_px**2 - cst_py**2 - cst_pz**2, EPS, None)
    )


def dot_prod_all_pairs(cnsts: np.ndarray) -> np.ndarray:
    """Return the dot product of all pairs of constituent four momenta."""

    # We need the cartesian representations for the constituent momenta
    cst_px, cst_py, cst_pz = ptetaphi_to_pxpypz(cnsts)
    cst_e = cnsts[..., 0] * np.cosh(cnsts[..., 1])  # From pt and eta

    # Calculate the minkowsi product
    px2 = np.expand_dims(cst_px, -1) * np.expand_dims(cst_px, -2)
    py2 = np.expand_dims(cst_py, -1) * np.expand_dims(cst_py, -2)
    pz2 = np.expand_dims(cst_pz, -1) * np.expand_dims(cst_pz, -2)
    e2 = np.expand_dims(cst_e, -1) * np.expand_dims(cst_e, -2)

    # Return the mass from p_mu^2
    return e2 - px2 - py2 - pz2


def build_jet_edges(
    nodes: np.ndarray, mask: np.ndarray, coords: list, delR_threshold: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the jet edges based on the current node features.

    args:
        nodes: Low level constituent information, pt, eta, phi
        high: High level jet information, pt, eta, phi, M, other
        mask: A boolean tensor showing which cnsts are real vs padded
        coords: A list of strings containing the requested edge features
    kwargs:
        delR_threshold: The limit to connect two nodes, if 0 no edges are made at all
    """
    n_nodes = len(nodes)

    # If we want fully connects with no edge features then return so
    if delR_threshold > 100 and not (coords):
        edges = np.empty((n_nodes * n_nodes, 0), dtype="f")
        adjmat = np.ones((n_nodes, n_nodes), dtype="bool")
        return edges, adjmat

    # If the del r value is zero then the edges will be empty
    if delR_threshold == 0:
        edges = np.empty((0, 0), dtype="f")
        adjmat = np.zeros((n_nodes, n_nodes), dtype="bool")
        return edges, adjmat

    # Create a padding matrix of possible connections between real nodes
    pad_mask = np.expand_dims(mask, -1) * np.expand_dims(mask, -2)

    # Create get the spacial seperations between nodes
    pt = nodes[..., 0:1]
    eta = nodes[..., 1:2]
    phi = nodes[..., 2:3]
    del_eta = np.expand_dims(eta, -2) - np.expand_dims(eta, -3)
    del_phi = signed_angle_diff(np.expand_dims(phi, -2), np.expand_dims(phi, -3))

    # Create the delR matrix between each node
    delR_matrix = np.sqrt(del_eta**2 + del_phi**2)

    # Build the adjacency matrix where the distances fall below the threshold
    adjmat = np.squeeze(delR_matrix < delR_threshold)
    adjmat *= ~np.identity(len(adjmat), dtype="bool")  # Remove self connections
    adjmat[~pad_mask] = 0  # Make sure the padded elements are false

    # Start building the dictionary of lambda functions
    edges = {
        "del_eta": lambda: del_eta,
        "del_phi": lambda: del_phi,
        "del_R": lambda: delR_matrix,
        "pt_send": lambda: np.expand_dims(pt * adjmat, -1),
        "pt_sum": lambda: np.expand_dims(pt, -2) + np.expand_dims(pt, -3),
        "psi": lambda: np.arctan2(del_eta, del_phi),
        "log_del_R": lambda: log_clip(delR_matrix),
        "m": lambda: inv_mass_all_pairs(nodes)[..., None],
        "dot_prod": lambda: dot_prod_all_pairs(nodes)[..., None],
    }

    # Extra variables derived from lambda functions
    edges["z"] = lambda: edges["pt_send"]() / (edges["pt_sum"]() + EPS)
    edges["kt"] = lambda: edges["pt_send"]() * edges["del_R"]()
    edges["log_kt"] = lambda: log_clip(edges["kt"]())
    edges["log_m"] = lambda: log_clip(edges["m"]())

    # Select the appropriate coordinates and return the edges
    if len(coords) > 0:
        edges = np.concatenate([edges[key]() for key in coords], axis=-1)
    else:
        edges = np.empty((0, 0), dtype="f")
    return edges, adjmat


def graph_coordinates(
    raw_nodes: np.ndarray,
    raw_high: np.ndarray,
    mask: np.ndarray,
    coordinates: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts the standard ATLAS 4 vector to a specific set of coordinates
    for use in a graph network.

     - The vectors are of the format pt, eta, phi, (M, other for jets)
     - Only the kinematics and mass variables are transformed/dropped
     - The "other" variables will always be in the final array

    args:
        raw_nodes: Low level constituent information, pt, eta, phi
        raw_high: High level jet information, pt, eta, phi, M, other
        mask: A boolean tensor showing which cnsts are real vs padded
        coordinates: A dict with keys for edge, node, high coordinates
    """

    # Extract the individual measurements from the underlying tensors
    # Splitting in this way does not result in any memory copy
    jet_pt = raw_high[..., 0:1]
    jet_eta = raw_high[..., 1:2]
    jet_phi = raw_high[..., 2:3]
    jet_mass = raw_high[..., 3:4]
    jet_oth = raw_high[..., 4:] if raw_high.shape[-1] > 4 else empty_0dim_like(jet_mass)
    cst_pt = raw_nodes[..., 0:1]
    cst_eta = raw_nodes[..., 1:2]
    cst_phi = raw_nodes[..., 2:3]
    cst_oth = (
        raw_nodes[..., 3:] if raw_nodes.shape[-1] > 3 else empty_0dim_like(cst_phi)
    )

    # Put all the jet tensors into a single dictionary
    high = {
        "pt": lambda: jet_pt,
        "eta": lambda: jet_eta,
        "phi": lambda: jet_phi,
        "mass": lambda: jet_mass,
        "eng": lambda: np.sqrt(jet_mass**2 + (jet_pt * np.cosh(jet_eta)) ** 2),
        "px": lambda: jet_pt * np.cos(jet_phi),
        "py": lambda: jet_pt * np.sin(jet_phi),
        "pz": lambda: jet_pt * np.sinh(jet_eta),
        "log_pt": lambda: log_clip(jet_pt),
        "log_mass": lambda: log_clip(jet_pt),
        "mopt": lambda: jet_mass / (jet_pt + EPS),
        "log_squash_pt": lambda: log_squash(jet_pt),
    }

    # Put all of the constituent tensors into a single dictionary
    nodes = {
        "pt": lambda: cst_pt,
        "eta": lambda: cst_eta,
        "phi": lambda: cst_phi,
        "eng": lambda: cst_pt * np.cosh(cst_eta),
        "px": lambda: cst_pt * np.cos(cst_phi),
        "py": lambda: cst_pt * np.sin(cst_phi),
        "pz": lambda: cst_pt * np.sinh(cst_eta),
        "del_eta": lambda: cst_eta - np.expand_dims(jet_eta, -1),
        "del_phi": lambda: signed_angle_diff(cst_phi, np.expand_dims(jet_phi, -1)),
        "log_pt": lambda: log_clip(cst_pt),
        "pt_frac": lambda: cst_pt / (np.expand_dims(jet_pt, -1) + EPS),
        "log_squash_pt": lambda: log_squash(cst_pt),
    }

    # Other derived variables require the dicts to already be created
    high["log_eng"] = lambda: log_clip(high["eng"]())
    nodes["log_eng"] = lambda: log_clip(nodes["eng"]())
    nodes["log_eng_frac"] = lambda: nodes["log_eng"]() - np.expand_dims(
        high["log_eng"](), -1
    )
    nodes["log_pt_frac"] = lambda: nodes["log_pt"]() - np.expand_dims(
        high["log_pt"](), -1
    )
    nodes["pt_frac_pc"] = lambda: nodes["pt"]() / (np.sum(nodes["pt"]()) + EPS)
    nodes["log_pt_frac_pc"] = lambda: log_clip(nodes["pt_frac_pc"]())
    nodes["del_R"] = lambda: np.sqrt(nodes["del_eta"]() ** 2 + nodes["del_phi"]() ** 2)

    # Combine the list of chosen arguments with the other information if present
    nodes = np.concatenate(
        [nodes[key]() for key in coordinates["node"]] + [cst_oth], axis=-1
    )
    if raw_high.size == 0:
        high = raw_high
    else:
        high = np.concatenate(
            [high[key]() for key in coordinates["high"]] + [jet_oth], axis=-1
        )

    # Ensure nodes still adheres to mask!
    # This is because the deltas and ratios will no longer be 0!
    nodes[~mask] = 0.0

    return nodes, high


def get_boost_from_spherical(
    pt: np.ndarray, eta: np.ndarray = None, phi: np.ndarray = None, mass: np.ndarray = 0
) -> np.ndarray:
    """Return the appropriate boost vector in cartesian coordinates given a 4
    momentum defined by pt, eta, phi, and optionally mass."""

    # If one arrgument assume it contains all
    if eta is None and phi is None:
        mass = mass if pt.shape[-1] < 4 else pt[..., -1]
        eta = pt[..., 1]
        phi = pt[..., 2]
        pt = pt[..., 0]
    denom = np.sqrt(mass**2 + (pt * np.cosh(eta)) ** 2)
    boost = np.vstack(ptetaphi_to_pxpypz(pt, eta, phi)).T / (denom + EPS)
    return np.squeeze(boost)


def apply_boost(vector: np.ndarray, boost: np.ndarray) -> np.ndarray:
    """Boosts a 4 vector into the frame given by a 3 vector.

    - Opposite of ROOT::TLorentzBoost!
    - Returns a relativistic vector-boost
    - The vector can be batched, only one boost vector though!
    args:
        vector: The vector being boosted as E/c, x, y, z
        boost: The boost vector as x, y, z
    """
    # Ensure boost is flat
    boost = np.squeeze(boost)

    # Calculate the beta/gamma coeff (some jets have 1 const cant boost that much!)
    beta2 = np.clip(np.sum(boost**2), EPS, 1 - EPS)
    gamma = 1 / np.sqrt(1 - beta2)

    # Create the boost matrix
    lmb = np.zeros([4, 4])
    lmb[1:, 1:] = np.expand_dims(boost, -2) * np.expand_dims(boost, -1) / beta2
    lmb[1:, 1:] *= gamma - 1
    lmb[1:, 1:] += np.diag([1, 1, 1])
    lmb[0, 1:] = -boost * gamma
    lmb[1:, 0] = -boost * gamma
    lmb[0, 0] = gamma

    # Boost the vector using batched matrix multiplication
    return np.einsum("ij,bj->bi", lmb, vector)


def boost_jet_mopt(cnsts: np.ndarray, jets: np.ndarray, mopt: float = 1.0) -> tuple:
    """Boost a jet and its constituents along the jet axis until the m/pt of
    the jet reaches a pre-defined value.

    This is done by boosting the entire jet first into its reference frame,
    then boosting again to a specific mass over pt value

    args:
        cnsts: The kinematics of the jet constituents (pt, eta, phi)
        jets: The kinematics of the jet (pt, eta, phi, mass)
        mask: Boolean array showing padded level of the constituents
    kwargs:
        mopt: The desired mass over pt for the output jet
        random: Randomly boost the jet using the mopt as the upper bound
    """

    # We need the cartesian representations the constituent momenta
    cst_px, cst_py, cst_pz = ptetaphi_to_pxpypz(cnsts)
    cst_e = cnsts[..., 0] * np.cosh(cnsts[..., 1])  # From pt and eta
    cnsts = np.stack([cst_e, cst_px, cst_py, cst_pz], axis=1)

    # The first boosting vector comes from the jet's velocity (divide p by e)!
    first_boost = get_boost_from_spherical(jets)
    cnsts = apply_boost(cnsts, first_boost)

    # If we want that reference frame then we leave it
    if mopt == -1:
        cnsts = np.stack(pxpypz_to_ptetaphi(cnsts[..., 1:]), axis=1).astype("f")
        jets[..., 0] = 0
        return cnsts, jets

    # The second boosting vector is derived from a specific mopt value (negative)
    second_boost = -get_boost_from_spherical(
        jets[..., 3] / mopt, jets[..., 1], jets[..., 2], jets[..., 3]
    )
    cnsts = apply_boost(cnsts, second_boost)

    # Get back to pt eta phi
    cnsts = np.stack(pxpypz_to_ptetaphi(cnsts[..., 1:]), axis=1).astype("f")
    jets[..., 0] = jets[..., -1] / mopt
    return cnsts, jets
