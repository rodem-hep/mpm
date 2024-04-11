import torch as T

from mattstools.mattstools.numpy_utils import undo_log_squash


def torch_locals_to_jet_pt_mass(nodes: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Calculate the overall jet kinematics using only the local info:

    - del_eta
    - del_phi
    - log_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = nodes[..., 0]
    phi = nodes[..., 1]
    x = nodes[..., 2]
    pt = T.sign(x) * (T.exp(T.abs(x)) - 1)

    # Calculate the total jet values (always include the mask when summing!)
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.sqrt(jet_px**2 + jet_py**2)
    jet_m = T.sqrt(T.clip(jet_e**2 - jet_pt**2 - jet_pz**2, 1e-8, None))

    return T.vstack([jet_pt, jet_m]).T
