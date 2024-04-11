"""Functions for calculating distances between elements of tensors."""

import torch as T
import torch.nn.functional as F

EPS = 1e-8  # epsilon for preventing division by zero


def masked_diff_matrix(
    tensor_a: T.Tensor,
    mask_a: T.BoolTensor,
    tensor_b: T.Tensor = None,
    mask_b: T.BoolTensor = None,
    pad_val: float = float("Inf"),
    allow_self: bool = False,
    track_grad: bool = False,
) -> T.Tensor:
    """Builds a difference matrix between two masked/padded tensors.

    - DOES NOT WORK WITH BATCH DIMENSION FOR NOW!

    - Will create the self distance matrix if only one tensor and mask is provided
    - The distance matrix will be padded
    - Uses the tensors as a->senders (dim 1) vs b->receivers (dim 2)
        - Symettrical if doing self distance (tensor_b is none)
    - Attributes of the matrix will be the senders-receivers

    args:
        tensor_a: The first tensor to use (n_nodes x n_features)
        mask_a: Shows which nodes in tensor_a are real (n_nodes)
    kwargs:
        tensor_b: The second tensor to use (N_nodes x n_features)
        mask_b: Shows which nodes in tensor_b are real (n_nodes)
        pad_val: The value to use for distances between fake and real (fake) nodes
        allow_self: Only applicable for self distances. Allows self connections.
        track_grad: If the gradients are tracked during this step (memory heavy!)
    returns:
        diff_matrix matrix: distance between tensor_a and tensor_a(b)
        matrix_mask: location of connections between real nodes
    """

    # Save current gradient settings then change to argument
    has_grad = T.is_grad_enabled()
    T.set_grad_enabled(track_grad)

    # Check if this is a self distance matrix
    is_self_dist = tensor_b is None
    if is_self_dist:
        tensor_b = tensor_a
        mask_b = mask_a

    # Calculate the matrix mask of real nodes to real nodes
    matrix_mask = mask_a.unsqueeze(-1) * mask_b.unsqueeze(-2)

    # Remove diagonal (loops) from the mask for self connections
    if not allow_self and is_self_dist:
        matrix_mask *= ~T.eye(len(mask_a)).bool()

    # Calculate the distance matrix as normal
    diff_matrix = tensor_a.unsqueeze(-2) - tensor_b.unsqueeze(-3)

    # Ensure the distances between fake nodes take the padding value
    diff_matrix[~matrix_mask] = pad_val

    # Revert the gradient tracking to the previous setting
    T.set_grad_enabled(has_grad)

    return diff_matrix, matrix_mask.detach()


def masked_dist_matrix(
    tensor_a: T.Tensor,
    mask_a: T.BoolTensor,
    tensor_b: T.Tensor = None,
    mask_b: T.BoolTensor = None,
    pad_val: float = float("Inf"),
    measure: str = "eucl",
    allow_self: bool = False,
    track_grad: bool = False,
) -> T.Tensor:
    """Builds a distance matrix between two masked/padded tensors.

    - Will create the self distance matrix if only one tensor and mask is provided
    - The distance matrix will be padded
    - Uses the tensors as a->senders (dim 1) vs b->receivers (dim 2)
        - Symettrical if doing self distance (tensor_b is none)

    args:
        tensor_a: The first tensor to use (batch x n_nodes x n_features)
        mask_a: Shows which nodes in tensor_a are real (batch x n_nodes)
    kwargs:
        tensor_b: The second tensor to use (batch x n_nodes x n_features)
        mask_b: Shows which nodes in tensor_b are real (batch x n_nodes)
        pad_val: The value to use for distances between fake and real (fake) nodes
        measure: If the euclidean or the dot procuct is used to define the nodes
        allow_self: Only applicable for self distances. Allows self connections.
        track_grad: If the gradients are tracked during this step (memory heavy!)
    returns:
        distance matrix: between tensor_a and tensor_a(b)
        matrix_mask: location of connections between real nodes
    """

    # Save current gradient settings then change to argument
    has_grad = T.is_grad_enabled()
    T.set_grad_enabled(track_grad)

    # Check if this is a self distance matrix
    is_self_dist = tensor_b is None
    if is_self_dist:
        tensor_b = tensor_a
        mask_b = mask_a

    # Calculate the matrix mask of real nodes to real nodes
    matrix_mask = mask_a.unsqueeze(-1) * mask_b.unsqueeze(-2)

    # Remove diagonal (loops) from the mask for self connections
    if not allow_self and is_self_dist:
        matrix_mask *= ~T.diag_embed(T.full_like(mask_a, True))

    # Calculate the distance matrix as normal
    if measure == "eucl":
        dist_matrix = T.cdist(tensor_a, tensor_b)
    if measure == "dot":
        a_info = matrix_mask.unsqueeze(-1) * tensor_a.unsqueeze(-2)
        b_info = matrix_mask.unsqueeze(-1) * tensor_b.unsqueeze(-3)
        dist_matrix = F.cosine_similarity(a_info, b_info, -1, EPS)

    # Ensure the distances between fake nodes take the padding value
    if track_grad:
        T.masked_fill(dist_matrix, ~matrix_mask, pad_val)
    else:
        dist_matrix[~matrix_mask] = pad_val

    # Revert the gradient tracking to the previous setting
    T.set_grad_enabled(has_grad)

    return dist_matrix, matrix_mask.detach()


def masked_fc_adjmat(
    mask_a: T.BoolTensor, mask_b: T.BoolTensor = None, allow_self: bool = False
) -> T.Tensor:
    """Build a masked adjacency matrix matrix between two masked/padded
    tensors.

    - Will create the self adjmat if only one tensor and mask is provided
    - Uses the tensors as a->senders (dim 1) vs b->receivers (dim 2)
        - Symettrical if doing self distance (tensor_b is none)

    args:
        mask_a: Which nodes in tensor_a are real (batch x n_nodes)
    kwargs:
        mask_b: Which nodes in tensor_b are real (batch x n_nodes)
        allow_self: Only applicable for self distances. Allows self connections.
    returns:
        adjmat connecting all real nodes in tensor_a and tensor_a(b)
    """

    # No gradients should be used in this step
    with T.no_grad():
        # Initialise a self distance matrix
        allow_self = allow_self and mask_b is None
        if mask_b is None:
            mask_b = mask_a

        # Calculate the matrix mask
        adjmat = mask_a.unsqueeze(-1) * mask_b.unsqueeze(-2)

        # Remove diagonal elements
        if not allow_self:
            adjmat = adjmat & ~T.diag_embed(T.full_like(mask_a, True))

    return adjmat.detach()


def knn(
    distmat: T.Tensor, k_val: int, k_restr_dim: str = "recv", top_k: bool = False
) -> T.BoolTensor:
    """Creates edges based on an infinite padded distance matrix.

    args:
        distmat: A batched infinite padded distance matrix with no self loops
        k_val: The value of K for the clustering
        k_restr_dim: The dimension over which to restrict (send or recv)
        top_k: If the top k distances should be used for clustering instead
    returns:
        adjmat: A new adjmat using knn over one of the dimensions in the dist mat
    """

    # Configuration checking
    if k_restr_dim not in ["send", "recv"]:
        raise ValueError("Unrecognised dimension restruction: ", k_restr_dim)

    # No gradients should be used in this step
    with T.no_grad():
        # If the size of the point cloud is smaller than k+1
        if distmat.shape[-1] - 1 <= k_val:
            # Simply return where the distance matrix is not infinite
            return distmat < distmat + EPS  # (Inf < Inf + EPS) is always false!

        # If looking for the top k connections then flip sign
        if top_k:
            distmat = distmat * -1

        # Check which dimension is being restricted
        restr_dim = {"send": 1, "recv": 2}[k_restr_dim]

        # Find the kth smallest distance across the correct dimension
        max_distances = T.kthvalue(distmat, k_val + 1, dim=restr_dim, keepdim=True)[0]
        max_distances = T.transpose(max_distances, -1, -2)

    # Build a connection if the distance between nodes is smaller or equal to the max
    return distmat < max_distances + EPS
