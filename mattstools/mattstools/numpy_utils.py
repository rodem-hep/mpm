from typing import Optional

import numpy as np


def unison_shuffled_copies(*args) -> tuple:
    """Shuffle multiple arrays in unison along the first axis.

    Parameters
    ----------
    *args : array_like
        One or more input arrays to be shuffled.

    Returns
    -------
    tuple
        A tuple of shuffled copies of the input arrays.

    Raises
    ------
    AssertionError
        If the input arrays do not have the same length along the first axis.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> c, d = unison_shuffled_copies(a, b)
    >>> print(c)
    [2 1 3]
    >>> print(d)
    [5 4 6]

    Notes
    -----
    This function uses `numpy.random.permutation` to generate a random permutation
    of indices and applies it to each input array.
    """
    n = len(args[0])
    assert all(len(a) == n for a in args)
    p = np.random.permutation(n)
    return (a[p] for a in args)


def onehot_encode(
    a: np.ndarray, max_idx: None | int = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    max_idx = max_idx or a.max()
    ncols = max_idx + 1
    out = np.zeros((a.size, ncols), dtype=dtype)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out


def interweave(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    """Combine two arrays by alternating along the first dimension
    args:
        a: array to take even indices
        b: array to take odd indices
    returns:
        combined array
    """
    arr_comb = np.empty(
        (arr_1.shape[0] + arr_2.shape[0], *arr_1.shape[1:]), dtype=arr_1.dtype
    )
    arr_comb[0::2] = arr_1
    arr_comb[1::2] = arr_2
    return arr_comb


def sum_other_axes(array: np.ndarray, axis: int) -> np.ndarray:
    """Applies numpy sum to all axes except one in an array."""
    axes_for_sum = [i for i in range(len(array.shape))]
    axes_for_sum.pop(axis)
    return array.sum(axis=tuple(axes_for_sum))


def mid_points(array: np.ndarray) -> np.ndarray:
    """Return the midpoints of an array, one smaller."""
    return (array[1:] + array[:-1]) / 2


def undo_mid(array: np.ndarray) -> np.ndarray:
    """Undo the midpoints, trying to get the bin boundaries."""
    array = np.array(array)  # Have to include this because of pandas
    half_bw = (array[1] - array[0]) / 2  # Assumes constant bin widths
    array = np.insert(array + half_bw, 0, array[0] - half_bw)
    return array


def chunk_given_size(a: np.ndarray, size: int, axis: int = 0) -> np.ndarray:
    """Split an array into chunks along an axis, the final chunk will be
    smaller."""
    return np.split(a, np.arange(size, a.shape[axis], size), axis=axis)


def mask_list(arrays: list, mask: np.ndarray) -> list:
    """Applies a mask to a list of arrays."""
    return [a[mask] for a in arrays]


def log_clip(
    data: np.ndarray, clip_min: Optional[float] = 1e-6, clip_max: Optional[float] = None
) -> np.ndarray:
    """Applies a clip and then the log function, typically to prevent neg
    infs."""
    return np.log(np.clip(data, clip_min, clip_max))


def min_loc(data: np.ndarray) -> tuple:
    """Returns the idx for the minimum of a multidimensional array."""
    return np.unravel_index(data.argmin(), data.shape)


def log_squash(data: np.ndarray) -> np.ndarray:
    """Apply a log squashing function for distributions with high tails."""
    return np.sign(data) * np.log(np.abs(data) + 1)


def undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return np.sign(data) * (np.exp(np.abs(data)) - 1)


def empty_0dim_like(arr: np.ndarray) -> np.ndarray:
    """Returns an empty array with similar size as the input but with its final
    dimension size reduced to 0."""

    # Get all but the final dimension
    all_but_last = arr.shape[:-1]

    # Ensure that this is a tuple/list so it can agree with return syntax
    if isinstance(all_but_last, int):
        all_but_last = [all_but_last]

    return np.empty((*all_but_last, 0), dtype=arr.dtype)


def group_by(a: np.ndarray) -> np.ndarray:
    """A groupby method which runs over a numpy array, binning by the first
    column and making many seperate arrays as results."""
    a = a[a[:, 0].argsort()]
    return np.split(a[:, 1:], np.unique(a[:, 0], return_index=True)[1][1:])
