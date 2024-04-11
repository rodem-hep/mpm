import torch as T


def get_timesteps(t_max: float, t_min: float, n_steps: int, p: float) -> T.Tensor:
    """Generate variable timesteps working back from t_max to t_min.

    Args:
        t_max: The maximum/starting time
        t_min: The minimum/final time
        n_steps: The number of time steps
        p: The degree of curvature, p=1 equal step size, recommened 7 for diffusion
    """
    idx = T.arange(n_steps, dtype=T.float32)
    invp = 1 / p
    tmax_invp = t_max**invp
    times = (tmax_invp + idx / (n_steps - 1) * (t_min**invp - tmax_invp)) ** p
    return times


def heun_sampler(
    model,
    initial_noise: T.Tensor,
    time_steps: T.Tensor,
    keep_all: bool = False,
    mask: T.Tensor | None = None,
    ctxt: T.BoolTensor | None = None,
    clip_predictions: tuple | None = None,
) -> None:
    # Get the initial noise for generation and the number of sammples
    batch_size = initial_noise.shape[0]
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)
    all_stages = [initial_noise]
    num_steps = len(time_steps)

    # Start with the initial noise
    x = initial_noise

    # Start iterating through each timestep
    for i in range(num_steps - 1):
        # Expancd the diffusion times for the number of samples in the batch
        diff_times = T.full((batch_size, 1), time_steps[i], device=model.device)
        diff_times_next = T.full(
            (batch_size, 1), time_steps[i + 1], device=model.device
        )

        # Calculate the derivative and apply the euler step
        # Note that this is the same as a single DDIM step! Triple checked!
        d = (x - model.denoise(x, diff_times, mask, ctxt)) / time_steps[i]
        x_next = x + (diff_times_next - diff_times).view(expanded_shape) * d

        # Apply the second order correction as long at the time doesnt go to zero
        if time_steps[i + 1] > 0:
            d_next = (
                x_next - model.denoise(x_next, diff_times_next, mask, ctxt)
            ) / time_steps[i + 1]
            x_next = (
                x
                + (diff_times_next - diff_times).view(expanded_shape) * (d + d_next) / 2
            )

        # Update the track
        x = x_next
        if keep_all:
            all_stages.append(x)

    return x, all_stages


def stochastic_sampler(
    model,
    initial_noise: T.Tensor,
    time_steps: T.Tensor,
    keep_all: bool = False,
    mask: T.Tensor | None = None,
    ctxt: T.BoolTensor | None = None,
    clip_predictions: tuple | None = None,
) -> None:
    # Get the initial noise for generation and the number of sammples
    batch_size = initial_noise.shape[0]
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)
    all_stages = [initial_noise]
    num_steps = len(time_steps)

    # Start with the initial noise
    x = initial_noise

    # Start iterating through each timestep
    for i in range(num_steps - 1):
        # Expancd the diffusion times for the number of samples in the batch
        diff_times = T.full((batch_size, 1), time_steps[i], device=model.device)
        diff_times_next = T.full(
            (batch_size, 1), time_steps[i + 1], device=model.device
        )

        # Calculate the derivative and apply the euler step
        d = (x - model.denoise(x, diff_times, mask, ctxt)) / time_steps[i]
        x_next = x + (diff_times_next - diff_times).view(expanded_shape) * d

        # Apply the second order correction as long at the time doesnt go to zero
        if time_steps[i + 1] > 0:
            d_next = (
                x_next - model.denoise(x_next, diff_times_next, mask, ctxt)
            ) / time_steps[i + 1]
            x_next = (
                x
                + (diff_times_next - diff_times).view(expanded_shape) * (d + d_next) / 2
            )

        # Update the track
        x = x_next
        if keep_all:
            all_stages.append(x)

    return x, all_stages
