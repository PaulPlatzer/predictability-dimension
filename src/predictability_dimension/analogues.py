"""Analogue search and simple analogue-forecast scores."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def find_analogues(
    dataset: np.ndarray,
    ind_cat: np.ndarray | None = None,
    ind_tar: np.ndarray | None = None,
    K: int = 500,
    nn_algo: str = "auto",
    step_subsampling_catalogue: int = 1,
    loo: bool = False,
    dt_loo: int = 0,
    separate: bool = False,
    dt_separate: int = 0,
    seed_separate: int = 1312,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Find K nearest analogues and return distances plus absolute sample indices."""

    dataset = np.asarray(dataset)
    if ind_cat is None:
        ind_cat = np.arange(len(dataset))
    if ind_tar is None:
        ind_tar = np.arange(len(dataset))
        loo = True

    catalogue_indices = np.asarray(ind_cat)[::step_subsampling_catalogue]
    target_indices = np.asarray(ind_tar)

    overfetch = min(K + int(loo) + 2 * int(dt_loo), len(catalogue_indices))
    if overfetch <= 0:
        raise ValueError("The analogue catalogue is empty.")
    nn = NearestNeighbors(algorithm=nn_algo, n_neighbors=overfetch, n_jobs=n_jobs)
    nn.fit(dataset[catalogue_indices])

    dist, ind = nn.kneighbors(dataset[target_indices], return_distance=True)
    ind = catalogue_indices[ind]

    dist, ind = loo_procedure(dist, ind, K=K, loo=loo, dt_loo=dt_loo, ind_tar=target_indices)
    dist, ind = separate_trajectories(
        dist,
        ind,
        separate=separate,
        dt_separate=dt_separate,
        seed=seed_separate,
    )
    return dist, ind


def loo_procedure(
    dist: np.ndarray,
    ind: np.ndarray,
    K: int = 500,
    loo: bool = False,
    dt_loo: int = 0,
    ind_tar: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the target itself and nearby times from analogue candidates."""

    dist = dist.copy()
    ind = ind.copy()

    if loo or dt_loo > 0:
        if ind_tar is None:
            ind_tar = np.arange(len(ind))

        time_diff = np.abs(ind - np.asarray(ind_tar)[:, None])
        if loo:
            dist[time_diff == 0] = np.inf
        if dt_loo > 0:
            dist[time_diff < dt_loo] = np.inf

        # Re-sort because masked candidates are now placed at the end.
        order = np.argsort(dist, axis=1)
        dist = np.take_along_axis(dist, order, axis=1)[:, :K]
        ind = np.take_along_axis(ind, order, axis=1)[:, :K]
    else:
        dist = dist[:, :K]
        ind = ind[:, :K]

    return dist, ind


def separate_trajectories(
    dist: np.ndarray,
    ind: np.ndarray,
    separate: bool = True,
    dt_separate: int = 0,
    which_neighbour: str = "random",
    seed: int = 1312,
    return_array: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most one analogue from each temporally contiguous trajectory."""

    if not separate:
        return dist, ind

    rng = np.random.default_rng(seed)
    K = ind.shape[1]
    order_by_time = np.argsort(ind, axis=1)
    sorted_indices = np.take_along_axis(ind, order_by_time, axis=1)

    selected_dist, selected_ind, selected_counts = [], [], []

    for row in tqdm(range(len(ind)), desc="separate trajectories"):
        clusters = [[order_by_time[row, 0]]]
        cluster_id = 0

        for pos in range(1, K):
            previous_time = sorted_indices[row, pos - 1]
            current_time = sorted_indices[row, pos]
            if current_time - previous_time <= dt_separate + 1:
                clusters[cluster_id].append(order_by_time[row, pos])
            else:
                clusters.append([order_by_time[row, pos]])
                cluster_id += 1

        row_ind, row_dist = [], []
        for cluster in clusters:
            if which_neighbour == "best":
                choice = int(np.argmin(dist[row, cluster]))
            elif which_neighbour == "first":
                choice = 0
            elif which_neighbour == "last":
                choice = -1
            elif which_neighbour == "random":
                choice = int(rng.choice(len(cluster)))
            else:
                raise ValueError(f"Unknown selection rule: {which_neighbour!r}")
            row_ind.append(ind[row, cluster][choice])
            row_dist.append(dist[row, cluster][choice])

        order = np.argsort(row_dist)
        selected_ind.append(np.asarray(row_ind)[order])
        selected_dist.append(np.asarray(row_dist)[order])
        selected_counts.append(len(order))

    if return_array:
        Kmin = int(np.min(selected_counts))
        selected_dist = np.asarray([row[:Kmin] for row in selected_dist])
        selected_ind = np.asarray([row[:Kmin] for row in selected_ind])

    return selected_dist, selected_ind


def compute_diffs(
    dataset_y: np.ndarray,
    ind_tar: np.ndarray,
    ind_ana: np.ndarray,
    ind_cat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-analogue and analogue-analogue absolute differences."""

    target_y = dataset_y[ind_tar]
    analogue_indices = ind_ana
    if ind_cat is not None and np.max(ind_ana) < len(ind_cat):
        # Compatibility with older notebooks that stored analogue positions
        # relative to a catalogue instead of absolute sample indices.
        analogue_indices = np.asarray(ind_cat)[ind_ana]

    analogue_y = dataset_y[analogue_indices]
    diff_y = np.abs(target_y[:, None, :] - analogue_y)
    diff_ana_y = np.abs(analogue_y[:, None, :, :] - analogue_y[:, :, None, :])
    return diff_y, diff_ana_y


def compute_mae_mad(diff_y: np.ndarray, diff_ana_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute CRPS ingredients: mean absolute error and ensemble spread."""

    mae = np.mean(diff_y, axis=(1, 2))
    mad = np.mean(diff_ana_y, axis=(1, 2, 3))
    return mae, mad


def compute_CRPSana(
    dataset_x: np.ndarray,
    dataset_y: np.ndarray,
    ind_cat: np.ndarray | None = None,
    ind_tar: np.ndarray | None = None,
    K: int = 500,
    nn_algo: str = "auto",
    step_subsampling_catalogue: int = 1,
    loo: bool = True,
    dt_loo: int = 0,
    separate: bool = False,
    dt_separate: int = 0,
    seed_separate: int = 1312,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute the Continuous Ranked Probability Score of analogue forecasts."""

    if ind_tar is None:
        ind_tar = np.arange(len(dataset_x))
        loo = True

    _, ind = find_analogues(
        dataset_x,
        ind_cat=ind_cat,
        ind_tar=ind_tar,
        K=K,
        nn_algo=nn_algo,
        step_subsampling_catalogue=step_subsampling_catalogue,
        loo=loo,
        dt_loo=dt_loo,
        separate=separate,
        dt_separate=dt_separate,
        seed_separate=seed_separate,
        n_jobs=n_jobs,
    )
    diff_y, diff_ana_y = compute_diffs(dataset_y, np.asarray(ind_tar), ind)
    mae, mad = compute_mae_mad(diff_y, diff_ana_y)
    return mae - 0.5 * mad


def compute_CRPSclim(dataset_y: np.ndarray, ind_tar: np.ndarray | None = None, step_subsampling_compute: int = 100):
    """Exact climatological CRPS for small plain NumPy datasets."""

    dataset_y = np.asarray(dataset_y)
    if ind_tar is None:
        ind_tar = np.arange(len(dataset_y))[::step_subsampling_compute]

    diff_clim_y = np.abs(dataset_y[::step_subsampling_compute, None, :] - dataset_y[None, ::step_subsampling_compute, :])
    mad = np.mean(diff_clim_y)

    absolute_error_clim = np.abs(dataset_y[ind_tar, None] - dataset_y[None, :])
    mae = np.mean(absolute_error_clim, axis=(1, 2))
    return mae - 0.5 * mad


def compute_mae_monte_carlo(dataset_y: np.ndarray, ind_tar: np.ndarray, n_ref: int = 1000, random_seed: int = 1312):
    """Monte-Carlo MAE between targets and a climatological reference sample."""

    rng = np.random.default_rng(random_seed)
    ref_inds = rng.choice(len(dataset_y), size=min(n_ref, len(dataset_y)), replace=False)
    errors = np.abs(dataset_y[ind_tar][:, None, :] - dataset_y[ref_inds][None, :, :])
    return np.mean(errors, axis=(1, 2))


def compute_CRPSclim_fast(
    dataset_y: np.ndarray,
    ind_tar: np.ndarray | None = None,
    n_sample_pairs: int = 10000,
    n_ref: int = 1000,
    random_seed: int = 1312,
):
    """Fast climatological CRPS approximation for large plain NumPy datasets."""

    rng = np.random.default_rng(random_seed)
    dataset_y = np.asarray(dataset_y)
    n = len(dataset_y)
    if ind_tar is None:
        ind_tar = np.arange(n)

    i1 = rng.integers(0, n, size=n_sample_pairs)
    i2 = rng.integers(0, n, size=n_sample_pairs)
    mad = np.mean(np.abs(dataset_y[i1] - dataset_y[i2]))
    mae = compute_mae_monte_carlo(dataset_y, ind_tar, n_ref=n_ref, random_seed=random_seed)
    return mae - 0.5 * mad


def compute_density_proxy(distances: np.ndarray, k: int = 50) -> np.ndarray:
    """Mean distance to the K nearest analogues — large value means low density."""

    return distances[:, :k].mean(axis=1)


def compute_analogue_spread_horizons(
    states: np.ndarray,
    indices: np.ndarray,
    horizons: list[int],
    k: int | None = None,
) -> np.ndarray:
    """RMS ensemble spread of evolved analogues at each lead time.

    Parameters
    ----------
    states:
        Full state matrix, shape (n_times, n_features).
    indices:
        Absolute time indices of the K analogues for each target,
        shape (n_targets, K).
    horizons:
        Lead times in time steps (days for daily ERA5 data).
    k:
        Number of analogues to use (defaults to all K).

    Returns
    -------
    np.ndarray, shape (n_targets, len(horizons))
        Spread = sqrt(mean over features of variance over members).
    """

    n_targets, K_max = indices.shape
    k = min(k or K_max, K_max)
    n_times = len(states)
    spread = np.full((n_targets, len(horizons)), np.nan)

    for h_idx, h in enumerate(horizons):
        evolved = indices[:, :k] + h  # (n_targets, k)
        for i in range(n_targets):
            valid = evolved[i] < n_times
            if valid.sum() < 2:
                continue
            members = states[evolved[i, valid]]  # (n_valid, n_features)
            spread[i, h_idx] = np.sqrt(np.mean(np.var(members, axis=0, ddof=1)))

    return spread


def compute_crps_ana_horizons(
    states: np.ndarray,
    tar_idx: np.ndarray,
    indices: np.ndarray,
    horizons: list[int],
    k: int | None = None,
    n_pairs_mad: int = 200,
    rng_seed: int = 1312,
) -> np.ndarray:
    """Feature-averaged CRPS of analogue forecasts vs the verified state.

    CRPS(t, h) = mean_features[ MAE(t,h) ] - 0.5 * mean_features[ MAD(t,h) ]

    where MAE is the mean absolute error of evolved analogue members against
    the true state at t+h, and MAD is estimated by random pairwise sampling
    among the evolved analogue members.

    Parameters
    ----------
    states:
        Full state matrix, shape (n_times, n_features).
    tar_idx:
        Absolute ERA5 time indices of the target dates, shape (n_targets,).
    indices:
        Analogue indices for each target, shape (n_targets, K).
    horizons:
        Lead times in time steps (days for daily ERA5).
    k:
        Number of analogues to use (default: all K).
    n_pairs_mad:
        Random pairs used for Monte-Carlo MAD estimation per target.
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_targets, len(horizons))
    """

    n_targets = len(tar_idx)
    n_times = len(states)
    k = min(k or indices.shape[1], indices.shape[1])
    rng = np.random.default_rng(rng_seed)
    crps = np.full((n_targets, len(horizons)), np.nan)

    for h_idx, h in enumerate(horizons):
        evolved_tar = tar_idx + h          # (n_targets,)
        evolved_ana = indices[:, :k] + h   # (n_targets, k)

        for i in range(n_targets):
            if evolved_tar[i] >= n_times:
                continue
            valid = evolved_ana[i] < n_times
            if valid.sum() < 2:
                continue

            target  = states[evolved_tar[i]]          # (n_features,)
            members = states[evolved_ana[i, valid]]   # (k_valid, n_features)
            k_valid = members.shape[0]

            mae = float(np.mean(np.abs(target - members)))

            # Monte-Carlo MAD with random pairs (no replacement)
            n_pairs = min(n_pairs_mad, k_valid * (k_valid - 1) // 2)
            i1 = rng.integers(0, k_valid, size=n_pairs)
            i2 = rng.integers(0, k_valid, size=n_pairs)
            same = i1 == i2
            i1[same] = (i1[same] + 1) % k_valid
            mad = float(np.mean(np.abs(members[i1] - members[i2])))

            crps[i, h_idx] = mae - 0.5 * mad

    return crps


def compute_crps_clim_horizons(
    states: np.ndarray,
    tar_idx: np.ndarray,
    horizons: list[int],
    n_ref: int = 300,
    rng_seed: int = 42,
) -> np.ndarray:
    """Feature-averaged CRPS of a stationary climatological forecast.

    Uses the full ERA5 record as the climatological ensemble.  The ensemble
    spread term (MAD_clim) is constant across targets and horizons.

    Parameters
    ----------
    states:
        Full state matrix, shape (n_times, n_features).
    tar_idx:
        Absolute time indices of the target dates, shape (n_targets,).
    horizons:
        Lead times in time steps (days for daily ERA5).
    n_ref:
        Number of randomly drawn climatological members.
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_targets, len(horizons))
    """

    n_targets = len(tar_idx)
    n_times = len(states)
    rng = np.random.default_rng(rng_seed)
    crps = np.full((n_targets, len(horizons)), np.nan)

    # MAD_clim is constant (does not depend on target or horizon)
    r1 = rng.choice(n_times, size=n_ref, replace=False)
    r2 = rng.choice(n_times, size=n_ref, replace=False)
    mad_clim = float(np.mean(np.abs(states[r1] - states[r2])))

    for h_idx, h in enumerate(horizons):
        evolved_tar = tar_idx + h
        ref = states[rng.choice(n_times, size=n_ref, replace=False)]  # (n_ref, n_features)

        valid_t = evolved_tar < n_times
        targets = states[evolved_tar[valid_t]]  # (n_valid, n_features)

        # Compute MAE for each target vs the same climatological sample
        # Process in chunks to limit peak memory to ~200 MB
        chunk = 256
        mae_all = np.empty(valid_t.sum())
        for start in range(0, len(targets), chunk):
            batch = targets[start : start + chunk]  # (chunk, n_features)
            mae_all[start : start + chunk] = np.mean(
                np.abs(batch[:, None, :] - ref[None, :, :]), axis=(1, 2)
            )

        crps[valid_t, h_idx] = mae_all - 0.5 * mad_clim

    return crps


def compute_theta(ind: np.ndarray, q: float) -> np.ndarray:
    """Estimate persistence from temporal clustering of analogue indices."""

    ind_sorted = np.sort(ind, axis=1)
    cluster_breaks = (ind_sorted[:, 1:] - ind_sorted[:, :-1] - 1) > 0
    Nc = np.sum(cluster_breaks, axis=1)
    N = ind_sorted.shape[1] - 1
    span = (1.0 - q) * (ind_sorted[:, -1] - ind_sorted[:, 0])
    return (span + N + Nc - np.sqrt((span + N + Nc) ** 2 - 8.0 * Nc * span)) / (2.0 * span)


def compute_velocity(ind: np.ndarray, cat: np.ndarray, dt_vel: int = 1) -> np.ndarray:
    """Average phase-space speed around each set of analogues."""

    ind_time_neighbor = ind + dt_vel
    out_of_bounds = ind_time_neighbor > (len(cat) - 1)
    ind_time_neighbor[out_of_bounds] = ind[out_of_bounds] - dt_vel

    velocity = np.mean(np.sqrt(np.sum((cat[ind_time_neighbor] - cat[ind]) ** 2, axis=-1)), axis=1)
    return velocity
