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
