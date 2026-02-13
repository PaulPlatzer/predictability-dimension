## All codes written by Paul Platzer.

import numpy as np
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors

## ----------------- ##
## ----------------- ##
## ----------------- ##
## FINDING ANALOGUES ##
## ----------------- ##
## ----------------- ##
## ----------------- ##

def find_analogues(dataset, ind_cat=None, ind_tar=None, K=500, nn_algo='auto',
                   step_subsampling_catalogue=1, loo=False, dt_loo=0,
                   separate=False, dt_separate=0, seed_separate=1312, n_jobs=1):
    """
    Find analogues in a dataset using k-nearest neighbors with optional post-processing to remove
    nearby points in time (leave-one-out) and correlated trajectories.

    Parameters:
    - dataset: np.ndarray, full dataset where each row is a sample.
    - ind_cat: indices of the catalogue set. If None, uses all.
    - ind_tar: indices of the target set. If None, uses all and enables LOO.
    - K: number of analogues to retain per target.
    - nn_algo: algorithm for NearestNeighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
    - step_subsampling_catalogue: subsampling step for the catalogue (e.g., for temporal thinning).
    - loo: whether to enable leave-one-out (i.e., target in catalogue).
    - dt_loo: time exclusion window around the target index (used with LOO).
    - separate: whether to avoid analogues from the same trajectory (within dt_separate).
    - dt_separate: temporal threshold to define "same trajectory".
    - seed_separate: seed for randomness when choosing between correlated analogues.
    - n_jobs: number of parallel jobs for NearestNeighbors.

    Returns:
    - dist: distances of retained analogues
    - ind: indices of retained analogues in the catalogue
    """
    
    if ind_cat is None:
        ind_cat = np.arange(len(dataset))  # Use full dataset as catalogue
    
    if ind_tar is None:
        ind_tar = np.arange(len(dataset))  # Use full dataset as targets
        loo = True                         # Enable leave-one-out

    # Define catalogue and target sets (catalogue possibly subsampled)
    catalogue = dataset[ind_cat[::step_subsampling_catalogue]]
    targets = dataset[ind_tar]

    # Initialize k-NN with overfetch to accommodate removal of unwanted neighbors
    nn = NearestNeighbors(algorithm=nn_algo, n_neighbors=K + int(loo) + 2 * dt_loo, n_jobs=n_jobs)
    nn.fit(catalogue)

    # Perform neighbor search
    dist, ind = nn.kneighbors(targets, return_distance=True)

    # Correct indices if catalogue was subsampled
    ind *= step_subsampling_catalogue

    # Remove self-analogues and temporal neighbors from the same trajectory (if needed)
    dist, ind = loo_procedure(dist, ind, K, loo, dt_loo, ind_tar)

    # Remove multiple analogues from the same trajectory
    dist, ind = separate_trajectories(dist, ind, separate=separate,
                                      dt_separate=dt_separate, seed=seed_separate)

    return dist, ind


def loo_procedure(dist, ind, K=500, loo=False, dt_loo=0, ind_tar=None):
    """
    Remove analogues that are too close in time to the target (including self if LOO is True).

    Parameters:
    - dist, ind: initial distances and indices from k-NN
    - K: number of analogues to keep
    - loo: remove first neighbor (self)
    - dt_loo: time exclusion range around target
    - ind_tar: target indices to compare to

    Returns:
    - filtered dist and ind arrays
    """
    # Remove first neighbor (target itself) if LOO
    dist = dist[:, int(loo):]
    ind = ind[:, int(loo):]

    if dt_loo > 0:
        if ind_tar is None:
            ind_tar = np.arange(len(ind))

        # Mask distances of temporal neighbors
        time_diff = np.abs(ind - np.repeat(ind_tar[:, np.newaxis], dist.shape[1], axis=1))
        dist[time_diff < dt_loo] = 1e12  # Temporarily mark large distance

        # Resort and truncate
        sort = np.argsort(dist, axis=1)
        dist = np.take_along_axis(dist, sort, axis=1)[:, :K]
        ind = np.take_along_axis(ind, sort, axis=1)[:, :K]

    return dist, ind


def separate_trajectories(dist, ind, separate=True, dt_separate=0,
                          which_neighbour='random', seed=1312, return_array=True):
    """
    Enforce independence by selecting only one analogue from each trajectory
    (where samples are within dt_separate of each other).

    Parameters:
    - dist, ind: input distance and index arrays
    - separate: whether to apply this filtering
    - dt_separate: time gap to consider analogues part of the same trajectory
    - which_neighbour: selection strategy from each group ['random', 'best', 'first', 'last']
    - seed: random seed
    - return_array: whether to return uniform-length arrays (min K retained)

    Returns:
    - dist, ind: filtered arrays (possibly truncated to minimum K across samples)
    """
    if not separate:
        return dist, ind

    K = ind.shape[1]
    argsort_indices = np.argsort(ind, axis=1)  # Sort analogues temporally
    sorted_indices = np.take_along_axis(ind, argsort_indices, axis=1)

    dist_separated = []
    ind_separated = []
    K_separated = []

    for j in tqdm(range(len(ind))):
        # Group analogues into trajectories based on time gap
        clusters = [[argsort_indices[j][0]]]
        i, l = 0, 0

        while i < K - 1:
            while i < K - 1 and sorted_indices[j][i + 1] - sorted_indices[j][i] <= dt_separate + 1:
                clusters[l].append(argsort_indices[j][i + 1])
                i += 1
            i += 1
            if i < K:
                clusters.append([argsort_indices[j][i]])
                l += 1

        ind_difftraj, dist_difftraj = [], []

        # Choose one analogue per cluster
        for cluster in clusters:
            if which_neighbour == 'best':
                k = np.argmin(dist[j][cluster])
            elif which_neighbour == 'random':
                rgn = np.random.default_rng(seed)
                k = rgn.choice(len(cluster))
            elif which_neighbour == 'first':
                k = 0
            elif which_neighbour == 'last':
                k = -1
            ind_difftraj.append(ind[j][cluster][k])
            dist_difftraj.append(dist[j][cluster][k])

        # Sort selected analogues by distance
        argsort_sep = np.argsort(dist_difftraj)
        ind_separated.append(np.array(ind_difftraj)[argsort_sep])
        dist_separated.append(np.array(dist_difftraj)[argsort_sep])
        K_separated.append(len(argsort_sep))

    if return_array:
        Kmin = np.min(K_separated)
        dist_separated = np.array([d[:Kmin] for d in dist_separated])
        ind_separated = np.array([i[:Kmin] for i in ind_separated])

    return dist_separated, ind_separated


## ------------------- ##
## ------------------- ##
## ------------------- ##
## PERSISTENCE INDICES ##
## ------------------- ##
## ------------------- ##
## ------------------- ##


# Theta estimator based on indices and q
def compute_theta( ind , q ):
    # TO COMMENT
    ind_sorted = np.sort(ind, axis=1) # sort analogues by time-index and not by growing distance to the target
    Nc = np.sum(  ( ( ind_sorted[:,1:] - ind_sorted[:,:-1] - 1) > 0 )   , axis = 1 )
    N = ind_sorted.shape[1] - 1
    tmp = ( 1.0 - q ) * ( ind_sorted[:,-1] - ind_sorted[:,0] )
    return ( tmp + N + Nc - np.sqrt( np.power( tmp + N + Nc , 2. ) - 8. * Nc * tmp ) ) / ( 2. * tmp )


def compute_velocity(ind, cat, dt_vel=1):
    """
    Compute average velocity over a set of analogs.

    Parameters:
    -----------
    ind : ndarray of shape (n_targets, n_analogs)
        Indices of analogs in the catalog for each target.
    cat : ndarray of shape (n_catalog, n_features)
        Catalog of state vectors.
    dt_vel : int
        Time step over which to compute velocity (default is 1).
    
    Returns:
    --------
    velocity : ndarray of shape (n_targets,)
        Average velocity for each target, estimated from the analogs.
    """
    ind_time_neighbor = ind + dt_vel

    # Handle boundary conditions: if out of bounds, use symmetric backward time step
    out_of_bounds = ind_time_neighbor > (len(cat) - 1)
    ind_time_neighbor[out_of_bounds] = ind[out_of_bounds] - dt_vel

    # Compute Euclidean distance between cat[t+dt] and cat[t]
    velocity = np.mean(np.sqrt(
        np.sum((cat[ind_time_neighbor] - cat[ind])**2, axis=-1)
    ), axis=1)

    return velocity


## -------------------- ##
## -------------------- ##
## -------------------- ##
## ANALOGUE FORECASTING ##
## -------------------- ##
## -------------------- ##
## -------------------- ##


def compute_diffs(dataset_y, ind_tar, ind_cat, ind_ana):
    """
    Compute differences in output space for evaluation of analogue forecasts.

    Parameters:
    - dataset_y: array of shape (n_samples, output_dim), output features of the dataset.
    - ind_tar: array of indices for the target points.
    - ind_cat: array of indices for the catalogue points.
    - ind_ana: array of shape (n_targets, K), indices of selected analogues (relative to ind_cat).

    Returns:
    - diff_y: absolute differences between targets and their analogues,
              shape = (n_targets, K, output_dim)
    - diff_ana_y: absolute pairwise differences between analogues of the same target,
                  shape = (n_targets, K, K, output_dim)
    """
    # Differences between target outputs and analogue outputs
    diff_y = np.abs(dataset_y[ind_tar][:, None, :] - dataset_y[ind_cat[ind_ana]])

    # Differences between analogue outputs (pairwise for each target)
    ana_y = dataset_y[ind_cat[ind_ana]]
    diff_ana_y = np.abs(ana_y[:, None, :, :] - ana_y[:, :, None, :])
    
    return diff_y, diff_ana_y


def compute_mae_mad(diff_y, diff_ana_y):
    """
    Compute mean absolute error (MAE) and mean absolute difference (MAD) from diffs.

    Parameters:
    - diff_y: absolute differences between targets and analogues, shape = (n_targets, K, output_dim)
    - diff_ana_y: pairwise differences between analogues, shape = (n_targets, K, K, output_dim)

    Returns:
    - mae: mean absolute error across analogues and features, shape = (n_targets,)
    - mad: mean absolute difference across analogue pairs and features, shape = (n_targets,)
    """
    mae = np.mean(diff_y, axis=(1, 2))       # Mean over analogues and output features
    mad = np.mean(diff_ana_y, axis=(1, 2, 3)) # Mean over all analogue pairs and output features

    return mae, mad
    

def compute_CRPSana(dataset_x, dataset_y, ind_cat=None, ind_tar=None, K=500, nn_algo='auto',
                   step_subsampling_catalogue=1, loo=True, dt_loo=0,
                   separate=False, dt_separate=0, seed_separate=1312, n_jobs=1):
    """
    Compute CRPS (Continuous Ranked Probability Score) for analogue forecasts.

    Parameters:
    - dataset_x: input space used to search for analogues (shape = [n_samples, features])
    - dataset_y: output space used to compute forecast skill (shape = [n_samples, output_features])
    - ind_cat, ind_tar, K, etc.: see `find_analogues`

    Returns:
    - CRPS: CRPS values for each target (shape = [n_targets])
    """

    # Default catalogue and target indices
    if ind_cat is None:
        ind_cat = np.arange(len(dataset_x))
    
    if ind_tar is None:
        ind_tar = np.arange(len(dataset_x))
        loo = True

    # Find analogues using k-NN + optional filtering
    dist, ind = find_analogues(dataset_x, ind_cat=ind_cat, ind_tar=ind_tar, K=K, nn_algo=nn_algo,
                               step_subsampling_catalogue=step_subsampling_catalogue, loo=loo, dt_loo=dt_loo,
                               separate=separate, dt_separate=dt_separate, seed_separate=seed_separate, n_jobs=n_jobs)
    
    # Compute differences in output space
    diff_y, diff_ana_y = compute_diffs(dataset_y, ind_tar=ind_tar, ind_cat=ind_cat, ind_ana=ind)

    # Compute mean absolute error (target vs. analogues) and analogue spread (MAD)
    mae, mad = compute_mae_mad(diff_y=diff_y, diff_ana_y=diff_ana_y)

    # CRPS is mean forecast error minus half the spread
    CRPS = mae - 0.5 * mad

    return CRPS


def compute_CRPSclim(dataset_y, ind_tar=None, step_subsampling_compute=100):
    """
    Compute CRPS for climatological forecasts.

    Parameters:
    - dataset_y: output values for the full dataset
    - ind_tar: optional subset of targets (defaults to subsampled full set)
    - step_subsampling_compute: step size for subsampling used in CRPS estimation

    Returns:
    - CRPS: CRPS values for each target
    """
    if ind_tar is None:
        ind_tar = np.arange(len(dataset_y))[::step_subsampling_compute]

    # Mean absolute difference across the climatology (MAD)
    diff_clim_y = np.abs(dataset_y[::step_subsampling_compute, None, :] -
                         dataset_y[None, ::step_subsampling_compute, :])
    mad = np.mean(diff_clim_y)

    # Mean absolute error between target values and full climatology
    absolute_error_clim = np.abs(dataset_y[ind_tar, None] - dataset_y[None, :])
    mae = np.mean(absolute_error_clim, axis=(1, 2))  # Result: (n_targets,)

    CRPS = mae - 0.5 * mad

    return CRPS


def compute_mae_monte_carlo(dataset_y, ind_tar, n_ref=1000, random_seed=1312):
    """
    Approximate mean absolute error between targets and climatology using Monte Carlo sampling.

    Parameters:
    - dataset_y: output data (n_samples, output_features)
    - ind_tar: indices of target samples
    - n_ref: number of samples to randomly draw from climatology
    - random_seed: RNG seed

    Returns:
    - mae: MAE between each target and the reference climatology (shape = [n_targets])
    """
    rng = np.random.default_rng(random_seed)
    ref_inds = rng.choice(len(dataset_y), size=n_ref, replace=False)
    ref_y = dataset_y[ref_inds]

    errors = np.abs(dataset_y[ind_tar][:, None, :] - ref_y[None, :, :])
    mae = np.mean(errors, axis=(1, 2))  # Average over reference samples and features

    return mae


def compute_CRPSclim_fast(dataset_y, ind_tar=None, n_sample_pairs=10000, n_ref=1000, random_seed=1312):
    """
    Fast approximation of CRPS for climatological forecasts using sampling.

    Parameters:
    - dataset_y: output values (n_samples, output_features)
    - ind_tar: optional target indices (default: all)
    - n_sample_pairs: number of random pairwise samples for MAD estimation
    - n_ref: number of climatology samples for each target (MAE estimation)
    - random_seed: RNG seed

    Returns:
    - CRPS: approximate CRPS values for each target
    """
    rng = np.random.default_rng(random_seed)
    N = len(dataset_y)

    if ind_tar is None:
        ind_tar = np.arange(N)

    # Estimate MAD from sampled climatology pairs
    i1 = rng.integers(0, N, size=n_sample_pairs)
    i2 = rng.integers(0, N, size=n_sample_pairs)
    mad = np.mean(np.abs(dataset_y[i1] - dataset_y[i2]))

    # Estimate MAE from Monte Carlo sampling
    mae = compute_mae_monte_carlo(dataset_y, ind_tar, n_ref=n_ref, random_seed=random_seed)

    CRPS = mae - 0.5 * mad
    return CRPS
