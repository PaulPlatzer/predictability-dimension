import numpy as np
from tqdm import tqdm
import xarray as xr
from utils.analogues_functions import compute_CRPSana

def comp_mad(y, n_sample_pairs = 10000):
    '''
    computes the mean absolute difference (mad) as the mean of abs(Y - Y*) where Y and Y* are samples of y
    Since this is computationnally heavy to compute for all pairs, I do it only for n_sample_pairs
    '''
    if n_sample_pairs < len(y):
        # in that case only, one can choose indices randomly
        i1 = np.random.choice(len(y), size = n_sample_pairs, replace = False)
        i2 = np.random.choice(len(y), size = n_sample_pairs, replace = False)
    else:
        # take all the indices of y and construct a shuffled version
        i1 = np.arange(len(y))
        rng = np.random.default_rng()
        i2 = rng.permutation(len(y))

    return np.mean(np.abs(y.data[i1] - y.data[i2]))

def comp_mae(y, tar, n_ref = 1000):
    '''
    computes the mean absolute error of target with respect to the elements of y (as the mean of abs(y - target))
    Since this can be heavy if there are a lot of points in y, this is done only for n_ref random samples in y
    '''
    if n_ref < len(y):
        idxs = np.random.choice(len(y), size = n_ref, replace = False)
    else:
        idxs = np.arange(len(y))

    return np.mean(np.abs(y.data[idxs] - np.expand_dims(tar.data, axis = 0)))


def comp_CRPS_clim(y, targets, window_clim, n_ref = 1000, n_samples_pairs = 10000):
    '''
    computes the CRPS of the targets wrt y
    If window_clim is 30, the CRPS_clim of the 01/01/2000 will be done using only points whose times are in december and january
    y has dimensions ('time', 'mode') and targets has dimensions ('target_time', 'mode')
    '''
    CRPS_clim = np.full(len(targets), np.nan)

    absdiff = np.abs(y.time.dayofyear - targets.target_time.dt.dayofyear)
    within_window_clim = np.logical_or(absdiff < window_clim, absdiff > 366 - window_clim) # this has dimension time and target_time
    
    for i, tar in enumerate(tqdm(targets)):
        # restrict the y dataset to points which are in the same period than y.isel(time = ind)
        restricted_y = y.where(within_window_clim.isel(target_time = i), drop = True)
        
        # compute mad for the restricted y
        mad = comp_mad(restricted_y, n_samples_pairs)
        
        # compute mae
        mae = comp_mae(restricted_y, tar, n_ref)
        
        CRPS_clim[i] = mae - 0.5 * mad
        
    return CRPS_clim

def comp_CRPS_clim_L63(y, targets, window_clim, n_ref = 1000, n_samples_pair = 1000):
    '''
    same as comp_CRPS_clim, but not restricting on the window
    '''
    CRPS_clim = np.full(len(targets), np.nan)

    for i, tar in tqdm(enumerate(targets)):
        # compute mad
        mad = comp_mad(y, n_samples_pair)
        
        # compute mae
        mae = comp_mae(y, tar, n_ref)
        
        CRPS_clim[i] = mae - 0.5 * mad
        
    return CRPS_clim

def comp_CRPS_clim_horizons(pcs_norm, ind_tar, horizons, window_clim):
    
    CRPS_clim = xr.DataArray(np.full((len(horizons), len(ind_tar)), np.nan),
                         dims = ('horizon', 'initial_time'),
                         coords = {'horizon' : horizons, 'initial_time': pcs_norm.time.isel(time = ind_tar).data},
                         attrs = {'window_clim': window_clim}
                        )
    
    max_dh = pcs_norm.data_per_day * horizons.max()//np.timedelta64(1, 'D')
    l = len(pcs_norm)
    
    for h in horizons:
        # the CRPS of the climatology should not depend on the horizon h, but we want dataset_y to be the same than for CRPS_ana
        dh = pcs_norm.data_per_day * h // np.timedelta64(1, 'D')           # number of time steps for this value of the horizon
        dataset_y = pcs_norm[dh: l-(max_dh - dh)]                          # dataset_y contains from which we compute the climatology, so we need to shift it by dh forward in time
        targets = pcs_norm[ind_tar].rename({'time': 'target_time'})        # the points for which we want to compute the CRPS
        CRPS_clim.loc[{'horizon': h}] = comp_CRPS_clim(dataset_y, targets, window_clim)

    return CRPS_clim

def comp_CRPS_ana_horizons(pcs_norm, ind_tar, horizons, Kvalues):
    CRPS_ana = xr.DataArray(np.full((len(horizons), len(Kvalues), len(ind_tar)), np.nan),
                            dims = ('horizon', 'K', 'initial_time'),
                            coords = {'horizon': horizons, 'K': Kvalues, 'initial_time': pcs_norm.time[ind_tar].data})

    max_dh = pcs_norm.data_per_day * horizons.max()//np.timedelta64(1, 'D')
    l = len(pcs_norm)

    for h in horizons:
        dh = pcs_norm.data_per_day* h // np.timedelta64(1, 'D')           # number of time steps for this value of the horizon
        dataset_x = pcs_norm[:-max_dh]                                    # contains the points in which we look for analogues
        dataset_y = pcs_norm[dh: l-(max_dh - dh)]                          # dataset_y is contains the points shifted by dh forward in time
        step_subsampling_catalogue = pcs_norm.data_per_day*4              # subsampling of the analogues catalog
        for j, K in enumerate(tqdm(Kvalues)):
            CRPS_ana.loc[{'horizon' : h, 'K': K}] = compute_CRPSana(dataset_x.data, dataset_y.data, ind_tar=ind_tar, loo=True, dt_loo = 2*30, K=K,
                                       step_subsampling_catalogue=step_subsampling_catalogue, n_jobs=14)

    return CRPS_ana

    