import xarray as xr
import numpy as np

def preprocess_data(data_folder, subsampling, max_horizon):
    # load
    allpcs = xr.open_dataset(data_folder + 'pcs.nc')['pseudo_pcs']
    eofs = xr.open_dataset(data_folder + 'eofs.nc')['eofs']
    pourc_EOF = xr.open_dataset(data_folder + 'pourc_eofs.nc')['variance_fractions']
    
    # normalize pcs but keep relative variances
    pcs_norm = allpcs / (allpcs.sel(mode=0)).std(dim='time')

    data_per_day = 2
    max_dh = data_per_day*max_horizon
    horizons = np.arange(np.timedelta64(0, 'D'), np.timedelta64(max_horizon, 'D'))
    
    # here I need to use the highest value for the horizon, in order for dataset_x to be compatible with the section where dependence of the CRPS on the horizon
    ind_tar = np.arange(len(pcs_norm) - max_dh)[::data_per_day*subsampling]
   
    pcs_norm.attrs['data_per_day'] = data_per_day
    
    return pcs_norm, ind_tar, horizons