import numpy as np
import skdim
from tqdm import tqdm

def comp_d_mle(pcs_norm, Kvalues, dist = None, ind = None, smooth = True):
    '''
    computes d_MLE using different values of K (the number of analogues)
    '''
    dd_mle = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    dd_mle_smooth = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    
    for i, K in tqdm(enumerate(Kvalues)):

        if dist is not None and ind is not None:
            precomputed_knn_arrays = (dist[:, :K], ind[:, :K])
        else:
            precomputed_knn_arrays = None
            
        result = skdim.id.MLE().fit_transform_pw(pcs_norm, precomputed_knn_arrays = precomputed_knn_arrays, smooth = smooth)

        dd_mle[:, i] = result[0]
        dd_mle_smooth[:, i] = result[1]

    return dd_mle.squeeze(), dd_mle_smooth.squeeze()

def comp_d_pca(pcs_norm, Kvalues, dist = None, ind = None, smooth = True):
    '''
    computes d_pca using different values of K (the number of analogues)
    '''
    dd_pca = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    dd_pca_smooth = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    
    for i, K in tqdm(enumerate(Kvalues)):
        
        if ind is not None:
            precomputed_knn = ind[:, :K]
        else:
            precomputed_knn = None
        
        result = skdim.id.lPCA().fit_pw( 
            pcs_norm, precomputed_knn = precomputed_knn, smooth = smooth, n_jobs=14)
        
        dd_pca[:, i] = result.dimension_pw_.copy()
        dd_pca_smooth[:, i] = result.dimension_pw_smooth_.copy()

    return dd_pca.squeeze(), dd_pca_smooth.squeeze()

def comp_d_ess(pcs_norm, Kvalues, dist = None, ind = None, smooth = True):

    dd_ess = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    dd_ess_smooth = np.full((len(pcs_norm), len(Kvalues)), np.nan)
    
    for i, K in tqdm(enumerate(Kvalues)):

        if dist is not None and ind is not None:
            precomputed_knn_arrays = (dist[:, :K], ind[:, :K])
        else:
            precomputed_knn_arrays = None
        
        result = skdim.id.ESS().fit_transform_pw( 
            pcs_norm, precomputed_knn_arrays = precomputed_knn_arrays, smooth = smooth, n_jobs=14)

        dd_ess[:, i] = result[0].copy()
        dd_ess_smooth[:, i] = result[1].copy()

    return dd_ess.squeeze(), dd_ess_smooth.squeeze()