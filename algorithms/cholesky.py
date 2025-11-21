# ./algorithms/circa.py
"""Wrapper function for running the "Cholesky" algorithm as described in 
Li et.al. (2025) "Root cause discovery via permutations and Cholesky decomposition"
https://arxiv.org/abs/2410.12151, with the original code found in https://github.com/Jinzhou-Li/RootCauseDiscovery.
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict

import pandas as pd

from algorithms.RootCauseDiscovery.funcs.root_cause_discovery_funcs import root_cause_discovery_highdim_parallel, root_cause_discovery_main


def apply_cholesky(normal_data: pd.DataFrame,
                   anomaly_data: pd.DataFrame,
                   cholesky_type: str="highdim") -> Dict[str, float]:
    """
    This is a wrapper around the method in 
    Li et.al. (2025) "Root cause discovery via permutations and Cholesky decomposition"
    https://arxiv.org/abs/2410.12151
    
    The code for this method can be found in https://github.com/Jinzhou-Li/RootCauseDiscovery
    """
    if anomaly_data.shape[0] > 1:
        anomaly_data = anomaly_data.iloc[[0]]
    
    variable_names = normal_data.columns
    
    if cholesky_type == "highdim":
        chol_scores = root_cause_discovery_highdim_parallel(
            X_obs=normal_data.to_numpy(),
            X_int=anomaly_data.to_numpy().flatten(),
            n_jobs=-1,
            y_idx_z_threshold=1.5,
            nshuffles=1,
            verbose=False,
            Precision_mat=None
        )
        result = {variable_names[i]: float(chol_scores[0][i]) for i in range(len(chol_scores[0]))}
    elif cholesky_type == "main":
        n_shuffles = 10
        chol_scores = root_cause_discovery_main(X_obs=normal_data.to_numpy(),
                                                X_int=anomaly_data.to_numpy().flatten(),
                                                nshuffles=n_shuffles,
                                                verbose=False)
        result = {variable_names[i]: float(chol_scores[i]) for i in range(len(chol_scores))}

    return result