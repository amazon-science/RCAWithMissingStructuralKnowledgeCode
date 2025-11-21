# ./algorithms/circa.py
"""Wrapper function for running the CIRCA algorithm as described in 
Li et.al. (2022) "Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition"
https://dl.acm.org/doi/10.1145/3534678.3539041, and implemented in PyRCA (https://github.com/salesforce/PyRCA).
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict

import networkx as nx
import pandas as pd

from utils import preprocess_data

from algorithms.petshop_root_cause_analysis_main.code.circa import make_circa


def apply_circa(graph: nx.DiGraph,
                target_node: str,
                normal_data: pd.DataFrame,
                anomaly_data: pd.DataFrame) -> Dict[str, float]:
    """
    This is a wrapper around the method in 
    Li et.al. (2022) "Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition"
    https://dl.acm.org/doi/10.1145/3534678.3539041
    
    The code for this method can be found in https://github.com/salesforce/PyRCA
    """
    normal_metrics = preprocess_data(normal_data)
    abnormal_metrics = preprocess_data(anomaly_data)

    circa_runner = make_circa(root_cause_top_k = len(graph), adjustment = False)

    causes = circa_runner(
        graph=graph.reverse(),
        target_node=target_node,
        target_metric="latency",
        target_statistic="Average",
        normal_metrics=normal_metrics,
        abnormal_metrics=abnormal_metrics
    )

    return {x.node: x.score for x in causes}
