# ./algorithms/epsilon_diagnosis.py
"""Wrapper around epsilon-diagnosis RCA method as implemented in https://github.com/salesforce/PyRCA. 
Described in https://dl.acm.org/doi/10.1145/3308558.3313653.
Wrapper code written by William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict

import networkx as nx
import pandas as pd

from utils import preprocess_data

from algorithms.petshop_root_cause_analysis_main.code.epsilon_diagnosis import make_epsilon_diagnosis


def apply_epsilon_diagnosis(graph: nx.DiGraph,
                target_node: str,
                normal_data: pd.DataFrame,
                anomaly_data: pd.DataFrame) -> Dict[str, float]:
    """
    Wrapper around epsilon-diagnosis RCA method as implemented in https://github.com/salesforce/PyRCA.

    Paper: https://dl.acm.org/doi/10.1145/3308558.3313653
    """
    if anomaly_data.shape[0] < 2:
        raise ValueError("The anomaly data must have at least two samples to run RCD!")
    
    normal_metrics = preprocess_data(normal_data)
    abnormal_metrics = preprocess_data(anomaly_data)

    epsilon_diagnosis_runner = make_epsilon_diagnosis(root_cause_top_k = len(graph))

    causes = epsilon_diagnosis_runner(
        graph=graph.reverse(),
        target_node=target_node,
        target_metric="latency",
        target_statistic="Average",
        normal_metrics=normal_metrics,
        abnormal_metrics=abnormal_metrics
    )

    return {x.node: x.score for x in causes}