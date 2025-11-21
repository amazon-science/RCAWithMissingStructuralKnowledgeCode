# ./algorithms/simple_traversal.py
"""Function for running the Traversal algorithm as described in 
Chen et.al. (2014) "CauseInfer: Automatic and distributed performance diagnosis with hierarchical causality graph in large distributed systems" 
https://ieeexplore.ieee.org/document/6848128.
Code written by Patrick Blöbaum, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Callable, Dict

import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.graph import DirectedGraph, node_connected_subgraph_view


def apply_simple_traversal(
        graph: DirectedGraph,
        target_node: str,
        normal_data: pd.DataFrame,
        anomaly_data: pd.DataFrame,
        anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
        anomaly_threshold: float = 3,
        debug: bool = True) -> Dict[str, float]:
    """
    This algorithm is a traversal algorithm as described in
    Chen et.al. (2014) "CauseInfer: Automatic and distributed performance diagnosis with hierarchical causality graph in large distributed systems"
    https://ieeexplore.ieee.org/document/6848128

    The implementation is ours.
    """
    traversal_sub_graph = node_connected_subgraph_view(graph, target_node)
    
    if anomaly_data.shape[0] > 1:
        anomaly_data = anomaly_data.iloc[[0]]

    anomaly_nodes = []

    for node in traversal_sub_graph.nodes:
        tmp_anomaly_scorer = anomaly_scorer()
        tmp_anomaly_scorer.fit(normal_data[node].to_numpy())
        tmp_score = tmp_anomaly_scorer.score(anomaly_data[node].to_numpy())

        if debug:
            print(f"Anomaly score of {node} is {tmp_score}")

        if tmp_score > anomaly_threshold:
            anomaly_nodes.append(node)

    result = {}

    for anomaly_node in anomaly_nodes:
        parents = traversal_sub_graph.predecessors(anomaly_node)
        if not set(anomaly_nodes) & set(parents):
            result[anomaly_node] = 1
    
    return result