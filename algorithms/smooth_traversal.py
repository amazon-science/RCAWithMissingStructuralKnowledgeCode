# ./algorithms/smooth_traversal.py
"""Function for running the SMOOTH TRAVERSAL algorithm from the paper. 
Code written by Patrick Blöbaum, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Callable, Dict

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.graph import node_connected_subgraph_view, is_root_node


def apply_smooth_traversal(graph: nx.DiGraph,
                           target_node: str,
                           normal_data: pd.DataFrame,
                           anomaly_data: pd.DataFrame,
                           anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
                           debug: bool = True) -> Dict[str, float]:
    """
    This is the implementation of the smooth traversal algorithm (algorithm 1 on the paper)
    """
    graph = node_connected_subgraph_view(graph, target_node)

    if anomaly_data.shape[0] > 1:
        anomaly_data = anomaly_data.iloc[[0]]

    all_scores = {}

    for node in graph.nodes:
        tmp_anomaly_scorer = anomaly_scorer()
        tmp_anomaly_scorer.fit(normal_data[node].to_numpy())
        tmp_score = tmp_anomaly_scorer.score(anomaly_data[node].to_numpy())

        if debug:
            print(f"Anomaly score of {node} is {tmp_score.squeeze()}")

        all_scores[node] = tmp_score.squeeze()

    score_gaps = {}

    for node in graph.nodes:
        if is_root_node(graph, node):
            score_gaps[node] = float(all_scores[node])
        else:
            score_gaps[node] = max(0, np.min([(all_scores[node] - all_scores[parent]) for parent in graph.predecessors(node)]))
    
    return score_gaps
