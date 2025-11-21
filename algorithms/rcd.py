# ./algorithms/rcd.py
"""Wrapper around the RCD method as implemented in https://github.com/salesforce/PyRCA originally
from https://github.com/azamikram/rcd and described in https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf.
Implemented in PyRCA (https://github.com/salesforce/PyRCA).
Wrapper code written by William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

from typing import Dict

import networkx as nx
import pandas as pd

from utils import preprocess_data

from algorithms.petshop_root_cause_analysis_main.code.hierarchical_rcd import make_hierarchical_rcd


def apply_rcd(graph: nx.DiGraph,
                target_node: str,
                normal_data: pd.DataFrame,
                anomaly_data: pd.DataFrame) -> Dict[str, float]:
    """
    Wrapper around the RCD method as implemented in https://github.com/salesforce/PyRCA originally
        from https://github.com/azamikram/rcd.

    Paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf
    """
    if anomaly_data.shape[0] < 2:
        raise ValueError("The anomaly data must have at least two samples to run RCD!")
    
    normal_metrics = preprocess_data(normal_data)
    abnormal_metrics = preprocess_data(anomaly_data)

    rcd_runner = make_hierarchical_rcd(root_cause_top_k = len(graph))

    causes = rcd_runner(
        graph=graph.reverse(),
        target_node=target_node,
        target_metric="latency",
        target_statistic="Average",
        normal_metrics=normal_metrics,
        abnormal_metrics=abnormal_metrics
    )

    return {x.node: x.score for x in causes}