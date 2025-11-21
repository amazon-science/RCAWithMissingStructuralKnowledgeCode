# ./algorithms/score_ordering.py
"""Function for running the SCORE ORDERING algorithm described in the paper.
Code written by Patrick Blöbaum, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""


from typing import Dict, Callable

import networkx as nx
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer


def apply_score_ordering(graph: nx.DiGraph,
                      normal_data: pd.DataFrame,
                      anomaly_data: pd.DataFrame,
                      anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer) -> Dict[str, float]:
    """
    This is the implementation of the score ordering algorithm (algorithm 2 in the paper)
    """
    all_nodes = list(graph)

    # Training the anomaly scorers with 2k observations of not anomalous data.
    scorers = {}
    scores = {}
    for n in all_nodes:
        scorers[n] = ITAnomalyScorer(anomaly_scorer())
        scorers[n].fit(normal_data[n].to_numpy())
        
        # Scoring the anomalous samples.
        scores[n] = float(scorers[n].score(anomaly_data[n].iloc[0:1].to_numpy()[0]))

    return scores