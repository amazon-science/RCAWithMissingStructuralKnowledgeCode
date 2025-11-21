# ./sock_shop.py
"""Utility functions for running Sock-shop experiments. 
Code written by William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Callable

from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer


def create_sock_shop_graph():
    """Creates a networkx DiGraph object based on the call graph 
    displayed in ./sockshop_call_graph.png
    """

    G = nx.DiGraph()

    edges = [
    # User interaction with front-end
    ('front-end', 'catalogue'),
    ('front-end', 'user'),
    ('front-end', 'carts'),
    ('front-end', 'orders'),

    # Orders service orchestration
    ('orders', 'user'),
    ('orders', 'carts'),
    ('orders', 'payment'),
    ('orders', 'orders-db'),

    # Asynchronous shipping process via message queue
    ('orders', 'queue-master'),
    ('queue-master', 'rabbitmq'),
    
    # shipping listens for and receives messages from rabbitmq
    ('shipping', 'rabbitmq'),

    # Service to Database connections
    ('user', 'user-db'),
    ('user', 'session-db'),
    ('catalogue', 'catalogue-db'),
    ('carts', 'carts-db'),
    ]

    G.add_edges_from(edges)
    return G


def preprocess_sock_shop(
        dataframe: pd.DataFrame,
        root_cause: str,
        issue: str,
        inject_time: int,
        anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
        anomaly_threshold: float = 3,
        use_all_anomaly_samples: bool = False,
):
    issue_map = {
        "delay": "latency",
        "cpu": "cpu",
        "mem": "mem",
        "disk": "latency",
        "loss": "latency"
    }

    dataframe = dataframe.loc[:, ~dataframe.columns.str.endswith("_latency-50")]
    dataframe = dataframe.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in dataframe.columns
            if c.endswith("_latency-90")
        }
    )

    normal_data = dataframe[dataframe["time"] < inject_time]
    anomaly_data = dataframe[dataframe["time"] >= inject_time]

    normal_data = normal_data.loc[:, normal_data.columns.str.endswith("_" + issue_map[issue])]
    anomaly_data = anomaly_data.loc[:, anomaly_data.columns.str.endswith("_" + issue_map[issue])]

    for df in (normal_data, anomaly_data):
        df.rename(
            columns={
                c: c.replace("_" + issue_map[issue], "")
                for c in df.columns
            },
            inplace=True
        )
    
    if "rabbitmq-exporter" in normal_data.columns:
        normal_data.drop(["rabbitmq-exporter"], axis=1, inplace=True)
        anomaly_data.drop(["rabbitmq-exporter"], axis=1, inplace=True)
    
    target_node = "front-end" #This seems to match the default in RCAEval

    if not use_all_anomaly_samples:
        target_scorer = ITAnomalyScorer(anomaly_scorer())
        target_scorer.fit(normal_data[target_node].to_numpy())

        #Find first index after anomaly injection where the target
        #node is detected as anomalous
        detection_index = None
        max_score_index = None
        max_score = 0
        
        for i, d in enumerate(anomaly_data[target_node]):
            score = target_scorer.score(np.array([d]))
            if score > max_score:
                max_score = score
                max_score_index = i
            if d > anomaly_threshold:
                detection_index = i
                break
        
        if detection_index is None:
            detection_index = max_score_index
        
        anomaly_data = anomaly_data.iloc[[detection_index]]

    call_graph = create_sock_shop_graph()
    causal_graph = call_graph.reverse()

    nodes_to_remove = []
    edges_to_add = []
    for node in causal_graph.nodes():
        if node not in normal_data.columns:
            children = list(causal_graph.successors(node))
            parents = list(causal_graph.predecessors(node))
            if len(children) > 1 and len(parents) > 0:
                raise ValueError(f"{node} is an unmeasured confounder!")
            else:
                nodes_to_remove.append(node)
                for p in parents:
                    for c in children:
                        edges_to_add.append((p, c))
    if len(nodes_to_remove) > 0:
        causal_graph.remove_nodes_from(nodes_to_remove)
        causal_graph.add_edges_from(edges_to_add)

    return {
        "graph": causal_graph,
        "training_sample": normal_data,
        "anomaly_sample": anomaly_data,
        "root_cause": root_cause,
        "target_node": target_node
    }

