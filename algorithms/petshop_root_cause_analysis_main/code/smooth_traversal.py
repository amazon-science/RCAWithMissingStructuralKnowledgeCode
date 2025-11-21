from typing import List, Callable, Set, Any

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.graph import node_connected_subgraph_view, is_root_node

from .rca_task import PotentialRootCause
from .data_preprocessing import reduce_df


def pad_and_fill(data_matrix: pd.DataFrame, fill_df: pd.DataFrame):
    original_columns = data_matrix.columns
    overall_mean = np.nanmean(data_matrix.mean())
    for c in fill_df.columns:
        if c not in data_matrix.columns:
            data_matrix[c] = fill_df[c].mean()
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    data_matrix.fillna(overall_mean, inplace=True)
    return data_matrix, original_columns


def pad_and_replace_nan(data_matrix: pd.DataFrame, required_columns: Set[str]):
    # TODO: Cleanup
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    overall_mean = np.nanmean(data_matrix.mean())
    data_matrix.fillna(overall_mean, inplace=True)
    for c in set(required_columns) - set(data_matrix.columns):
        data_matrix[c] = overall_mean
    return data_matrix


def make_smooth_traversal(
    anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
):
    """
    Smooth traversal method.

    Paper:

    Args:
        anomaly_scorer: DoWhy Anomaly Scorer. Default: RescaledMedianCDFQuantileScorer
    """

    def analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[Any]:

        causal_graph = graph.reverse()
        traversal_sub_graph = node_connected_subgraph_view(causal_graph, target_node)

        normal_data = pad_and_replace_nan(
            reduce_df(
                normal_metrics.copy(), metric=target_metric, statistic=target_statistic
            ),
            required_columns=causal_graph.nodes,
        )
        anomaly_data, original_abnormal_columns = pad_and_fill(
            reduce_df(
                abnormal_metrics.copy(),
                metric=target_metric,
                statistic=target_statistic,
            ),
            fill_df=normal_data,
        )
        anomaly_data = anomaly_data.iloc[2:3]

        if anomaly_data.shape[0] > 1:
            raise ValueError("Currently only support a single anomaly sample!")

        all_scores = {}

        for node in traversal_sub_graph.nodes:
            tmp_anomaly_scorer = ITAnomalyScorer(anomaly_scorer())
            tmp_anomaly_scorer.fit(normal_data[node].to_numpy())
            tmp_score = tmp_anomaly_scorer.score(anomaly_data[node].to_numpy())

            all_scores[node] = tmp_score.squeeze()

        final_scores_dict = {}

        for node in traversal_sub_graph.nodes:
            if is_root_node(traversal_sub_graph, node):
                final_score = all_scores[node]
            else:
                final_score = np.min(
                    [
                        (all_scores[node] - all_scores[parent])
                        for parent in traversal_sub_graph.predecessors(node)
                    ]
                )
            final_scores_dict[node] = max(0, final_score.flatten()[0])

        results = [
            PotentialRootCause(
                node=node, metric=target_metric, score=final_scores_dict[node]
            )
            for node in final_scores_dict
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    return analyze_root_causes
