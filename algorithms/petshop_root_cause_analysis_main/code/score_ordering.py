from typing import List, Callable, Set

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer

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


def make_score_ordering(
    anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
):
    """
    Score ordering method.

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
    ) -> List[str]:

        normal_data = pad_and_replace_nan(
            reduce_df(
                normal_metrics.copy(), metric=target_metric, statistic=target_statistic
            ),
            required_columns=graph.nodes,
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

        all_nodes = list(graph)
        scorers = {}
        scores = {}
        for n in all_nodes:
            scorers[n] = ITAnomalyScorer(anomaly_scorer())
            scorers[n].fit(normal_data[n].to_numpy())

            # Scoring the anomalous samples.
            scores[n] = (
                scorers[n].score(anomaly_data[n].iloc[0:1].to_numpy()[0]).flatten()[0]
            )

        results = [
            PotentialRootCause(node=node, metric=target_metric, score=scores[node])
            for node in all_nodes
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    return analyze_root_causes
