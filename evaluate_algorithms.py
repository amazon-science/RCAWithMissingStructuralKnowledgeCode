# ./evaluate_algorithms.py
"""Functions for collecting output data from each algorithm and evaluating its performance. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import time
from typing import Dict, Any

from dowhy.gcm import RescaledMedianCDFQuantileScorer

from algorithms import (
    apply_cholesky,
    apply_circa,
    apply_counterfactual_contribution,
    apply_score_ordering,
    apply_simple_traversal,
    apply_smooth_traversal,
    apply_epsilon_diagnosis,
    apply_rcd
)


def is_in_top_k(scores: dict[str, float], root_cause: str, k: int, adjust_for_ties: bool = False) -> float:
    """
    Returns the probability that `root_cause` is among the top-k highest scoring varibales.

    Parameters
    ----------
    scores : dict[str, float]
        Mapping from variable to score given by an algorithm being evaluated. Higher score is "better".
    root_cause : str
        Ground truth root cause variable.
    k : int
        The number of top score groups to include.
    adjust_for_ties : bool, default=False
        If False: returns 1.0 if root_cause is in the first k distinct score "blocks", else 0.0.
        If True: returns the probability that the root_cause falls in the top-k highest scoring
                 variables when one selects among tied variables at random.
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    unique_scores = sorted(set(scores.values()), reverse=True)

    # Handle case where the algorithm doesn't score the root cause at all (e.g. in traversal if it has an anomalous parent)
    if root_cause not in scores:
        return 0.0
    
    # --- Case 1: No tie adjustment ---
    if not adjust_for_ties:
        root_cause_score = scores[root_cause]
        score_rank = unique_scores.index(root_cause_score) + 1  # 1-based
        return 1.0 if score_rank <= k else 0.0

    # --- Case 2: Tie-adjusted version ---
    rank_start = 0
    for score in unique_scores:
        tied_nodes = [node for node, val in sorted_items if val == score]
        tie_size = len(tied_nodes)
        rank_end = rank_start + tie_size  # exclusive

        if root_cause in tied_nodes:
            if rank_start >= k:
                return 0.0
            elif rank_end <= k:
                return 1.0
            else:
                overlap = k - rank_start
                return max(0.0, min(1.0, overlap / tie_size))
        rank_start = rank_end

    return 0.0


def evaluate_score_ordering(apply_score_ordering, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_score_ordering(
        graph=graph,
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
        anomaly_scorer=RescaledMedianCDFQuantileScorer,
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)
    
    return is_correct, final_time - initial_time


def evaluate_traversal(apply_simple_traversal, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_simple_traversal(
        graph=graph,
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
        anomaly_scorer=RescaledMedianCDFQuantileScorer,
        anomaly_threshold=3.0,
        debug=False
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_smooth_traversal(apply_algorithm_smooth_traverse, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_algorithm_smooth_traverse(
        graph=graph,
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
        anomaly_scorer=RescaledMedianCDFQuantileScorer,
        debug=False
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_cholesky(apply_cholesky, experiment_data, k, adjust_for_ties, cholesky_type: str="highdim"):
    initial_time = time.time()
    causes = apply_cholesky(
        normal_data = experiment_data["training_sample"],
        anomaly_data = experiment_data["anomaly_sample"],
        cholesky_type = cholesky_type,
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_counterfactual_contribution(apply_counterfactual_contribution, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_counterfactual_contribution(
        graph=graph,
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)
    
    return is_correct, final_time - initial_time


def evaluate_circa(apply_circa, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_circa(
        graph=graph.reverse(),
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_rcd(apply_rcd, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_rcd(
        graph=graph.reverse(),
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_epsilon_diagnosis(apply_epsilon_diagnosis, graph, experiment_data, k, adjust_for_ties):
    initial_time = time.time()
    causes = apply_epsilon_diagnosis(
        graph=graph.reverse(),
        target_node=experiment_data["target_node"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
    )
    final_time = time.time()
    is_correct = is_in_top_k(causes, experiment_data["root_cause"], k, adjust_for_ties)

    return is_correct, final_time - initial_time


def evaluate_algorithms(
    experiment_data,
    methods: list,
    k: int = 1,
    adjust_for_ties: bool = False
) -> Dict[str, Any]:
    """
    Evaluate multiple algorithms and return a dictionary containing their correctness and timing.
    """
    graph = experiment_data["graph"]
    results = {}

    # IT Anomaly Score
    if "score_ordering" in methods:
        score_ordering_correct, score_ordering_time = evaluate_score_ordering(
            apply_score_ordering, graph, experiment_data, k, adjust_for_ties
        )
        results["score_ordering_correct"] = score_ordering_correct
        results["score_ordering_time"] = score_ordering_time

    # Simple Traversal
    if "traversal" in methods:
        traversal_correct, traversal_time = evaluate_traversal(
            apply_simple_traversal, graph, experiment_data, k, adjust_for_ties
        )
        results["traversal_correct"] = traversal_correct
        results["traversal_time"] = traversal_time

    # Smooth Traversal
    if "smooth_traversal" in methods:
        smooth_correct, smooth_time = evaluate_smooth_traversal(
            apply_smooth_traversal, graph, experiment_data, k, adjust_for_ties
        )
        results["smooth_traversal_correct"] = smooth_correct
        results["smooth_traversal_time"] = smooth_time

    # Cholesky-Based RCA
    if "cholesky" in methods:
        cholesky_correct, cholesky_time = evaluate_cholesky(
            apply_cholesky, experiment_data, k, adjust_for_ties)
        results["cholesky_correct"] = cholesky_correct
        results["cholesky_time"] = cholesky_time

    # Counterfactual contribution
    if "counterfactual" in methods:
        cf_correct, cf_time = evaluate_counterfactual_contribution(
            apply_counterfactual_contribution, graph, experiment_data, k, adjust_for_ties
        )
        results["counterfactual_contribution_correct"] = cf_correct
        results["counterfactual_contribution_time"] = cf_time

    # CIRCA
    if "circa" in methods:
        circa_correct, circa_time = evaluate_circa(
            apply_circa, graph, experiment_data, k, adjust_for_ties
        )
        results["circa_correct"] = circa_correct
        results["circa_time"] = circa_time
    
    # RCD
    if "rcd" in methods:
        rcd_correct, rcd_time = evaluate_rcd(
            apply_rcd, graph, experiment_data, k, adjust_for_ties
        )
        results["rcd_correct"] = rcd_correct
        results["rcd_time"] = rcd_time
    
    # epsilon-Diagnosis
    if "epsilon_diagnosis" in methods:
        epsilon_diagnosis_correct, epsilon_diagnosis_time = evaluate_epsilon_diagnosis(
            apply_epsilon_diagnosis, graph, experiment_data, k, adjust_for_ties
        )
        results["epsilon_diagnosis_correct"] = epsilon_diagnosis_correct
        results["epsilon_diagnosis_time"] = epsilon_diagnosis_time

    return results
