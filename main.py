# ./main.py
"""Main script for taking arguments from the command line and running experiments. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import numpy as np

from experiment_runner import run_experiments
from result_saver import save_results


def parse_config_from_args():
    parser = argparse.ArgumentParser(
        description="Root Cause Analysis Experiment Configuration"
    )

    parser.add_argument("--experiment-mode", type=str, choices=["vary_anomaly_strength", "vary_graph_size", "pro_rca", "sock_shop"], default="vary_graph_size",
                        help="Type of experiment: vary anomaly, graph size, use pro_rca benchmark or run on real-world Sock-shop 2 dataset.")
    
    parser.add_argument("--methods", type=str, default="score_ordering,smooth_traversal,traversal,cholesky,circa,counterfactual",
                        help="Comma separated list of methods to evaluate with options "
                        "being 'score_ordering', 'smooth_traversal', 'traversal', 'counterfactual', "
                        "'circa', 'cholesky', 'rcd' and 'epsilon_diagnosis'.")

    parser.add_argument("--n-observations-not-anomalous", type=int, default=1000,
                        help="Number of observations used for training (non-anomalous)")
    
    parser.add_argument("--n-observations-anomalous", type=int, default=1,
                        help="Number of observations in the anomalous period.")

    parser.add_argument("--anomaly-probability", type=float, default=0.05,
                        help="P-value threshold to consider a node as anomalous")

    parser.add_argument("--k", type=int, default=1,
                        help="Number of top-k root causes to evaluate")

    parser.add_argument("--number-trials", type=int, default=5,
                        help="How many graphs/data samples to try per experiment setting")

    parser.add_argument("--anomaly-values", type=str, default="2,3,11",
                        help='Anomaly strenghts: comma separated list with min,max,num as used in np.linspace, e.g., "2,3,11"')

    parser.add_argument("--fixed-anomaly-value", type=float, default=3.0,
                        help="Anomaly value to use when varying graph size")

    parser.add_argument("--number-of-nodes", type=str, default="20,100,5",
                        help='Number of nodes: comma separated list with min,max,num as used in np.linspace, e.g., "20,100,5"')

    parser.add_argument("--fixed-number-of-nodes", type=int, default=50,
                        help="Fixed graph size when varying anomaly strength")
    
    parser.add_argument("--graph-type", type=str, choices=["dag", "polytree", "collider_free_polytree"], default="dag",
                        help="What structural assumption to place on DAG generation")
    
    parser.add_argument("--adjust-for-ties", action=argparse.BooleanOptionalAction,
                        help="When computing top-k recalls, whether to account for (potential) ties in ranking produced by an algorithm.")

    parser.add_argument("--use-all-sock-shop-anomaly-samples", action=argparse.BooleanOptionalAction,
                        help="Flag to say to use all samples in the anomalous period (for RCD and epsilon-Diagnosis).")
    
    parser.add_argument("--sock-shop-data-path", type=str, default="./datasets/sock-shop-2/",
                        help="Path to the Sock-shop 2 data directory, if using experiment mode 'sock_shop'")

    parser.add_argument("--results-path", type=str, default="./results/results.npy",
                        help="Path to save the experiment results")

    args = parser.parse_args()

    # Parse comma-separated values
    anomaly_values_min, anomaly_values_max, anomaly_values_num = [float(x) for x in args.anomaly_values.split(",")]
    number_of_nodes_min, number_of_nodes_max, number_of_nodes_num = [float(x) for x in args.number_of_nodes.split(",")]

    config = {
        "experiment_mode": args.experiment_mode,
        "methods": [name.strip() for name in args.methods.split(',')],
        "n_observations_not_anomalous": args.n_observations_not_anomalous,
        "n_observations_anomalous": args.n_observations_anomalous,
        "anomaly_probability": args.anomaly_probability,
        "k": args.k,
        "number_trials": args.number_trials,
        "anomaly_values": np.linspace(anomaly_values_min, anomaly_values_max, num=int(anomaly_values_num)),
        "fixed_anomaly_value": args.fixed_anomaly_value,
        "number_of_nodes": np.linspace(number_of_nodes_min, number_of_nodes_max, num=int(number_of_nodes_num), dtype=int),
        "fixed_number_of_nodes": args.fixed_number_of_nodes,
        "graph_type": args.graph_type,
        "adjust_for_ties": args.adjust_for_ties,
        "use_all_sock_shop_anomaly_samples": args.use_all_sock_shop_anomaly_samples,
        "sock_shop_data_path": args.sock_shop_data_path,
        "results_path": args.results_path,
    }

    # Special config for pro_rca mode (hardcoded, still part of config)
    config["pro_rca_anomaly_list"] = [
        ('ExcessiveDiscount', 0.6, 'DISCOUNT', "Apparel"),
        #('COGs', 10 , 'UNIT_COST', 'Footwear'),
        ('FulfillmentSpike', 3, 'FULFILLMENT_COST', 'Beauty'),
        ('ReturnSurge', 10, 'RETURN_COST', 'Accessories'),
        ('ShippingDisruption', 5, 'SHIPPING_REVENUE', 'PersonalCare'),
    ]

    return config


def main():
    config = parse_config_from_args()
    results = run_experiments(config)
    save_results(results, config["results_path"])

if __name__ == "__main__":
    main()
