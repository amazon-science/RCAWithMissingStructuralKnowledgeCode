# ./experiment_runner.py
"""Main functions for generating synthetic data and running the experiments. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import os
from tqdm import tqdm
import pandas as pd

from dowhy.gcm.util.general import set_random_seed

from evaluate_algorithms import evaluate_algorithms
from result_saver import load_results
from data_simulators.prorca import pro_rca_data_generator
from data_simulators.synthetic_data import generate_scm_and_data
from data_simulators.sock_shop import preprocess_sock_shop


def run_experiments(config):
    results = load_results(config["results_path"])
    
    if not "config" in results:
        results["config"] = config

    if config["experiment_mode"] == "vary_anomaly_strength":
        parameter_list = config["anomaly_values"]
        partial_generate_scm_and_data = lambda x: generate_scm_and_data(config, anomaly_value=x, number_of_nodes=config["fixed_number_of_nodes"])
    elif config["experiment_mode"] == "vary_graph_size":
        parameter_list = config["number_of_nodes"]
        partial_generate_scm_and_data = lambda x: generate_scm_and_data(config, anomaly_value=config["fixed_anomaly_value"], number_of_nodes=x)
    elif config["experiment_mode"] == "pro_rca":
        parameter_list = config["pro_rca_anomaly_list"]
        partial_generate_scm_and_data = pro_rca_data_generator
    elif config["experiment_mode"] == "sock_shop":
        # Check sock-shop data has been downloaded
        if not os.path.exists(config["sock_shop_data_path"]):
            raise FileNotFoundError(
                f"Base path not found: {config['sock_shop_data_path']}\n"
                "If you have not downloaded the sock-shop dataset, you will need\n"
                "to run the 'download_sock_shop.py' script first."
            )
        parameter_list = ["cpu", "delay", "disk", "loss", "mem"]
        sock_shop_path = config['sock_shop_data_path']

    for s, parameter in enumerate(parameter_list):
        print(f"Parameter = {parameter}")
        results[parameter] = initialise_result_storage(config)

        if config["experiment_mode"] == "sock_shop":
            # List only directories ending with the current issue
            issue_dirs = [
                d for d in os.listdir(sock_shop_path)
                if d.endswith(f"_{parameter}") and os.path.isdir(os.path.join(sock_shop_path, d))
            ]

            for service_dir in issue_dirs:
                service_path = os.path.join(sock_shop_path, service_dir)

                # Iterate through each replicate (1 to 5)
                for replicate in range(1, 6):
                    replicate_path = os.path.join(service_path, str(replicate))
                    csv_path = os.path.join(replicate_path, "simple_data.csv")

                    if os.path.exists(csv_path):
                        raw_data = pd.read_csv(csv_path)
                        print(f"Loaded {csv_path}")
                    else:
                        print(f"Missing file: {csv_path}")
                    
                    with open(os.path.join(replicate_path, "inject_time.txt")) as f:
                        inject_time = int(f.readlines()[0].strip())
                    
                    root_cause = os.path.basename(service_path.rstrip("/")).rsplit("_", 1)[0]
                    experiment_data = preprocess_sock_shop(
                        raw_data,
                        root_cause,
                        parameter,
                        inject_time,
                        use_all_anomaly_samples=config["use_all_sock_shop_anomaly_samples"]
                    )
                    
                    result_metrics = evaluate_algorithms(experiment_data,
                                                    methods=config["methods"],
                                                    k=config["k"],
                                                    adjust_for_ties=config["adjust_for_ties"])
                
                    update_results(results[parameter], result_metrics)
        else:
            i = 0
            seed = 0
            with tqdm(total=config["number_trials"]) as progress_bar:
                while i != config["number_trials"]:
                    set_random_seed(s * config["number_trials"] + seed)
                    seed += 1
                    
                    experiment_data = partial_generate_scm_and_data(parameter)
                    if experiment_data is None:
                        continue
                    i += 1

                    result_metrics = evaluate_algorithms(experiment_data,
                                                        methods=config["methods"],
                                                        k=config["k"],
                                                        adjust_for_ties=config["adjust_for_ties"])
                    
                    update_results(results[parameter], result_metrics)
                    progress_bar.update(1)

    return results


def initialise_result_storage(config):
    methods = config["methods"]
    result = {}

    if "score_ordering" in methods:
        result["score_ordering_time"] = []
        result["score_ordering_correct"] = 0
    if "smooth_traversal" in methods:
        result["smooth_traversal_time"] = []
        result["smooth_traversal_correct"] = 0
    if "counterfactual" in methods:
        result["counterfactual_contribution_time"] = []
        result["counterfactual_contribution_correct"] = 0
    if "circa" in methods:
        result["circa_time"] = []
        result["circa_correct"] = 0
    if "cholesky" in methods:
        result["cholesky_time"] = []
        result["cholesky_correct"] = 0
    if "traversal" in methods:
        result["traversal_time"] = []
        result["traversal_correct"] = 0
    if "rcd" in methods:
        result["rcd_time"] = []
        result["rcd_correct"] = 0
    if "epsilon_diagnosis" in methods:
        result["epsilon_diagnosis_time"] = []
        result["epsilon_diagnosis_correct"] = 0      

    return result


def update_results(results, metrics):
    for key in metrics:
        if "time" in key: # If time, append it to a list
            results[key].append(metrics[key])
        else: # If it is not a list, then it is the number of correct runs.
            results[key] += metrics[key]

