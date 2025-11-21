# ./data_simulators/prorca.py
"""Basic wrapper functions for initialising the causal graph and running the ProRCA package (https://github.com/profitopsai/ProRCA). 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import networkx as nx
import numpy as np
from typing import Tuple, Dict

import prorca
from data_generators.synthetic_sales_data import generate_fashion_data_with_brand, inject_anomalies_by_date


def create_prorca_marginalised_graph():
    """Creates a networkx graph based on Figure 3 of the ProRCA paper
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges as per the diagram
    edges = [
        ("PRICEEACH", "SALES"),
        ("PRICEEACH", "UNIT_COST"),
        ("QUANTITYORDERED", "SALES"),
        ("QUANTITYORDERED", "COST_OF_GOODS_SOLD"),
        ("QUANTITYORDERED", "SHIPPING_REVENUE"),
        ("QUANTITYORDERED", "FULFILLMENT_COST"),
        ("UNIT_COST", "COST_OF_GOODS_SOLD"),
        ("SALES", "SHIPPING_REVENUE"),
        ("SALES", "DISCOUNT"),
        ("SALES", "NET_SALES"),
        ("DISCOUNT", "NET_SALES"),
        ("NET_SALES", "PROFIT"),
        ("NET_SALES", "PROFIT_MARGIN"),
        ("NET_SALES", "RETURN_COST"),
        ("COST_OF_GOODS_SOLD", "PROFIT"),
        ("FULFILLMENT_COST", "PROFIT"),
        ("SHIPPING_REVENUE", "PROFIT"),
        ("RETURN_COST", "PROFIT"),
        ("MARKETING_COST", "PROFIT"),
        ("PROFIT", "PROFIT_MARGIN"),
    ]

    # Add edges to the graph

    G.add_edges_from(edges)

    return G


def pro_rca_data_generator(anomaly_tuple: Tuple[str, float, str]) -> Dict:
    """ Creates a dataset based on the ProRCA package
    """
    type_of_anomaly, anomaly_size, _, merchandise_hierarchy = anomaly_tuple

    DATE = '2023-01-07'
    # Generate the data
    df = generate_fashion_data_with_brand(start_date="2023-01-01", end_date="2023-01-10")

    # Inject anomaly
    anomaly_schedule = {
        DATE: (type_of_anomaly, anomaly_size, 'AFFECTED_VARIABLE', merchandise_hierarchy)
    }
    df_anomalous = inject_anomalies_by_date(df, anomaly_schedule)

    relevant_variables = ['QUANTITYORDERED','PRICEEACH','UNIT_COST','SALES','DISCOUNT','NET_SALES',
                      'FULFILLMENT_COST','MARKETING_COST','RETURN_COST','COST_OF_GOODS_SOLD',
                      'SHIPPING_REVENUE','PROFIT','PROFIT_MARGIN']
    
    anomaly_type_to_rootcause_dictionary = {
        'ExcessiveDiscount': 'DISCOUNT',
        'COGs': 'UNIT_COST',
        'FulfillmentSpike': 'FULFILLMENT_COST',
        'ReturnSurge': 'RETURN_COST',
        'ShippingDisruption': 'SHIPPING_REVENUE'
    }

    return {
        "graph": create_prorca_marginalised_graph(),
        "training_sample": df[relevant_variables].loc[np.logical_and(df["MERCHANDISE_HIERARCHY"].apply(lambda x: merchandise_hierarchy in x), 
                                                     df["ORDERDATE"]<DATE)],
        "anomaly_sample": df_anomalous[relevant_variables].loc[np.logical_and(df_anomalous["MERCHANDISE_HIERARCHY"].apply(lambda x: merchandise_hierarchy in x), 
                                                     df_anomalous["ORDERDATE"]==DATE)][:1],
        "root_cause": anomaly_type_to_rootcause_dictionary[type_of_anomaly],
        "target_node": "PROFIT"
    }