# ./utils.py
""" Basic utility functions. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import pandas as pd


def sort_dict_by_values(input_dict):
    sorted_keys = sorted(input_dict, key=input_dict.get)
    sorted_keys.reverse()
    return sorted_keys


def preprocess_data(df):
    df = df.copy()
    df.index.name = "time"
    df["metric"] = "latency"
    df = df.set_index("metric", append=True).unstack("metric")

    df["statistic"] = "Average"
    df = df.set_index("statistic", append=True).unstack("statistic")
    return df
