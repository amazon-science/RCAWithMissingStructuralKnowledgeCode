# ./result_saver.py
""" Utility functions for saving output data from experiments. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import os
import numpy as np
from warnings import warn


def get_unique_filename(path):
    directory, base = os.path.split(path)
    name, ext = os.path.splitext(base)
    if not ext:
        ext = '.npy'  # Default extension if none is given
    i = 0
    while True:
        filename = f"{name}_{i}{ext}" if i > 0 else f"{name}{ext}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(get_unique_filename(path), results)


def load_results(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    warn(f"{path} not found, returning an empty dictionary.")
    return {}
