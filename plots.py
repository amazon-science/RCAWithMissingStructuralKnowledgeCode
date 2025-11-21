# ./plots.py
"""Plotting functions for displaying output data from experiments. 
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure LaTeX fonts
# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin/'
# context_dict = {
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.size': 13,
#     'lines.markersize': 10
# }

# Define plotting utilities
def plot_line(x, y_dict, xlabel, ylabel, output_path, legend_loc=(0.5, 0.45)):
    # with mpl.rc_context(context_dict):
    for label, (y_vals, marker) in y_dict.items():
        plt.plot(x, y_vals, label=label, marker=marker)
    plt.xlabel(xlabel, fontdict={"size": 18})
    plt.ylabel(ylabel, fontdict={"size": 18})
    plt.legend(frameon=False, ncol=2, bbox_to_anchor=legend_loc, loc="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_runtime_boxplot(times, labels, output_path):
    # with mpl.rc_context(context_dict):
    plt.boxplot(times, labels=labels)
    plt.ylabel("Runtime (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Define common parameters
methods = {
    "SCORE ORDERING": ("score_ordering_correct", "^", "score_ordering_time"),
    "Traversal": ("traversal_correct", "v", "traversal_time"),
    "Cholesky": ("cholesky_correct", "<", "cholesky_time"),
    "SMOOTH TRAVERSAL": ("smooth_traversal_correct", ">", "smooth_traversal_time"),
    "Counterfactual": ("counterfactual_contribution_correct", "1", "counterfactual_contribution_time"),
    "Circa": ("circa_correct", "2", "circa_time")
}
