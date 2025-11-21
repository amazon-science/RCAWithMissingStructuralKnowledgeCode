# ./data_simulators/synthetic_data.py
"""Functions for generating random DAGs, SCMs, anomalies and sampling from the resulting models.
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard, Patrick Blöbaum.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0
"""

import random
from typing import List, Optional, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from dowhy.gcm.causal_mechanisms import StochasticModel, AdditiveNoiseModel
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.gcm.fitting_sampling import _parent_samples_of
from dowhy.gcm import InvertibleStructuralCausalModel, PredictionModel
from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm import RescaledMedianCDFQuantileScorer
from dowhy.gcm.stochastic_models import ScipyDistribution
from dowhy.gcm.util.general import shape_into_2d

from dowhy.graph import validate_acyclic, get_ordered_predecessors, is_root_node
from dowhy.gcm._noise import compute_data_from_noise

from algorithms.score_ordering import apply_score_ordering


class _GaussianMixtureDistribution(StochasticModel):
    """
        Gaussian mixture distribution object.
        NOTE: This only allows an univariate distribution!

        Class attributes:
        __means: A list of means of the Gaussians distributions.
        __std_values: A list of standard deviations of the Gaussians distributions.
        __parameters: A numpy array representing the means and standard deviations.
        __weights: Weights of the gaussian distributions. Here. they are uniform.
        __normalize: Indicates whether the output should be normalized by X / (max[means] - min[means])
    """

    def __init__(self, means: List[float], std_values: List[float], normalize: Optional[bool] = False) -> None:
        """
        Initializes the GaussianMixtureDistribution.

        :param means: The means of the Gaussians.
        :param std_values: The standard deviations of the Gaussians.
        :param normalize: True if the output should be normalized in terms of (output - min(means)) / (max(means) -
        min(means)). False if the output should not be normalized. Default: False
        :return: None
        """
        self.__means = None
        self.__std_values = None

        self.__normalize = normalize

        self._set_parameters(means, std_values)

    @property
    def means(self) -> List[float]:
        """
        :return: The means of the Gaussians.
        """
        return self.__means

    @property
    def std_values(self) -> List[float]:
        """
        :return: The standard deviations of the Gaussians.
        """
        return self.__std_values

    def _set_parameters(self, means: List[float], std_values: List[float]) -> None:
        """
        Sets the means and standard deviations of the Gaussians.

        :param means: The means of the Gaussians.
        :param std_values: The standard deviation of the Gaussians.
        :return: None
        """
        if len(means) < 2 or len(std_values) < 2:
            raise RuntimeError(
                'At least two means and standard deviations are needed! %d means and %d standard deviations ' \
                'were given.' % (len(means), len(std_values)))

        self.__means = means
        self.__std_values = std_values
        self._init_parameters()

    def fit(self,
            X: np.ndarray) -> None:
        pass

    def _init_parameters(self) -> None:
        """
        Initializes the parameters. Here, it zips together the means and standard deviations into tuples and
        initializes the weights of the Gaussians uniformly.

        :return: None
        """
        self.__parameters = np.array([list(x) for x in zip(self.__means, self.__std_values)])
        self.__weights = np.ones(self.__parameters.shape[0], dtype=np.float64) / float(self.__parameters.shape[0])

    def draw_samples(self,
                     num_samples: int) -> np.ndarray:
        """
        Randomly samples from the defined Gaussian mixture distribution.

        :param num_samples: The number of samples.
        :return: The generated samples.
        """
        mixture_ids = np.random.choice(self.__parameters.shape[0], size=num_samples, replace=True, p=self.__weights)

        result = np.fromiter((stats.norm.rvs(*(self.__parameters[i])) for i in mixture_ids), dtype=np.float64)

        if self.__normalize:
            result = 2 * (result - np.min(self.__means)) / (np.max(self.__means) - np.min(self.__means)) - 1

        return shape_into_2d(result)

    def clone(self):
        return _GaussianMixtureDistribution(self.__means, self.__std_values, self.__normalize)


class _SimpleFeedForwardNetwork(PredictionModel):
    """
        A simple feed forward network. This is mostly used for creating random MLPs.

        Class attributes:
        __init_data: Initial data for finding the minimum and maximum output value of the network in order to
        normalize it.
        __weights: Network weights for each layer.
        __min_val: Minimum value of the output. This is used to normalize the data on [-1, 1].
        __max_val: Maximum value of the output. This is used to normalize the data on [-1, 1].
    """

    def __init__(self, weights: Dict[int, np.ndarray]) -> None:
        self.__weights = weights
        self.__min_val = None
        self.__max_val = None

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Optional[Any]) -> None:
        """ Only needed to find the min and max output value for normalization. """
        tmp_output = self.predict(X, normalize=False)
        self.__min_val = np.min(tmp_output)
        self.__max_val = np.max(tmp_output)

    def clone(self):
        return _SimpleFeedForwardNetwork(self.__weights)

    def predict(self, X: np.ndarray, normalize: Optional[bool] = True) -> np.ndarray:
        if np.isclose(X.std(), 0):
            current_result = (X - X.mean())
        else:
            current_result = (X - X.mean()) / X.std()
        
        keys = list(self.__weights.keys())

        for q in range(len(keys) - 1):
            current_result = 1 / (1 + np.exp(-np.dot(current_result, self.__weights[keys[q]])))

        predictions = np.dot(current_result, self.__weights[keys[len(keys) - 1]]).squeeze()
        if normalize:
            return 2 * (predictions - self.__min_val) / (self.__max_val - self.__min_val) - 1
        else:
            return predictions


def _get_root_node_model() -> StochasticModel:
    """
    Returns a random distribution. These distributions can be:
    - A uniform distribution.
    - A Gaussian distribution.
    - A Gaussian mixture model.

    :return: A random distribution.
    """
    rand_val = np.random.randint(0, 3)

    if rand_val == 0:
        return ScipyDistribution(stats.norm, loc=0, scale=1)
    elif rand_val == 1:
        return ScipyDistribution(stats.uniform, loc=-1, scale=2)
    elif rand_val == 2:
        num_components = np.random.choice(4, 1)[0] + 2

        means = []
        std_vals = []

        for i in range(num_components):
            means.append(np.random.uniform(-1, 1))
            std_vals.append(0.5)

        means = np.array(means)
        means = means - np.mean(means)

        return _GaussianMixtureDistribution(means=means, std_values=std_vals, normalize=True)
    else:
        raise RuntimeError("Invalid index")


def _get_random_non_root_model(num_inputs: int, gaussian_noise_std: float = 0.1) -> AdditiveNoiseModel:
    """
    Returns a random function. The functions can be:
    - With 20% chance: A linear function with random coefficients on [-1, 1].
    - With 80% chance: A non-linear function generated by a random neural network with outputs on [-1, 1]

    :param num_inputs: All input variables of the function.
    :param gaussian_noise_std: Noise of the gaussian distribution for the additive noise models.
    :return: A random function.
    """
    rand_val = np.random.uniform(0, 1)

    probability_nonlinear = 0.8

    if rand_val < probability_nonlinear:
        layers = {0: np.random.uniform(-5, 5, (num_inputs, np.random.randint(2, 100)))}
        layers[1] = np.random.uniform(-5, 5, (layers[0].shape[1], np.random.randint(2, 100)))
        layers[2] = np.random.uniform(-5, 5, (layers[1].shape[1], 1))

        return AdditiveNoiseModel(_SimpleFeedForwardNetwork(weights=layers),
                                  noise_model=ScipyDistribution(stats.norm,
                                                                loc=0,
                                                                scale=gaussian_noise_std))

    elif rand_val >= probability_nonlinear:
        linear_reg = LinearRegression()
        linear_reg.coef_ = np.random.uniform(-1, 1, num_inputs)
        linear_reg.intercept_ = 0

        return AdditiveNoiseModel(SklearnRegressionModel(linear_reg),
                                  noise_model=ScipyDistribution(stats.norm,
                                                                loc=0,
                                                                scale=gaussian_noise_std))


def _sample_natural_number(init_mass: Optional[float] = 0.5) -> int:
    """
    Samples and returns a natural number in [1, ...]. The greater the natural number, the less likely. This is based on
    the initial probability mass for sampling '1'.

    :param init_mass: Initial probability mass. Default: 0.5
    :return: A randomly sampled natural number in [1, ...].
    """
    current_mass = init_mass
    probability = np.random.uniform(0, 1)
    k = 1

    is_searching = True

    while is_searching:
        if probability <= current_mass:
            return k
        else:
            k += 1
            current_mass += 1 / (k ** 2)


def assign_random_scms(graph: nx.DiGraph) -> InvertibleStructuralCausalModel:
    """ Assigns an SCM to each node in the graph
    
    :param init_mass: Initial probability mass. Default: 0.5
    :return: An intervetible structural causal model.
    """
    validate_acyclic(graph)

    invertible_scm = InvertibleStructuralCausalModel(graph)
    training_data = pd.DataFrame()

    for node in nx.topological_sort(invertible_scm.graph):
        if is_root_node(graph, node):
            random_stochastic_model = _get_root_node_model()
            invertible_scm.set_causal_mechanism(node, random_stochastic_model)

            training_data[node] = random_stochastic_model.draw_samples(1000).squeeze()
        else:
            parents = get_ordered_predecessors(invertible_scm.graph, node)
            causal_model = _get_random_non_root_model(len(parents))
            invertible_scm.set_causal_mechanism(node, causal_model)

            if isinstance(causal_model.prediction_model, _SimpleFeedForwardNetwork):
                # Only calling fit here to learn the normalization of the inputs. It does not train weights etc., i.e.,
                # the Y values doesn't matter.
                causal_model.prediction_model.fit(X=training_data[parents].to_numpy(), Y=np.zeros((1000, 1)))

            training_data[node] = causal_model.draw_samples(training_data[parents].to_numpy())

        # Update local hash
        invertible_scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(invertible_scm.graph, node)

    return invertible_scm


def generate_random_dag(num_roots: int, 
                        num_children: int,
                        graph_type: str = "dag") -> nx.DiGraph:
    """ Generates a random DAG structure.

    :param num_roots: Number of root nodes.
    :param num_children: Number of non-root nodes.
    :param graph_type: {"dag", "polytree", "collider_free_polytree"}, default = "dag"
        Structural assumption on DAG generation.
        - "dag": general DAGs
        - "polytree": DAGs whose underlying skeleton is an undirected tree
        - "collider_free_polytree": DAG where no node has more than one parent
    :return: A randomly generated DAG.
    """
    ALLOWED_GRAPH_TYPES = {"dag", "polytree", "collider_free_polytree"}

    if graph_type not in ALLOWED_GRAPH_TYPES:
        raise ValueError(f"Invalid method '{graph_type}'. "
                        f"Expected one of {sorted(ALLOWED_GRAPH_TYPES)}.")

    if graph_type == "collider_free_polytree":
        G = nx.random_labeled_rooted_tree(n = num_roots + num_children) #undirected tree
        graph = nx.bfs_tree(G, 0) #
        node_labels = [f"X{i + 1}" for i in range(num_roots + num_children)]
        np.random.shuffle(node_labels)
        rename_map = {old: new for old, new in zip(graph.nodes, node_labels)}
        return nx.relabel_nodes(graph, rename_map)

    graph = nx.DiGraph()

    for i in range(num_roots):
        new_root = 'X' + str(i)
        graph.add_node(new_root)

    for i in range(num_children):
        potential_parents = list(graph.nodes)
        
        new_child = 'X' + str(i + num_roots)
        graph.add_node(new_child)

        num_parents = min(_sample_natural_number(init_mass=0.6), len(potential_parents))

        chosen_parents = []
        for parent in np.random.permutation(potential_parents):
            if len(chosen_parents) >= num_parents:
                break
            graph.add_edge(parent, new_child)
            if graph_type == "polytree":
                skeleton = nx.Graph(graph.to_undirected())
                if nx.is_tree(skeleton):
                    chosen_parents.append(parent)
                else:
                    graph.remove_edge(parent, new_child)
            else:
                chosen_parents.append(parent)

    return graph


def generate_random_invertible_scm(num_roots: int, num_children: int, graph_type: str = "dag") -> InvertibleStructuralCausalModel:
    """ Generates a randomly generated invertible SCM with a random DAG structure and random functional causal models.

    :param num_roots: Number of root nodes.
    :param num_children: Number of non-root nodes.
    :return: A randomly generated invertible SCM.
    """
    return assign_random_scms(generate_random_dag(num_roots, num_children, graph_type))


def sample_noise(causal_model, num_samples):
    sorted_nodes = list(nx.topological_sort(causal_model.graph))

    drawn_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)
    drawn_noise_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)

    for node in sorted_nodes:
        if is_root_node(causal_model.graph, node):
            noise = causal_model.causal_mechanism(node).draw_samples(num_samples).reshape(-1)
            drawn_noise_samples[node] = noise.squeeze()
            drawn_samples[node] = noise.squeeze()
        else:
            noise = causal_model.causal_mechanism(node).draw_noise_samples(num_samples).reshape(-1)
            drawn_noise_samples[node] = noise.squeeze()
            drawn_samples[node] = (
                causal_model.causal_mechanism(node)
                .evaluate(_parent_samples_of(node, causal_model, drawn_samples), noise)
                .squeeze()
            )

    return drawn_samples, drawn_noise_samples


def generate_scm_and_data(config, anomaly_value, number_of_nodes):
    num_roots = int(number_of_nodes * random.uniform(0.2, 0.4))
    num_children = number_of_nodes - num_roots
    scm = generate_random_invertible_scm(num_roots, num_children, config["graph_type"])

    all_nodes = list(scm.graph.nodes)
    training_sample, _ = sample_noise(scm, config["n_observations_not_anomalous"])
    root_cause = random.choice(all_nodes)

    _, drawn_noise = sample_noise(scm, config["n_observations_anomalous"])
    drawn_noise[root_cause] += (
        np.sign(drawn_noise[root_cause]) *
        np.std(training_sample[root_cause]) *
        anomaly_value
    )
    anomaly_sample = compute_data_from_noise(scm, drawn_noise)
    
    # Getting the scores of all variables
    score_dict = {}
    for node in scm.graph.nodes:
        tmp_anomaly_scorer = RescaledMedianCDFQuantileScorer()
        tmp_anomaly_scorer.fit(training_sample[node].to_numpy())
        tmp_score = tmp_anomaly_scorer.score(anomaly_sample[node].to_numpy())
        score_dict[node] = tmp_score#.squeeze()

    descendants = list(nx.descendants(scm.graph, root_cause))
    
    #judge anomalousness based on first sample
    anomalous_nodes = [n for n in descendants + [root_cause]
                       if np.exp(-score_dict[n][0]) < config["anomaly_probability"]]

    if not anomalous_nodes:
        return None

    return {
        "graph": scm.graph,
        "training_sample": training_sample,
        "anomaly_sample": anomaly_sample,
        "root_cause": root_cause,
        "target_node": random.choice(anomalous_nodes),
    }
