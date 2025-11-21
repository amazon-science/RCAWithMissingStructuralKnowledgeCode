# Root Cause Analysis of Outliers with Missing Structural Knowledge in Python

This project includes the implementations of the experiments in the paper 

`Orchard, W. R.*, Okati, N.*, Garrido Mejia, S.H., Blöbaum, P. and Janzing, D. (2025) Root Cause Analysis of Outliers with Missing Structural Knowledge. Accepted NeurIPS 2025.`

It includes experiment runners, evaluation utilities, result saving, and plotting scripts for comparing algorithms under different experimental setups.

---

## 📌 Features

- **Synthetic data generation** via random SCM generation followed by anomaly injection.
- **Multiple experiment modes**:
  - `vary_graph_size`: Fix anomaly strength, vary graph size.
  - `vary_anomaly_strength`: Fix graph size, vary anomaly strength.
  - `pro_rca`: Run domain-specific RCA experiments
  - `sock_shop`: Run methods on semi-synthetic Sock-shop 2 dataset (Pham et al. 2024, Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?, https://dl.acm.org/doi/10.1145/3691620.3695065)
- **Evaluation** of multiple algorithms:
  - IT anomaly score ordering (this paper)
  - Smooth traversal (this paper)
  - Traversal (e.g. Liu et al. 2021, Microhecl: high-efficient root cause localization in large-scale microservice systems.)
  - Cholesky-based methods (Li et al. 2024, Root cause discovery via permutations and cholesky decomposition)
  - Counterfactual attribution (Budhathoki et al. 2022, Causal structure-based root cause analysis of outliers)
  - CIRCA (Li et al. 2022, Causal inference-based root cause analysis for online service systems with intervention recognition)
  - RCD (Ikram et al. 2022, Root Cause Analysis of Failures in Microservices
through Causal Discovery)
  - ε-Diagnosis (Shan et al. 2019, ε-Diagnosis: Unsupervised and Real-time Diagnosis of Small- window Long-tail Latency in Large-scale Microservice Platforms)
- **Plotting scripts** to visualize accuracy vs anomaly, accuracy vs graph size, and runtime comparisons.

---

## ⚙️ Installation

Create a conda environment:

```bash
conda create -n rca-missing-knowledge python=3.10
conda activate rca-missing-knowledge
```

clone the repository and install dependencies:

```bash
git clone git@github.com:amazon-science/RCAWithMissingStructuralKnowledgeCode.git
cd RCAWithMissingStructuralKnowledgeCode
pip install -r requirements.txt
```

Install PyRCA dependency by cloning the repository and installing:
```bash
git clone git@github.com:salesforce/PyRCA.git
cd PyRCA
pip install .
```

If one encounters the following ValueError when running main.py (see below):
```bash
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

You must force reinstall numpy and sklearn

```bash
conda install --force-reinstall numpy scikit-learn
```

---

## 🚀 Usage

Run experiments via the command line:

```bash
python main.py --experiment-mode "vary_anomaly_strength" --n-observations-not-anomalous 1000 --number-trials 10 --fixed-number-of-nodes 50 --anomaly-values "2,3,11" --results-path "./results/vary_anomaly_strength_results.npy"
```

### 🔧 Primary command-line arguments

- `--experiment-mode`: Which type of experiment to run. Options:
  - `"vary_graph_size"`
  - `"vary_anomaly_strength"`
  - `"pro_rca"`
  - `"sock_shop"`
- `--methods`: Comma separated list of which methods to evaluate.
- `--n-observation-not-anomalous`: Number of observations used for training (non-anomalous).
- `--anomaly-probability`: P-value threshold to consider a node as anomalous.
- `--k`: Number of top-k root causes to evaluate.
- `--number-trials`: How many random graphs to generate per setting.
- `--anomaly-values`: Anomaly strenghts: comma separated list with min,max,num as used in np.linspace, e.g., "2,3,11".
- `--fixed-anomaly-value`: Fixed anomaly strength for graph size experiments.
- `--number-of-nodes`: Number of nodes: comma separated list with min,max,num as used in np.linspace, e.g., "20,100,5"
- `--fixed-number-of-nodes`: Fixed graph size when varying anomaly strength.
- `--graph-type`: Structural assumption on DAG generation. Either `"dag"`, `"polytree"` or `"collider_free_polytree"`
- `--adjust-for-ties`: Whether to account for potential ranking ties when evaluating the top-k recall of each method.
- `--results-path`: Path to save results (`.npy`).

---

## 📊 Plotting Results

After experiments are saved in `./results/`, generate plots by working through `plot_generation.ipynb`, making sure to change the relevant results file paths according to how you specified them when generating your results.

This creates:

- **Accuracy vs anomaly strength**
- **Accuracy vs graph size**
- **Runtime comparisons (boxplots)**

Saved in `./results/` as `.pdf` files.

---

## 🧪 Example Workflow

1. Running graph size experiments:

```bash
python main.py --experiment-mode "vary_graph_size" --fixed-anomaly-value 3.0 --number-of-nodes "20,100,5" --results-path "./results/vary_graph_size_results.npy"
```

2. Running anomaly size experiments:

```bash
python main.py --experiment-mode "vary_anomaly_strength" --fixed-number-of-nodes 50 --anomaly-values "2,3,11" --results-path "./results/vary_anomaly_strength_results.npy"
```

3. Running ProRCA experiments:

```bash
python main.py --experiment-mode "pro_rca"
```

4. Running Sock-shop experiments:

    (a) If you have not downloaded the Sock-shop 2 dataset then first you must run `download_sock_shop.py` to save it to `./datasets/sock-shop-2/`:
    ```bash
    python download_sock_shop.py 
    ```
    (b)
    ```bash
    python main.py --experiment-mode "sock_shop"
    ```

5. Plotting, run each cell in `plot_generation.ipynb` according to which experiments you have run, making sure to change any file paths to match those you provided when running the experiments.

6. To run PetShop experiments we have provided all the necessary code in `./algorithms/petshop_root_cause_analysis_main/code/`,
which can be run according to the instructions given by the original PetShop repository (https://github.com/amazon-science/petshop-root-cause-analysis) using our provided `run_experiments.*` files.

---

## Reference
If you use this code in your own research, please cite our paper

Orchard, W. R.\*, Okati, N.\*, Garrido Mejia, S.H., Blöbaum, P. and Janzing, D. (2025) Root Cause Analysis of Outliers with Missing Structural Knowledge. Accepted NeurIPS 2025.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

