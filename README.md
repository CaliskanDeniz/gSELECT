# gSELECT: Gene Selection for Single-Cell Data

**gSELECT** is a Python package for gene selection and classification in single-cell RNA sequencing (scRNA-seq) data.  
It provides efficient methods for **feature selection**, **classification**, **data handling**, and **visualization** to facilitate downstream analysis.

---

## Features

- **Feature Selection**: Compute mutual information to identify informative genes, with flexible options for exclusion, custom gene lists, and output control.
- **Gene Classification**: Multiple entry points for classification using selected genes, custom gene sets, exhaustive or greedy subset search, and all non-constant genes.
- **Data Loading & Exploration**: Load and filter `.h5ad` and CSV-based single-cell datasets; explore dataset structure and metadata.
- **Visualization**: Publication-ready plots for classification performance, misclassification rates, and gene subset rankings.
- **Utilities**: Memory usage logging, unique file naming, and more.

---

## Installation

```sh
pip install gSELECT
```

or for development:

```sh
git clone https://github.com/caliskandeniz/gSELECT.git
cd gSELECT
pip install -e .
```

---

## Dependencies

- `scanpy`
- `anndata`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `polars`
- `scipy`
- `psutil`
- `principal_feature_analysis`
- `tqdm`
- `openpyxl`
- `pyarrow`

---

## Usage

### 1. Data Loading & Exploration

**Explore an `.h5ad` dataset:**

```python
from gSELECT.io import explore_h5ad

explore_h5ad("data/dataset.h5ad")
```

**Load and filter an `.h5ad` dataset:**

```python
from gSELECT.io import load_h5ad

gene_names, expression_data = load_h5ad(
    "data/dataset.h5ad",
    filter_column="cell_type",
    filter_values=["type1", "type2"]
)
```

**Load from CSV:**

```python
from gSELECT.io import load

gene_names, expression_data = load("data/dataset.csv")
```

---

### 2. Feature Selection

**Compute mutual information scores with all options:**

```python
from gSELECT.feature_selection import compute_mutual_information

gene_mutual_info = compute_mutual_information(
    gene_names,
    expression_data,
    gene_list=None,                  # Optional: restrict to specific genes
    top_mutual_information=50,       # Number of top genes to retain
    min_datapoints=10,               # Minimum samples per gene
    basis_log=2,                     # Logarithm base for entropy
    exclusion_list=["ACTB", "GAPDH"],# Exclude specific genes
    should_save_mutual_info=True,    # Save results to CSV
    output_folder="output"           # Output directory
)
```

---

### 3. Gene Classification

**Available entry points:**

| Function | Purpose | Input | Output |
|----------|--------|-------|--------|
| `run_selected_genes` | Classify using top N genes | data, MI, N | results list |
| `run_multiple_gene_selections` | Compare multiple panel sizes | data, MI, list of N | results dict |
| `run_with_custom_gene_set` | Use user-defined gene list | data, gene names, MI | results list |
| `run_explorative_gene_selections` | Exhaustive/greedy search of top N | data, MI, N | results dict |
| `run_explorative_gene_selections_with_custom_set` | Exhaustive search of custom set | data, gene names, MI | results dict |
| `run_all_genes` | Use all non-constant genes | data, MI | results list |

**Example usage:**

```python
from gSELECT.classification import (
    run_selected_genes,
    run_multiple_gene_selections,
    run_with_custom_gene_set,
    run_explorative_gene_selections,
    run_explorative_gene_selections_with_custom_set,
    run_all_genes
)

# Top N genes
results = run_selected_genes(expression_data, gene_mutual_info, test_data=test_data, number_sweeps=10, top_n_genes=5)

# Multiple panel sizes
results_dict = run_multiple_gene_selections(expression_data, gene_mutual_info, test_data=test_data, number_sweeps=10, gene_selection=[1, 2, 5, 10, 100])

# Custom gene set
results = run_with_custom_gene_set(expression_data, ["GeneA", "GeneB"], gene_mutual_info, test_data=test_data, number_sweeps=10)

# Exhaustive/greedy search of top N
results_dict = run_explorative_gene_selections(expression_data, gene_mutual_info, test_data=test_data, number_sweeps=2, top_n_genes=11)

# Exhaustive search of custom set
results_dict = run_explorative_gene_selections_with_custom_set(expression_data, ["GeneA", "GeneB"], gene_mutual_info, test_data=test_data, number_sweeps=2)

# All non-constant genes
results = run_all_genes(expression_data, gene_mutual_info, test_data=test_data, number_sweeps=10)
```

---

### 4. Visualization

**Available entry points:**

| Function | Purpose | Input | Output |
|----------|--------|-------|--------|
| `plot_results` | Compare strategies | results list | PNG, CSV |
| `plot_multiple_gene_selections` | Panel size sweep | results dict | PNG, CSV |
| `plot_explorative_gene_selections` | Subset ranking | results dict | PNG, CSV |
| `plot_all_genes` | All strategies | results list | PNG, CSV |

**Example usage:**

```python
from gSELECT.visualization import (
    plot_results,
    plot_multiple_gene_selections,
    plot_explorative_gene_selections,
    plot_all_genes
)

plot_results(results, output_folder="output")
plot_multiple_gene_selections(results_dict, output_folder="output")
plot_explorative_gene_selections(results_dict, output_folder="output")
plot_all_genes(results, output_folder="output")
```

---

### 5. Utilities

- `get_memory_usage_mb()`: Returns current memory usage in MB.
- `log_peak_memory_usage()`: Logs peak RAM usage.
- `save_dataframe_to_csv()`, `save_figure_png()`: Save results and figures with unique filenames.

---

## Example Notebook

A comprehensive Jupyter notebook demonstrating end-to-end usage is available under `examples/example_notebook.ipynb`.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! Please create a pull request or open an issue for feature requests or bug reports.