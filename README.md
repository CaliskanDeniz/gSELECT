# gSELECT: Gene Selection for Single-Cell Data

**gSELECT** is a Python package designed for gene selection in single-cell RNA sequencing (scRNA-seq) data. 
It provides efficient methods for **feature selection**, **classification**, **data handling**, and **visualization** to facilitate downstream analysis.

---

## Features

- **Feature Selection**: Computes mutual information to identify informative genes.
- **Gene Classification**: Implements a neural network classifier (MLP) to assess gene selection strategies.
- **Data Loading & Exploration**: Handles `.h5ad` and CSV-based single-cell datasets.
- **Visualization**: Plots classification performance and misclassification rates.

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
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `polars`
- `seaborn`
- `scipy`
- `principal_feature_analysis`
---

## Usage

### **1. Loading Data**

Load an `.h5ad` single-cell dataset and filter based on metadata:

```python
from gSELECT.io import load_h5ad

gene_names, expression_data = load_h5ad("data/dataset.h5ad", filter_column="cell_type", filter_values=["type1", "type2"])
```

---

### **2. Feature Selection**

Compute mutual information to rank genes:

```python
from gSELECT.feature_selection import compute_mutual_information

gene_mutual_info = compute_mutual_information(gene_names, expression_data, top_mutual_information=50)
```

---

### **3. Gene Classification**

Run classification using selected genes:

```python
from gSELECT.classification import run_selected_genes

results = run_selected_genes(expression_data, gene_mutual_info, number_sweeps=10, top_n_genes=5)
```

Run classification with all genes:

```python
from gSELECT.classification import run_all_genes

results = run_all_genes(expression_data, gene_mutual_info, number_sweeps=10)
```

---

### **4. Visualization**

Plot classification results:

```python
from gSELECT.visualization import plot_results

plot_results(results)
```

---

## Example Notebook

An example Jupyter notebook demonstrating end-to-end usage is available under `examples/example_notebook.ipynb`.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! Please create a pull request or open an issue for feature requests or bug reports.
