# gSELECT: Gene Selection and Classification Toolkit for Single-Cell RNA-seq

**gSELECT** is a Python package for efficient gene selection and classification tailored for single-cell RNA sequencing (scRNA-seq) data.  
It streamlines feature selection, model evaluation, and performance visualization - from raw `.h5ad` or `.csv` data to publication-ready plots.

---

## Installation

Install the latest stable release:

```bash
pip install gSELECT
```

For development:

```bash
git clone https://github.com/caliskandeniz/gSELECT.git
cd gSELECT
pip install -e .
```

---

## Dependencies

- `scanpy`, `anndata`
- `pandas`, `numpy`, `polars`, `scipy`
- `scikit-learn`, `principal_feature_analysis`
- `matplotlib`, `seaborn`
- `pyarrow`, `openpyxl`, `psutil`, `tqdm`

---

## Quick Start

### 1. **Explore and Load Data**

```python
from gSELECT import io as gsio

# Explore structure of .h5ad file
gsio.explore_h5ad("data/sample.h5ad")

# Load filtered expression data
genes, data = gsio.load_h5ad(
    "data/sample.h5ad",
    filter_column="cell_type",
    filter_values=["T cells", "B cells"]
)

# Or load from CSV
genes, data = gsio.load("data/sample.csv")
```

---

### 2. **Optional: Create Train/Test Split**

```python
# Transpose to (samples x genes)
data_total = data.transpose()

# Random 80/20 split
training_data = data_total.sample(frac=0.8)
test_data = data_total.drop(training_data.index)

# Transpose back
training_data = training_data.transpose()
test_data = test_data.transpose()
```

---

### 3. **Feature Selection (Mutual Information)**

```python
from gSELECT.feature_selection import compute_mutual_information

mi_scores = compute_mutual_information(
    genes,
    training_data,
    exclusion_list=["GAPDH", "ACTB"],
    output_folder="output"
)
```

---

### 4. **Run Classifiers**

```python
from gSELECT.classification import run_selected_genes

results = run_selected_genes(
    training_data,
    mi_scores,
    test_data=test_data,
    number_sweeps=5,
    top_n_genes=10,
    include_random=True
)
```

You can also explore:
- Multiple gene panel sizes: `run_multiple_gene_selections()`
- Custom gene lists: `run_with_custom_gene_set()`
- Exhaustive or greedy searches: `run_explorative_gene_selections()` and more

---

### 5. **Visualize Results**

```python
from gSELECT.visualization import plot_results

plot_results(
    results,
    output_folder="output",
    save_csv=True,
    save_png=True,
    csv_name="results.csv",
    dpi=300
)
```

---

## Core Functionality

### Data I/O

| Function | Purpose |
|---------|---------|
| `explore_h5ad()` | Inspect `.h5ad` structure |
| `load_h5ad()` | Load filtered `.h5ad` data |
| `load()` | Load expression data from CSV |

---

### Feature Selection

| Function | Purpose |
|---------|---------|
| `compute_mutual_information()` | Score genes based on MI vs. class labels |

---

### Classifiers

| Function | Description |
|----------|-------------|
| `run_selected_genes()` | Classify with top N MI-ranked genes |
| `run_multiple_gene_selections()` | Evaluate multiple panel sizes |
| `run_with_custom_gene_set()` | Use a user-defined gene list |
| `run_explorative_gene_selections()` | Exhaustive/greedy search of top N genes |
| `run_explorative_gene_selections_with_custom_set()` | Exhaustive search on custom gene sets |
| `run_all_genes()` | Use all non-constant genes |

---

### Visualization

| Function | Description |
|----------|-------------|
| `plot_results()` | Accuracy & comparison (single strategy) |
| `plot_multiple_gene_selections()` | Accuracy vs. panel size |
| `plot_explorative_gene_selections()` | Rank subset performance |
| `plot_all_genes()` | Visualize results from all genes |

---

## Output Summary

Each classification run produces:
- **Train/test accuracy**
- **Misclassified samples**
- **Number of genes used**
- Optional CSV and PNG files for result sharing or publication

---

## Example Notebook

See `examples/example_notebook.ipynb` for a full walkthrough from data to visualization.

---

## License

MIT License

---

## Contributing

Pull requests and issues are welcome!  
Start by cloning the repo and running:

```bash
pip install -e .
```

Then contribute via GitHub or submit ideas through issues.
