import pandas as pd
import scanpy as sc
import polars as pl
from scipy.sparse import issparse
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and higher-level messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: add timestamp and level
)


def explore_h5ad(file_path):
    """
    Load and explore an H5AD single-cell dataset.

    Produces:
    • Prints summary of dataset shape, cell and gene metadata, and matrix structure.

    Parameters
    ----------
    file_path : str
        Path to the H5AD file.
    """
    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load H5AD file '{file_path}': {e}")

    print(f"Loaded H5AD file: {file_path}")
    print(f"Shape of the data (cells, genes): {adata.shape}")
    
    print("\n--- Summary of `adata.obs` (cell metadata) ---")
    print(adata.obs.info())
    print(adata.obs.head())

    print("\n--- Unique values in `adata.obs` columns ---")
    for col in adata.obs.columns:
        unique_values = adata.obs[col].unique()
        num_unique = len(unique_values)
        print(f"Column: {col} | Unique values ({num_unique}):")
        print(unique_values[:10])
        if num_unique > 10:
            print("...")
        print()

    print("\n--- Summary of main data matrix (`adata.X`) ---")
    print("Matrix type:", type(adata.X))
    if hasattr(adata.X, "shape"):
        print(f"Shape: {adata.X.shape}")
        print("Preview of non-zero values (first 5 rows):")
        print(adata.X[:5, :].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :])

    print("\n--- Summary of `adata.var` (gene metadata) ---")
    print(adata.var.info())
    print(adata.var.head())


def load_h5ad(file_path, filter_column, filter_values):
    """
    Load an H5AD file and filter its contents.

    Produces:
    • Gene names and expression data after filtering cells by metadata.

    Parameters
    ----------
    file_path : str
        Path to the H5AD file.
    filter_column : str
        Column name in `adata.obs` used for filtering.
    filter_values : list
        List of values to retain in `filter_column`.

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        Gene names and expression data.
    """

    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        logging.error(f"Failed to load H5AD file: {file_path}. Error: {e}")
        return None
    logging.info(f"Loaded H5AD file: {file_path}")

    replace_map = {filter_values[0]: 0, filter_values[1]: 1}

    filtered_adata = adata[adata.obs[filter_column].isin(filter_values)]
    filtered_adata.obs[filter_column] = filtered_adata.obs[filter_column].replace(replace_map)
    filtered_adata.obs_names = filtered_adata.obs[filter_column].astype(int)

    if issparse(filtered_adata.X):
        expression_df = pd.DataFrame.sparse.from_spmatrix(filtered_adata.X).T
    else:
        expression_df = pd.DataFrame(filtered_adata.X).T

    expression_df.loc[-1] = filtered_adata.obs_names.tolist()
    expression_df.index += 1 # Shift index to start from 1
    expression_df.sort_index(inplace=True)  # Reorder indices to keep proper order

    adjusted_var_names = ["label"] + filtered_adata.var_names.tolist()
    expression_df.insert(0, "label", adjusted_var_names[: len(expression_df)])

    gene_names = expression_df[["label"]].reset_index(drop=True)
    gene_names.columns = ["gene_name"]
    
    if isinstance(expression_df.dtypes[0], pd.SparseDtype):
        expression_data = expression_df.iloc[:, 1:].reset_index(drop=True).sparse.to_dense()
    else:
        expression_data = expression_df.iloc[:, 1:].reset_index(drop=True)
    return gene_names, expression_data


def load(path, n_threads=4, use_low_memory=True):
    """
    Load gene expression data from a CSV file using Polars.

    Produces:
    • Gene names and expression data as DataFrames.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    n_threads : int, default 4
        Number of threads to use for reading.
    use_low_memory : bool, default True
        Whether to optimize memory usage.

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        Gene names and expression data.
    """
    logging.info(f"Loading gene names and expression data with Polars (using {n_threads} threads)...")

    df = pl.read_csv(
        path,
        has_header=False,
        n_threads=n_threads,
        low_memory=use_low_memory,
        infer_schema_length=100,
        use_pyarrow = True,
    )
    gene_names = df[:, 0].to_frame("gene_name").to_pandas()
    expression_data = df[:, 1:].to_pandas()    
    return gene_names, expression_data
