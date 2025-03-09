import pandas as pd
import scanpy as sc
import polars as pl
from scipy.sparse import issparse
import os
import logging


def explore_h5ad(file_path):
    """
    Load and explore an H5AD single-cell dataset.

    This function prints key metadata from the AnnData object, including:
    - Shape of the dataset (cells x genes)
    - Summary of cell metadata (`adata.obs`)
    - Unique values per column in `adata.obs`
    - Summary of gene metadata (`adata.var`)
    - Matrix structure of `adata.X`

    Parameters:
    -----------
    file_path : str
        Path to the H5AD file.

    Raises:
    -------
    RuntimeError:
        If the file cannot be loaded.
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

    # Display unique values for each column in `adata.obs`
    print("\n--- Unique values in `adata.obs` columns ---")
    for col in adata.obs.columns:
        unique_values = adata.obs[col].unique()
        num_unique = len(unique_values)
        print(f"Column: {col} | Unique values ({num_unique}):")
        print(unique_values[:10])  # Show the first 10 unique values
        if num_unique > 10:
            print("...")  # Indicate that there are more unique values
        print()  # Add a blank line for readability

    # Sparse matrix: Summarize without converting to dense
    print("\n--- Summary of main data matrix (`adata.X`) ---")
    print("Matrix type:", type(adata.X))
    if hasattr(adata.X, "shape"):
        print(f"Shape: {adata.X.shape}")
        print("Preview of non-zero values (first 5 rows):")
        print(adata.X[:5, :].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :])

    # Optional: Keep summary of `adata.var` for context
    print("\n--- Summary of `adata.var` (gene metadata) ---")
    print(adata.var.info())
    print(adata.var.head())


def load_h5ad(file_path, filter_column, filter_values):
    """
    Load an H5AD file and filter its contents.

    This function:
    - Reads an H5AD file using Scanpy
    - Filters cells based on a specified metadata column (`filter_column`)
    - Replaces values in the specified column (`replace_map`)
    - Converts expression data to a DataFrame
    - Optionally saves the output as a CSV

    Parameters:
    -----------
    file_path : str
        Path to the H5AD file.
    filter_column : str
        Column name in `adata.obs` used for filtering.
    filter_values : list
        List of values to retain in `filter_column`.
    replace_map : dict
        Dictionary mapping old values to new values in `filter_column`.

    Returns:
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        - Gene names as a DataFrame
        - Expression data as a DataFrame
    """

    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        logging.error(f"Failed to load H5AD file: {file_path}. Error: {e}")
        return None
    print(f"Loaded H5AD file: {file_path}")

    replace_map = {filter_values[0]: 0, filter_values[1]: 1}

    filtered_adata = adata[adata.obs[filter_column].isin(filter_values)]
    filtered_adata.obs[filter_column] = filtered_adata.obs[filter_column].replace(replace_map)
    filtered_adata.obs_names = filtered_adata.obs[filter_column].astype(int)

    if issparse(filtered_adata.X):
        expression_df = pd.DataFrame.sparse.from_spmatrix(filtered_adata.X).T
    else:
        expression_df = pd.DataFrame(filtered_adata.X).T

    expression_df.loc[-1] = filtered_adata.obs_names.tolist()
    expression_df.index += 1  # Shift index to make room for the new row
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

    This function:
    - Uses Polars to efficiently read large CSV files
    - Separates gene names from expression data
    - Supports multi-threaded reading for faster performance

    Parameters:
    -----------
    path : str
        Path to the CSV file.
    n_threads : int, optional (default=4)
        Number of threads to use for reading the file.
    use_low_memory : bool, optional (default=True)
        Whether to optimize memory usage.

    Returns:
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        - Gene names as a DataFrame
        - Expression data as a DataFrame
    """

    print(f"Loading gene names and expression data with Polars (using {n_threads} threads)...")

    df = pl.read_csv(
        path,
        has_header=False,
        n_threads=n_threads,
        low_memory=use_low_memory,
        infer_schema_length=10,
        use_pyarrow = True,
    )
    # Separate gene names and expression data
    gene_names = df[:, 0].to_frame("gene_name").to_pandas()
    expression_data = df[:, 1:].to_pandas()    
    return gene_names, expression_data
