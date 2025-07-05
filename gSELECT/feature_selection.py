import numpy as np
import pandas as pd
import os
from scipy.sparse import issparse
import principal_feature_analysis as pfa


def get_mutual_information( gene_names, expression_data, gene_list, top_mutual_information, min_datapoints=10, basis_log=2, number_output_functions=1):
    """
    Compute mutual information for gene selection.
    
    This function calculates mutual information scores between genes and a target variable 
    to identify the most informative features for classification. If a gene list is provided, 
    it verifies the presence of those genes and assigns mock mutual information values.
    
    Parameters:
    -----------
    gene_names : pd.DataFrame
        DataFrame containing gene names.
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_list : list of str or None
        List of specific genes to evaluate. If None, computes mutual information for all genes.
    top_mutual_information : int
        Number of top genes to retain based on mutual information scores.
    min_datapoints : int, optional (default=10)
        Minimum number of data points required for valid mutual information computation.
    basis_log : int, optional (default=2)
        Base logarithm for entropy calculation.
    number_output_functions : int, optional (default=1)
        Number of output variables to consider.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing mutual information scores for genes.
    """

    if gene_list:
        genes = gene_names["gene_name"]
        
        index_feature = pd.Series(range(len(genes)), name="index feature")
        mutual_information = pd.Series([0] * len(genes), name="mutual information")
        
        missing_genes = []
        for gene in gene_list:
            if gene in genes.values:
                mutual_information.loc[genes == gene] = 1
            else:
                missing_genes.append(gene)
        if missing_genes:
            raise ValueError(f"ERROR: The following genes were not found: {', '.join(missing_genes)}")

        gene_mutual_information = pd.concat([index_feature, mutual_information, gene_names], axis=1)
        print("Mock mutual information created successfully.")
        
        return gene_mutual_information  

    # Calculate mutual information
    # non_constant_features = [i for i in range(expression_data.shape[0]) if expression_data.iloc[i].nunique() > 1]
    non_constant_features = np.where(np.ptp(expression_data, axis=1) > 0)[0].tolist()
    list_variables = [i for i in range(number_output_functions)] + non_constant_features

    num_samples = expression_data.shape[1]
    desired_number_of_bins = 10
    min_n_data_points_a_bin = max(min_datapoints, num_samples // desired_number_of_bins)
    min_n_data_points_a_bin = min(min_n_data_points_a_bin, num_samples // 2)
    
    mutual_info_results = pfa.get_mutual_information(
        expression_data,
        number_output_functions,
        list_variables,
        min_n_data_points_a_bin,
        basis_log
    )
    mutual_info_df = mutual_info_results[0]
    mutual_info_df['gene_name'] = gene_names['gene_name'].iloc[mutual_info_df['index feature']].values

    gene_mutual_information = mutual_info_df[['gene_name', 'index feature', 'mutual information']]
    gene_mutual_information = gene_mutual_information[
        gene_mutual_information['gene_name'].str.lower() != 'label'
    ]
    return gene_mutual_information


def compute_mutual_information(gene_names, expression_data, gene_list=None, 
                               top_mutual_information=1, min_datapoints=10, 
                               basis_log=2, exclusion_list=[],
                               should_save_mutual_info=True, output_folder="output"):
    """
    Compute and optionally save mutual information scores.
    
    This function calculates mutual information for feature selection and optionally saves
    the results to a CSV file. If a saved mutual information file already exists, it loads 
    the precomputed values.
    
    Parameters:
    -----------
    gene_names : pd.DataFrame
        DataFrame containing gene names.
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_list : list of str or None, optional (default=None)
        List of specific genes to evaluate. If None, computes mutual information for all genes.
    top_mutual_information : int, optional (default=1)
        Number of top genes to retain based on mutual information scores.
    min_datapoints : int, optional (default=10)
        Minimum number of data points required for valid mutual information computation.
    basis_log : int, optional (default=2)
        Base logarithm for entropy calculation.
    exclusion_list : list of str, optional (default=[])
        List of genes to exclude from the final mutual information results.
    should_save_mutual_info : bool, optional (default=True)
        Whether to save the computed mutual information values to a CSV file.
    output_folder : str, optional (default="output")
        Directory where the CSV file will be saved.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing mutual information scores for genes.
    """

    os.makedirs(output_folder, exist_ok=True)

    mutual_info_path = os.path.join(output_folder, "mutual_information.csv")
    gene_mutual_information = None

    if(os.path.isfile(mutual_info_path)):
        gene_mutual_information = pd.read_csv(mutual_info_path)
        gene_mutual_information = gene_mutual_information.sort_values(
        by="mutual information", ascending=False
        )
        print(f"Mutual information taken from file {mutual_info_path}")
    else:
        gene_mutual_information = get_mutual_information(
            gene_names=gene_names,
            expression_data=expression_data,
            gene_list=gene_list if gene_list else None,
            top_mutual_information=top_mutual_information if not gene_list else len(gene_list),
            min_datapoints=min_datapoints,
            basis_log=basis_log
        )
        # Sort by mutual information in descending order
        gene_mutual_information = gene_mutual_information.sort_values(
            by="mutual information", ascending=False
        )
        if should_save_mutual_info:
            mutual_info_csv_path = os.path.join(output_folder, "mutual_information.csv")
            gene_mutual_information.to_csv(mutual_info_csv_path, index=False)
            print(f"Full mutual information saved to {mutual_info_csv_path}")

    if exclusion_list:
        gene_mutual_information = gene_mutual_information[~gene_mutual_information["gene_name"].isin(exclusion_list)]

    return gene_mutual_information