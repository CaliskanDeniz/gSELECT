import numpy as np
import pandas as pd
import os
from scipy.sparse import issparse
import principal_feature_analysis as pfa


def get_mutual_information(
    gene_names,
    expression_data,
    gene_list,
    min_datapoints=10,
    basis_log=2,
    number_output_functions=1,
    bin_count=None
):
    """
    Compute mutual information scores for gene selection.

    Produces:
    • DataFrame of mutual information scores for each gene.

    Parameters
    ----------
    gene_names : pd.DataFrame
        DataFrame containing gene names.
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_list : list of str or None
        List of genes to evaluate. If None, computes for all genes.
    min_datapoints : int, default 10
        Minimum data points for valid computation.
    basis_log : int, default 2
        Logarithm base for entropy calculation.
    number_output_functions : int, default 1
        Number of output variables to consider.
    bin_count : int, default None (sturge's rule)
        Number of bins for discretization.

    Returns
    -------
    pd.DataFrame
        Mutual information scores for genes.
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

    non_constant_features = np.where(np.ptp(expression_data, axis=1) > 0)[0].tolist()
    list_variables = [i for i in range(number_output_functions)] + non_constant_features

    num_samples = expression_data.shape[1]
    #sturge's_rule(bin_count, num_samples)
    desired_number_of_bins = 1 + int(np.log2(num_samples)) if bin_count is None else bin_count
    print(f"Desired number of bins: {desired_number_of_bins}")
    
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


def compute_mutual_information(
    gene_names,
    expression_data,
    gene_list=None, 
    min_datapoints=10, 
    basis_log=2,
    bin_count=None,
    exclusion_list=[],
    should_save_mutual_info=True,
    output_folder="output"
):
    """
    Compute and optionally save mutual information scores for gene selection.

    Produces:
    • DataFrame of mutual information scores for each gene.
    • Optionally saves results to CSV.

    Parameters
    ----------
    gene_names : pd.DataFrame
        DataFrame containing gene names.
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_list : list of str or None, optional
        List of genes to evaluate. If None, computes for all genes.
    top_mutual_information : int, default 1
        Number of top genes to retain.
    min_datapoints : int, default 10
        Minimum data points for valid computation.
    basis_log : int, default 2
        Logarithm base for entropy calculation.
    bin_count : int, default None (Sturge's rule)
        Number of bins for discretization.
    exclusion_list : list of str, optional
        Genes to exclude from results.
    should_save_mutual_info : bool, default True
        If True, save results to CSV.
    output_folder : str, default "output"
        Directory for output files.

    Returns
    -------
    pd.DataFrame
        Mutual information scores for genes.
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
            min_datapoints=min_datapoints,
            basis_log=basis_log,
            bin_count=bin_count
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