import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn import preprocessing
import concurrent.futures
import random


def run_gene_classification(expression_data, selected_gene_indices, gene_selection, number_sweeps=10):
    """
    Run gene classification using an MLP classifier.
    
    This function trains a neural network classifier on selected genes and evaluates
    its performance across multiple sweeps. It calculates balanced accuracy and 
    tracks misclassified samples.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    selected_gene_indices : list of int
        Indices of genes to be used for classification.
    gene_selection : int
        Mode of gene selection (0: selected genes, 1: random genes, 2: all non-constant genes).
    number_sweeps : int, optional (default=10)
        Number of iterations to run the classification.
    
    Returns:
    --------
    tuple (np.ndarray, np.ndarray, int, np.ndarray)
        - Test accuracy across sweeps.
        - Train accuracy across sweeps.
        - Gene selection mode.
        - Number of misclassified samples per sweep.
    """

    data_total = expression_data.transpose()
    data_train = data_total.sample(frac=0.8)
    data_test = data_total.drop(data_train.index)
    data_train = data_train.transpose().values
    data_test = data_test.transpose().values

    list_variables = np.sort(selected_gene_indices)

    r2_train = np.zeros((1, number_sweeps))
    r2_test = np.zeros((1, number_sweeps))
    number_wrongly_classified = np.zeros((1, number_sweeps))

    for sweep in range(0, number_sweeps):
        print(f"Running sweep {sweep + 1} of {number_sweeps}...")
        
        if gene_selection == 1:
            non_constant_metrics = np.where(np.ptp(data_train, axis=1) > 0)[0].tolist()
            list_variables = np.sort(random.sample(non_constant_metrics, len(list_variables)))

        X_train = data_train[list_variables, :].transpose()
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        y_train = data_train[0, :]
        X_test = data_test[list_variables, :].transpose()
        X_test_scaled = scaler.transform(X_test)
        y_test = data_test[0, :]

        mlp = MLPClassifier(max_iter=500)
        mlp.fit(X_train_scaled, y_train)

        # Make predictions on test and training data
        y_pred = mlp.predict(X_test_scaled)
        y_train_pred = mlp.predict(X_train_scaled)

        # Calculate balanced accuracy
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

        # Store results
        r2_test[0, sweep] = test_balanced_accuracy
        r2_train[0, sweep] = train_balanced_accuracy
        cm_mlp = confusion_matrix(y_test, y_pred)
        number_wrongly_classified[0, sweep] = cm_mlp[1, 0] + cm_mlp[0, 1]

    return r2_test, r2_train, gene_selection, number_wrongly_classified


def run_all_genes(expression_data, gene_mutual_information, number_sweeps=10):
    """
    Run classification using all non-constant genes.
    
    This function selects all genes with non-zero variance and evaluates classification 
    performance using the MLP classifier.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    
    Returns:
    --------
    list of tuple
        Contains classification accuracy and misclassification metrics.
    """

    print("Selecting all non-constant genes for classification...")

    selected_genes = gene_mutual_information.sort_values(by="mutual information", ascending=False)
    selected_gene_indices = selected_genes["index feature"].tolist()


    r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
        expression_data, selected_gene_indices, 2, number_sweeps
    )
    return [(r2_test, r2_train, gene_selection, number_wrongly_classified)]


def run_selected_genes(expression_data, gene_mutual_information, number_sweeps=10, top_n_genes=1, include_random=True):
    """
    Run classification using a subset of top-ranked genes.
    
    This function selects the top `n` genes based on mutual information scores and
    evaluates classification performance.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    top_n_genes : int, optional (default=1)
        Number of top-ranked genes to use for classification.
    include_random : bool, optional (default=True)
        Whether to also include a random selection of genes for comparison.
    
    Returns:
    --------
    list of tuple
        Contains classification accuracy and misclassification metrics for different gene selections.
    """

    results = []
    gene_selection_modes = [0, 1] if include_random else [0]

    selected_genes = gene_mutual_information.nlargest(top_n_genes, "mutual information")
    selected_gene_indices = sorted(list(selected_genes["index feature"]))

    print(f"Running experiment with {selected_genes}...")
    for gene_selection_it in gene_selection_modes:
        r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
            expression_data, selected_gene_indices, 
            gene_selection_it, number_sweeps
        )
        results.append((r2_test, r2_train, gene_selection, number_wrongly_classified))

    return results


def run_with_custom_gene_set(expression_data, selected_gene_names, gene_mutual_information, number_sweeps=10, include_random=True):
    """
    Run classification using a custom gene set.
    
    This function allows the user to specify a custom set of genes for classification.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    selected_gene_names : list of str
        List of gene names to use for classification.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    include_random : bool, optional (default=True)
        Whether to also include a random selection of genes for comparison.
    
    Returns:
    --------
    list of tuple
        Contains classification accuracy and misclassification metrics for different gene selections.
    """

    results = []
    gene_selection_modes = [0, 1] if include_random else [0]

    selected_genes = gene_mutual_information[gene_mutual_information["gene_name"].isin(selected_gene_names)]
    selected_gene_indices = sorted(selected_genes["index feature"].tolist())

    print(f"Running experiment with custom gene selection: {selected_genes})")
    for gene_selection_it in gene_selection_modes:
        r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
            expression_data, selected_gene_indices, gene_selection_it, number_sweeps
        )
        results.append((r2_test, r2_train, gene_selection, number_wrongly_classified))

    return results


def run_multiple_gene_selections(expression_data, gene_mutual_information, number_sweeps=10, gene_selection=[1, 2, 3]):
    """
    Run multiple classification experiments with different gene selection sizes.
    
    This function iterates over multiple gene selection sizes and evaluates 
    classification performance for each.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    gene_selection : list of int, optional (default=[1, 2, 3])
        List of different gene selection sizes to test.
    
    Returns:
    --------
    dict
        Mapping of gene selection sizes to classification results.
    """

    results = {}
    for top_n in gene_selection:
        results[top_n] = run_selected_genes(expression_data, gene_mutual_information, number_sweeps, top_n, False)
    return results


def run_explorative_gene_selections(expression_data, gene_mutual_information, number_sweeps=10, top_n_genes=5, num_threads=None):
    """
    Run an exploratory analysis of gene selection.
    
    This function generates all possible subsets of a given number of top-ranked genes
    and evaluates their classification performance.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    top_n_genes : int, optional (default=5)
        Number of top-ranked genes to explore in subset combinations.
    num_threads : int, optional (default=None)
        Number of parallel threads to use. If None, defaults to the number of available CPU cores.
    
    Returns:
    --------
    dict
        Mapping of gene subsets to classification results.

    Notes:
    ------
    - The function generates all possible subsets, ranging from 
      single-gene selections to the full set.
    - Each subset is evaluated independently using `run_with_custom_gene_set`.
    - The computation is parallelized using `ThreadPoolExecutor` to optimize performance.
    - If `num_threads` is not specified, the function will automatically determine an 
      appropriate number based on the system's CPU count.
    """

    gene_selection_combinations = generate_explorative_gene_selections(top_n_genes)
    top_genes = gene_mutual_information.nlargest(top_n_genes, "mutual information")["gene_name"].tolist()

    if num_threads is None:
        num_threads = max(2, os.cpu_count())
    
    results = {}

    def run_experiment(gene_subset):
        selected_gene_names = [top_genes[i-1] for i in gene_subset]
        print(f"Running experiment with gene subset: {selected_gene_names}")
        results[tuple(selected_gene_names)] = run_with_custom_gene_set(
            expression_data, selected_gene_names, gene_mutual_information, number_sweeps, include_random=False
        )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_experiment, gene_selection_combinations)

    return results


def run_explorative_gene_selections_with_custom_set(expression_data, selected_gene_names, gene_mutual_information, number_sweeps=10, num_threads=None):
    """
    Perform an exploratory analysis of gene selection using a custom set of genes.
    
    This function evaluates all possible subsets of a user-defined gene list by 
    iterating through different combinations of the selected genes and measuring 
    their classification performance. The computation is parallelized using 
    multi-threading to improve efficiency.

    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    selected_gene_names : list of str
        A list of gene names selected by the user for exploration.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps to run for each gene subset.
    num_threads : int, optional (default=None)
        Number of parallel threads to use. If None, defaults to the number of available CPU cores.

    Returns:
    --------
    dict
        Mapping of gene subsets (as tuples) to classification results.

    Notes:
    ------
    - The function generates all possible subsets of the provided gene set, ranging from 
      single-gene selections to the full set.
    - Each subset is evaluated independently using `run_with_custom_gene_set`.
    - The computation is parallelized using `ThreadPoolExecutor` to optimize performance.
    - If `num_threads` is not specified, the function will automatically determine an 
      appropriate number based on the system's CPU count.
    """

    gene_selection_combinations = []
    num_genes = len(selected_gene_names)

    for r in range(1, num_genes + 1):  # Generate subsets of size 1 to max
        gene_selection_combinations.extend(combinations(selected_gene_names, r))

    results = {}

    def run_experiment(gene_subset):
        print(f"Running experiment with gene subset: {gene_subset}")
        results[tuple(gene_subset)] = run_with_custom_gene_set(
            expression_data, list(gene_subset), gene_mutual_information, number_sweeps, include_random=False
        )

    if num_threads is None:
        num_threads = max(2, os.cpu_count())

    print(f"Using {num_threads} threads for parallel execution...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(run_experiment, gene_selection_combinations)
    
    return results
