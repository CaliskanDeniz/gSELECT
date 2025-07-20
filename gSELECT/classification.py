import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn import preprocessing
from itertools import combinations
import concurrent.futures
import psutil
import os
import gSELECT.utils as gsutils
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and higher-level messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: add timestamp and level
)


def run_gene_classification(
    expression_data,
    selected_gene_indices,
    gene_selection,
    test_data=None,
    number_sweeps=10,
    max_iterations=500
):
    """
    Train and evaluate an MLP classifier on selected genes.

    Produces:
    • Test and train balanced accuracy across sweeps.
    • Misclassified sample counts per sweep.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    selected_gene_indices : list of int
        Indices of genes to use for classification.
    gene_selection : int
        Mode of gene selection (0: selected, 1: random, 2: all non-constant).
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    tuple
        (test_accuracy, train_accuracy, gene_selection, misclassified_counts)
    """
    list_variables = np.sort(selected_gene_indices)

    r2_train = np.zeros((1, number_sweeps))
    r2_test = np.zeros((1, number_sweeps))
    number_wrongly_classified = np.zeros((1, number_sweeps))

    for sweep in range(0, number_sweeps):
        logging.info(f"Running sweep {sweep + 1} of {number_sweeps}...")
        gsutils.log_peak_memory_usage()

        if test_data is not None and sweep >= number_sweeps - 1:
            # if last sweep and we have testdata available:
            data_train = expression_data.values
            data_test = test_data.values
        else:
            # Monte Carlo CV
            data_total = expression_data.transpose()
            data_train = data_total.sample(frac=0.8)
            data_test = data_total.drop(data_train.index)
            data_train = data_train.transpose().values
            data_test = data_test.transpose().values

        if gene_selection == 1:
            np.random.seed()
            non_constant_metrics = np.where(np.ptp(data_train, axis=1) > 0)[0].tolist()
            list_variables = np.sort(np.random.choice(non_constant_metrics, size=len(list_variables), replace=False))

        X_train = data_train[list_variables, :].transpose()
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        y_train = data_train[0, :]
        X_test = data_test[list_variables, :].transpose()
        X_test_scaled = scaler.transform(X_test)
        y_test = data_test[0, :]

        mlp = MLPClassifier(max_iter=max_iterations)
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


def run_all_genes(
    expression_data,
    gene_mutual_information,
    test_data=None,
    number_sweeps=10,
    max_iterations=500
):
    """
    Run classification using all non-constant genes.

    Produces:
    • Test and train balanced accuracy across sweeps.
    • Misclassified sample counts per sweep.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    list of tuple
        Contains classification accuracy and misclassification metrics.
    """
    logging.info("Selecting all non-constant genes for classification...")

    selected_genes = gene_mutual_information.sort_values(by="mutual information", ascending=False)
    selected_gene_indices = selected_genes["index feature"].tolist()


    r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
        expression_data, selected_gene_indices, 2, test_data, number_sweeps, max_iterations
    )
    return [(r2_test, r2_train, gene_selection, number_wrongly_classified)]


def run_selected_genes(
    expression_data,
    gene_mutual_information,
    test_data=None,
    number_sweeps=10,
    top_n_genes=1,
    include_random=True,
    max_iterations=500
):
    """
    Run classification using a subset of top-ranked genes.

    Produces:
    • Test and train balanced accuracy across sweeps.
    • Misclassified sample counts per sweep.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    top_n_genes : int, default 1
        Number of top-ranked genes to use for classification.
    include_random : bool, default True
        If True, also include random gene selection for comparison.
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    list of tuple
        Contains classification accuracy and misclassification metrics for each selection mode.
    """

    results = []
    gene_selection_modes = [0, 1] if include_random else [0]

    selected_genes = gene_mutual_information.nlargest(top_n_genes, "mutual information")
    selected_gene_indices = sorted(list(selected_genes["index feature"]))

    logging.info(f"Running experiment with {selected_genes}...")
    for gene_selection_it in gene_selection_modes:
        r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
            expression_data, selected_gene_indices, 
            gene_selection_it, test_data, number_sweeps, max_iterations
        )
        results.append((r2_test, r2_train, gene_selection, number_wrongly_classified))

    return results


def run_with_custom_gene_set(
    expression_data,
    selected_gene_names,
    gene_mutual_information,
    test_data=None, number_sweeps=10,
    include_random=True,
    max_iterations=500
):
    """
    Run classification using a custom set of genes.

    Produces:
    • Test and train balanced accuracy across sweeps.
    • Misclassified sample counts per sweep.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    selected_gene_names : list of str
        List of gene names to use for classification.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    include_random : bool, default True
        If True, also include random gene selection for comparison.
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    list of tuple
        Contains classification accuracy and misclassification metrics for each selection mode.
    """

    results = []
    gene_selection_modes = [0, 1] if include_random else [0]

    selected_genes = gene_mutual_information[gene_mutual_information["gene_name"].isin(selected_gene_names)]
    selected_gene_indices = sorted(selected_genes["index feature"].tolist())

    logging.info(f"Running experiment with custom gene selection: {selected_genes})")
    for gene_selection_it in gene_selection_modes:
        r2_test, r2_train, gene_selection, number_wrongly_classified = run_gene_classification(
            expression_data, selected_gene_indices, gene_selection_it, test_data, number_sweeps, max_iterations
        )
        results.append((r2_test, r2_train, gene_selection, number_wrongly_classified))

    return results


def run_multiple_gene_selections(
    expression_data,
    gene_mutual_information,
    test_data=None,
    number_sweeps=10,
    gene_selection=[1, 2, 3],
    max_iterations=500
):
    """
    Run classification experiments for multiple gene selection sizes.

    Produces:
    • Test and train balanced accuracy across sweeps for each panel size.
    • Misclassified sample counts per sweep.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    gene_selection : list of int, default [1, 2, 3]
        List of panel sizes to test.
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    dict
        Mapping of panel size to classification results.
    """

    results = {}
    for top_n in gene_selection:
        results[top_n] = run_selected_genes(
            expression_data,
            gene_mutual_information,
            test_data, number_sweeps,
            top_n, False,
            max_iterations=max_iterations
        )
    return results


def run_explorative_gene_selections(
    expression_data,
    gene_mutual_information,
    test_data=None,
    number_sweeps: int = 10,
    top_n_genes: int = 5,
    num_threads: int | None = None,
    max_iterations: int = 500,
    use_greedy_if_large: bool = True,
    greedy_threshold: int = 10
):
    """
    Explore all subsets of the top-ranked genes or use greedy selection if too large.

    Produces:
    • Classification results for all non-empty subsets or greedy panels.
    • Parallel execution for efficiency.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps per subset.
    top_n_genes : int, default 5
        Number of top-ranked genes to explore.
    num_threads : int or None, default None
        Number of parallel threads (auto if None).
    max_iterations : int, default 500
        Maximum MLP iterations.
    use_greedy_if_large : bool, default True
        Use greedy selection if subset count is too large.
    greedy_threshold : int, default 10
        Switch to greedy if top_n_genes exceeds this value.

    Returns
    -------
    dict
        Mapping of gene-subset tuples to classification results.
    """
    if use_greedy_if_large and top_n_genes > greedy_threshold:
        logging.info(
            f"top_n_genes={top_n_genes} exceeds greedy_threshold="
            f"{greedy_threshold}.  Switching to greedy forward selection."
        )
        return run_greedy_selection(
            expression_data=expression_data,
            test_data=test_data,
            gene_mutual_information=gene_mutual_information,
            number_sweeps=number_sweeps,
            top_n_genes=top_n_genes,
            max_panel_size=greedy_threshold,
            max_iterations=max_iterations,
        )
    gene_selection_combinations = gsutils.generate_explorative_gene_selections(
        top_n_genes
    )
    logging.warning(
        "Total number of gene subset combinations: "
        f"{len(gene_selection_combinations)} — runtime can be long."
    )

    top_genes = (
        gene_mutual_information.nlargest(top_n_genes, "mutual information")
        ["gene_name"]
        .tolist()
    )
    if num_threads is None:
        base_size_mb = expression_data.memory_usage().sum() / 1e6

        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Total usable = available RAM + available Swap
        total_usable_mb = (vmem.available + swap.free) / 1e6
        total_usable_mb *= 0.8 # safety factor

        if base_size_mb <= 0:
            num_threads =1

        num_threads = max(int(total_usable_mb // base_size_mb), 1)
        num_threads = min(os.cpu_count(), num_threads)
        logging.info(f"Selected number of Threads based on free memory and cpu count: {num_threads}")
    
    results: dict[tuple[str, ...], list] = {}

    def run_experiment(gene_subset):
        selected_gene_names = [top_genes[i - 1] for i in gene_subset]
        logging.info("Running subset: %s", selected_gene_names)
        results[tuple(selected_gene_names)] = run_with_custom_gene_set(
            expression_data,
            selected_gene_names,
            gene_mutual_information,
            test_data,
            number_sweeps,
            include_random=False,
            max_iterations=max_iterations,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as ex:
        fut_map = {ex.submit(run_experiment, s): s for s in gene_selection_combinations}
        for fut in concurrent.futures.as_completed(fut_map):
            try:
                fut.result()
            except Exception as e:
                logging.error("Subset %s failed: %s", fut_map[fut], e)

    return results


def run_greedy_selection(
    expression_data,
    gene_mutual_information,
    test_data=None,
    number_sweeps: int = 10,
    top_n_genes: int = 50,
    max_panel_size: int | None = None,
    max_iterations: int = 500,
    allow_swaps: bool = False,
    beam_width: int | None = None
):
    """
    Helper for greedy forward feature selection with optional swaps and beam search.

    Produces:
    • Classification results for panels built by greedy selection.
    • Optional swaps and beam search for improved panel selection.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps.
    top_n_genes : int, default 50
        Number of top-ranked genes to consider.
    max_panel_size : int or None, default None
        Maximum panel size.
    max_iterations : int, default 500
        Maximum MLP iterations.
    allow_swaps : bool, default False
        If True, allow swaps after each add.
    beam_width : int or None, default None
        Beam width for beam search.

    Returns
    -------
    dict
        Mapping of gene panels to classification results.
    """
    top_genes = (
        gene_mutual_information.nlargest(top_n_genes, "mutual information")
        ["gene_name"]
        .tolist()
    )

    Panel = tuple[str, ...]
    results: dict[Panel, list] = {}
    beam    = [()]
    best_panel: Panel | None = None
    best_score: float = -float("inf")

    def score_panel(panel: Panel) -> float:
        if panel not in results:
            res = run_with_custom_gene_set(
                expression_data,
                list(panel),
                gene_mutual_information,
                test_data,
                number_sweeps,
                include_random=False,
                max_iterations=max_iterations,
            )
            results[panel] = res
        return float(np.mean([r[0] for r in results[panel]]))

    iter_no_gain = 0
    while beam and (max_panel_size is None or len(beam[0]) < max_panel_size):
        candidates: list[tuple[Panel, float]] = []

        for panel in beam:
            remaining = [g for g in top_genes if g not in panel]
            for g in remaining:
                new_panel = tuple(panel + (g,))
                candidates.append((new_panel, score_panel(new_panel)))

        if not candidates:
            break

        candidates.sort(key=lambda t: t[1], reverse=True)
        top_k = candidates[: (beam_width or 1)]

        best_new_panel, best_new_score = top_k[0]

        if best_new_score > best_score:
            best_panel, best_score = best_new_panel, best_new_score
            iter_no_gain = 0
        else:
            iter_no_gain += 1

        if iter_no_gain >= len(top_genes):
            logging.info("Early stopping (searched full pool since last gain).")
            break

        if allow_swaps:
            improved = False
            panel_set = set(best_new_panel)
            for out_gene in panel_set:
                for in_gene in set(top_genes) - panel_set:
                    swapped = tuple(sorted((panel_set - {out_gene}) | {in_gene}))
                    swapped_score = score_panel(swapped)
                    if swapped_score > best_new_score:
                        best_new_panel, best_new_score = swapped, swapped_score
                        improved = True
                        break
                if improved:
                    break
        beam = [p for p, _ in top_k]

    if best_panel is not None:
        best_panel = tuple(best_panel)
        results.setdefault(best_panel, results[best_panel])

    return results




def run_explorative_gene_selections_with_custom_set(
    expression_data,
    selected_gene_names,
    gene_mutual_information,
    test_data=None,
    number_sweeps=10,
    num_threads=None,
    max_iterations=500
):
    """
    Explore all subsets of a custom gene set for classification.

    Produces:
    • Classification results for all non-empty subsets.
    • Parallel execution for efficiency.

    Parameters
    ----------
    expression_data : np.ndarray
        Gene expression data (cells × genes).
    selected_gene_names : list of str
        List of gene names to explore.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    test_data : np.ndarray, optional
        Separate test data (if available).
    number_sweeps : int, default 10
        Number of classification sweeps per subset.
    num_threads : int, default None
        Number of parallel threads (auto if None).
    max_iterations : int, default 500
        Maximum MLP iterations.

    Returns
    -------
    dict
        Mapping of gene-subset tuples to classification results.
    """

    gene_selection_combinations = []
    num_genes = len(selected_gene_names)

    for r in range(1, num_genes + 1):  # Generate subsets of size 1 to max
        gene_selection_combinations.extend(combinations(selected_gene_names, r))

    results = {}

    def run_experiment(gene_subset):
        logging.info(f"Running experiment with gene subset: {gene_subset}")
        results[tuple(gene_subset)] = run_with_custom_gene_set(
            expression_data,
            list(gene_subset),
            gene_mutual_information,
            test_data, number_sweeps,
            include_random=False,
            max_iterations=max_iterations
    )

    if num_threads is None:
        num_threads = max(2, os.cpu_count())

    logging.info(f"Using {num_threads} threads for parallel execution...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(run_experiment, gene_selection_combinations)
    
    return results
