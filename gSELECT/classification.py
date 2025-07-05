import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn import preprocessing
from itertools import combinations
import concurrent.futures
import random
import os
import gSELECT.utils as gsutils
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and higher-level messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: add timestamp and level
)


def run_gene_classification(expression_data, selected_gene_indices, gene_selection, test_data=None, number_sweeps=10, max_iterations=500):
    """
    Run gene classification using an MLP classifier.
    
    This function trains a neural network classifier on selected genes and evaluates
    its performance across multiple sweeps. It calculates balanced accuracy and 
    tracks misclassified samples.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    test_data: np.ndarray
        Gene expression data that is only used for testing
    selected_gene_indices : list of int
        Indices of genes to be used for classification.
    gene_selection : int
        Mode of gene selection (0: selected genes, 1: random genes, 2: all non-constant genes).
    number_sweeps : int, optional (default=10)
        Number of iterations to run the classification.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm
    
    Returns:
    --------
    tuple (np.ndarray, np.ndarray, int, np.ndarray)
        - Test accuracy across sweeps.
        - Train accuracy across sweeps.
        - Gene selection mode.
        - Number of misclassified samples per sweep.
    """
    list_variables = np.sort(selected_gene_indices)

    r2_train = np.zeros((1, number_sweeps))
    r2_test = np.zeros((1, number_sweeps))
    number_wrongly_classified = np.zeros((1, number_sweeps))

    for sweep in range(0, number_sweeps):
        logging.info(f"Running sweep {sweep + 1} of {number_sweeps}...")
        gsutils.log_peak_memory_usage()

        if  test_data is not None and sweep < number_sweeps - 1:
            data_total = expression_data.transpose()
            data_train = data_total.sample(frac=0.8)
            data_test = data_total.drop(data_train.index)
            data_train = data_train.transpose().values
            data_test = data_test.transpose().values
        else:
            data_train = expression_data.values
            data_test = test_data.values
        
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


def run_all_genes(expression_data, gene_mutual_information, test_data=None, number_sweeps=10, max_iterations=500):
    """
    Run classification using all non-constant genes.
    
    This function selects all genes with non-zero variance and evaluates classification 
    performance using the MLP classifier.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    test_data: np.ndarray
        Gene expression data that is only used for testing
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm
    
    Returns:
    --------
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


def run_selected_genes(expression_data, gene_mutual_information, test_data=None, number_sweeps=10, top_n_genes=1, include_random=True, max_iterations=500):
    """
    Run classification using a subset of top-ranked genes.
    
    This function selects the top `n` genes based on mutual information scores and
    evaluates classification performance.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    test_data: np.ndarray
        Gene expression data that is only used for testing
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    top_n_genes : int, optional (default=1)
        Number of top-ranked genes to use for classification.
    include_random : bool, optional (default=True)
        Whether to also include a random selection of genes for comparison.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm
    
    Returns:
    --------
    list of tuple
        Contains classification accuracy and misclassification metrics for different gene selections.
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


def run_with_custom_gene_set(expression_data, selected_gene_names, gene_mutual_information, test_data=None, number_sweeps=10, include_random=True, max_iterations=500):
    """
    Run classification using a custom gene set.
    
    This function allows the user to specify a custom set of genes for classification.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    test_data: np.ndarray
        Gene expression data that is only used for testing
    selected_gene_names : list of str
        List of gene names to use for classification.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    include_random : bool, optional (default=True)
        Whether to also include a random selection of genes for comparison.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm
    
    Returns:
    --------
    list of tuple
        Contains classification accuracy and misclassification metrics for different gene selections.
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


def run_multiple_gene_selections(expression_data, gene_mutual_information, test_data=None, number_sweeps=10, gene_selection=[1, 2, 3], max_iterations=500):
    """
    Run multiple classification experiments with different gene selection sizes.
    
    This function iterates over multiple gene selection sizes and evaluates 
    classification performance for each.
    
    Parameters:
    -----------
    expression_data : np.ndarray
        Gene expression data (cells x genes).
    test_data: np.ndarray
        Gene expression data that is only used for testing
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps.
    gene_selection : list of int, optional (default=[1, 2, 3])
        List of different gene selection sizes to test.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm
    
    Returns:
    --------
    dict
        Mapping of gene selection sizes to classification results.
    """

    results = {}
    for top_n in gene_selection:
        results[top_n] = run_selected_genes(expression_data, gene_mutual_information, test_data, number_sweeps, top_n, False, max_iterations=max_iterations)
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
    Exhaustive search of all subsets drawn from the `top_n_genes` **or**—when
    that search would explode combinatorially—an automatic fall-back to greedy
    forward selection.

    The routine is a convenience wrapper around two lower-level engines:

    * **Exhaustive engine** – evaluates *every* non-empty subset
      (``2^n − 1`` possibilities) of the `top_n_genes` with
      `run_with_custom_gene_set`, using a thread pool for parallelism.
    * **Greedy engine** – delegates to :pyfunc:`run_greedy_selection`
      (one-gene-at-a-time forward selection) when
      ``top_n_genes > greedy_threshold`` *and* ``use_greedy_if_large`` is
      ``True``.

    Returns a dictionary ``{ subset_tuple : result_list }`` identical in shape
    to the outputs of the underlying helpers, so downstream code remains
    unchanged.

    Parameters
    ----------
    expression_data : numpy.ndarray
        Gene-expression matrix of shape *(cells × genes)*.
    test_data: np.ndarray
        Gene expression data that is only used for testing
    gene_mutual_information : pandas.DataFrame
        DataFrame with at least two columns:

        ``gene_name`` – gene identifier  
        ``mutual information`` – MI score used to rank genes
    number_sweeps : int, default 10
        How many train/validation “sweeps” to perform per subset.
    top_n_genes : int, default 5
        Rank-cutoff for the exhaustive (or greedy) search.  Genes are chosen by
        descending mutual-information score.
    num_threads : int or None, default None
        Size of the :pyclass:`~concurrent.futures.ThreadPoolExecutor`.  If
        *None*, the runner uses ``max(2, os.cpu_count())``.
    max_iterations : int, default 500
        Upper bound on the MLP iterations passed through to
        :pyfunc:`run_with_custom_gene_set`.
    use_greedy_if_large : bool, keyword-only, default True
        Toggle for the automatic fall-back.  Set to *False* to force the
        (potentially huge) exhaustive enumeration.
    greedy_threshold : int, keyword-only, default 10
        If ``top_n_genes > greedy_threshold`` *and*
        ``use_greedy_if_large is True``, the function runs
        :pyfunc:`run_greedy_selection` instead of the exhaustive engine.

    Returns
    -------
    dict[tuple[str, ...], list]
        Mapping from *gene-subset tuples* to the list returned by
        :pyfunc:`run_with_custom_gene_set` or
        :pyfunc:`run_greedy_selection`.

    Notes
    -----
    * Exhaustive mode launches one thread per subset, so the total number of
      tasks is ``2^n − 1``.  A warning with the exact count is logged before
      execution.
    * In greedy mode the maximum panel size equals ``greedy_threshold`` (the
      same cut-off that triggered the fall-back), providing a consistent cap on
      runtime.
    * Any exception raised while evaluating an individual subset is caught,
      logged, and that subset is skipped; remaining jobs continue unaffected.
    """

    # ------------------------------------------------------------------ #
    # 0.  Automatic fall-back to greedy if the search would explode       #
    # ------------------------------------------------------------------ #
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
            max_panel_size=greedy_threshold,   # ← use threshold as panel cap
            max_iterations=max_iterations,
        )

    # ------------------------------------------------------------------ #
    # 1.  Proceed with exhaustive enumeration as before                  #
    # ------------------------------------------------------------------ #
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

    # pick sensible thread count
    if num_threads is None:
        num_threads = max(2, os.cpu_count())

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
    allow_swaps: bool = False,       # ← upgrade #2
    beam_width: int | None = None,   # ← upgrade #3
):
    """
    Forward greedy feature selection **with backtracking to the best panel seen**.

    Stops when:
    * every candidate gene has failed to improve the *current* panel **and**
    * we have looped once over the entire remaining pool **after** the last
      positive gain.

    Optional extras
    ---------------
    allow_swaps : bool
        If *True* perform a “with-replacement” pass after each successful add:
        swap one selected gene with one unselected if the swap increases
        accuracy.
    beam_width : int or None
        If given, keep the `beam_width` best partial panels at every depth
        (classic beam search).  ``None`` reproduces single-path greedy.
    """

    # ---------- prep ---------------------------------------------------
    top_genes = (
        gene_mutual_information.nlargest(top_n_genes, "mutual information")
        ["gene_name"]
        .tolist()
    )

    Panel = tuple[str, ...]           # readable alias
    results: dict[Panel, list] = {}
    beam    = [()]                    # list of current partial panels
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

    # ---------- main loop ----------------------------------------------
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

        # choose best K (=beam_width) panels
        candidates.sort(key=lambda t: t[1], reverse=True)
        top_k = candidates[: (beam_width or 1)]

        best_new_panel, best_new_score = top_k[0]

        # update global best if needed
        if best_new_score > best_score:
            best_panel, best_score = best_new_panel, best_new_score
            iter_no_gain = 0
        else:
            iter_no_gain += 1

        # early-stop when we looped once w/o any gain
        if iter_no_gain >= len(top_genes):
            logging.info("Early stopping (searched full pool since last gain).")
            break

        # optional “swap” pass ------------------------------------------
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

        # advance beam
        beam = [p for p, _ in top_k]

    # ensure best panel & its result are retained
    if best_panel is not None:
        best_panel = tuple(best_panel)
        results.setdefault(best_panel, results[best_panel])

    return results




def run_explorative_gene_selections_with_custom_set(expression_data, selected_gene_names, gene_mutual_information, test_data=None, number_sweeps=10, num_threads=None, max_iterations=500):
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
    test_data: np.ndarray
        Gene expression data that is only used for testing
    selected_gene_names : list of str
        A list of gene names selected by the user for exploration.
    gene_mutual_information : pd.DataFrame
        DataFrame containing mutual information scores for genes.
    number_sweeps : int, optional (default=10)
        Number of classification sweeps to run for each gene subset.
    num_threads : int, optional (default=None)
        Number of parallel threads to use. If None, defaults to the number of available CPU cores.
    max_iterations: int, optional (default=500)
        Number of iterations of the MLP algorithm

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
        logging.info(f"Running experiment with gene subset: {gene_subset}")
        results[tuple(gene_subset)] = run_with_custom_gene_set(
            expression_data, list(gene_subset), gene_mutual_information, test_data, number_sweeps, include_random=False, max_iterations=max_iterations
        )

    if num_threads is None:
        num_threads = max(2, os.cpu_count())

    logging.info(f"Using {num_threads} threads for parallel execution...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(run_experiment, gene_selection_combinations)
    
    return results
