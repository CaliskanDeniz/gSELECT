import matplotlib.pyplot as plt
from gSELECT.utils import get_unique_filename
import seaborn as sns
import os
import numpy as np
import logging


def plot_results(results, output_folder="output"):
    """
    Generate and save plots for classification results.
    
    This function computes key statistics (mean accuracy, standard deviation, misclassification rates)
    for different gene selection strategies and saves them as text files and visual plots.
    
    It generates:
    - A text summary of classification performance.
    - A line plot comparing balanced accuracy across different sweeps.
    - A bar plot showing the number of misclassified samples.
    
    Parameters:
    -----------
    results : list of tuples
        List containing classification results in the format:
        (r2_test, r2_train, gene_selection, number_wrongly_classified).
    output_folder : str, optional (default="output")
        Directory where plots and summary files will be saved.
    
    Outputs:
    --------
    - Text file: "experiment_statistics.txt" with accuracy and misclassification summaries.
    - PNG plot: "balanced_accuracy_test_train_plot.png" showing accuracy trends.
    - PNG plot: "misclassified_samples_plot.png" visualizing misclassification rates.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Initialize variables for output data
    output_data = []
    gene_selection_labels = {0: "Selected Genes", 1: "Random Genes", 2: "All Non-constant Genes"}

    # Compute statistics for each result set
    for r2_test, r2_train, gene_selection, number_wrongly_classified in results:
        mode_label = gene_selection_labels.get(gene_selection, f"Unknown Mode {gene_selection}")
        test_mean = r2_test.mean()
        train_mean = r2_train.mean()
        test_std = r2_test.std()
        train_std = r2_train.std()
        misclassified_mean = number_wrongly_classified.mean()
        misclassified_std = number_wrongly_classified.std()

        # Append stats to output data
        output_data.append(
            f"Gene Selection Mode: {mode_label}\n"
            f"  Test Mean Accuracy: {test_mean:.4f}\n"
            f"  Train Mean Accuracy: {train_mean:.4f}\n"
            f"  Test Accuracy Std Dev: {test_std:.4f}\n"
            f"  Train Accuracy Std Dev: {train_std:.4f}\n"
            f"  Misclassified Samples Mean: {misclassified_mean:.2f}\n"
            f"  Misclassified Samples Std Dev: {misclassified_std:.2f}\n"
            f"{'-' * 40}\n"
        )

    # Write statistics to a text file
    stats_file_path = get_unique_filename(os.path.join(output_folder, "experiment_statistics.txt"))
    with open(stats_file_path, "w") as f:
        f.writelines(output_data)
    print(f"Experiment statistics saved to {stats_file_path}")

    # Print statistics to the console
    print("\n".join(output_data))

    # Plot balanced accuracy across sweeps (Test and Train)
    print("Plotting balanced accuracy across sweeps (Test and Train)...")
    plt.figure(figsize=(12, 8))
    colors = {0: 'blue', 1: 'green', 2: 'pink'}
    labels = {0: 'Selected Genes', 1: 'Random Genes', 2: 'All Non-constant Genes'}

    for r2_test, r2_train, gene_selection, _ in results:
        label = labels[gene_selection]
        # Plot Test Balanced Accuracy
        plt.plot(range(len(r2_test[0])), r2_test[0], linestyle='-', color=colors[gene_selection], label=f"{label} (Test)", linewidth=2)
        # Plot Train Balanced Accuracy
        plt.plot(range(len(r2_train[0])), r2_train[0], linestyle='--', color=colors[gene_selection], label=f"{label} (Train)", linewidth=2)

    plt.xlabel('Sweep Number', fontsize=14, fontweight='bold')
    plt.ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    plt.title('Balanced Accuracy Across Sweeps (Test and Train)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True)

    accuracy_plot_path = os.path.join(output_folder, "balanced_accuracy_test_train_plot.png")
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Plot misclassified samples with standard deviation
    print("Plotting misclassified samples with standard deviation...")
    misclassified_means = [number_wrongly_classified.mean() for _, _, _, number_wrongly_classified in results]
    misclassified_stds = [number_wrongly_classified.std() for _, _, _, number_wrongly_classified in results]

    x_labels = [labels[result[2]] for result in results]

    x = range(len(misclassified_means))
    plt.figure(figsize=(10, 6))
    plt.bar(x, misclassified_means, yerr=misclassified_stds, capsize=5, alpha=0.7, color='skyblue')
    plt.xticks(ticks=x, labels=x_labels, fontsize=12, fontweight='bold', rotation=45, ha="right")
    plt.xlabel('Gene Selection', fontsize=14, fontweight='bold')
    plt.ylabel('Misclassified Samples (Mean ± Std Dev)', fontsize=14, fontweight='bold')
    plt.title('Misclassified Samples Across Gene Selections', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', linewidth=0.7)


    misclassified_plot_path = get_unique_filename(os.path.join(output_folder, "misclassified_samples_plot.png"))
    plt.savefig(misclassified_plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_gene_selections(results, output_folder="output"):
    """
    Generate and save plots for classification performance across different gene selection sizes.
    
    This function analyzes classification performance based on the number of selected genes,
    computing:
    - Mean and standard deviation of test and train accuracy.
    - Mean and standard deviation of misclassified samples.
    
    It generates:
    - A line plot comparing accuracy as a function of selected genes.
    - A bar plot showing misclassification rates.
    
    Parameters:
    -----------
    results : dict
        Dictionary where keys are the number of selected genes and values contain classification results.
    output_folder : str, optional (default="output")
        Directory where plots will be saved.
    
    Outputs:
    --------
    - PNG plot: "accuracy_vs_selected_genes.png" showing accuracy trends.
    - PNG plot: "misclassified_vs_selected_genes.png" visualizing misclassification rates.
    """

    os.makedirs(output_folder, exist_ok=True)

    top_n_values = sorted(results.keys())

    # Extract metrics
    test_means = []
    train_means = []
    test_stds = []
    train_stds = []
    misclassified_means = []
    misclassified_stds = []

    for top_n in top_n_values:
        r2_test, r2_train, _, number_wrongly_classified = results[top_n][0]  # Extract first result (gene_selection = 0)

        test_means.append(np.mean(r2_test))
        train_means.append(np.mean(r2_train))
        test_stds.append(np.std(r2_test))
        train_stds.append(np.std(r2_train))

        misclassified_means.append(np.mean(number_wrongly_classified))
        misclassified_stds.append(np.std(number_wrongly_classified))

    plt.figure(figsize=(10, 6))
    plt.errorbar(top_n_values, test_means, yerr=test_stds, label="Test Accuracy", marker="o", capsize=5, linestyle="-")
    plt.errorbar(top_n_values, train_means, yerr=train_stds, label="Train Accuracy", marker="s", capsize=5, linestyle="--")
    plt.xlabel("Number of Selected Genes", fontsize=14, fontweight="bold")
    plt.ylabel("Balanced Accuracy", fontsize=14, fontweight="bold")
    plt.title("Accuracy vs. Number of Selected Genes", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True)

    accuracy_plot_path = get_unique_filename(os.path.join(output_folder, "accuracy_vs_selected_genes.png"))
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    plt.figure(figsize=(10, 6))
    plt.bar(top_n_values, misclassified_means, yerr=misclassified_stds, capsize=5, alpha=0.7, color="skyblue")
    plt.xlabel("Number of Selected Genes", fontsize=14, fontweight="bold")
    plt.ylabel("Misclassified Samples (Mean ± Std Dev)", fontsize=14, fontweight="bold")
    plt.title("Misclassified Samples vs. Number of Selected Genes", fontsize=16, fontweight="bold")
    plt.grid(axis="y", linestyle="--", linewidth=0.7)

    misclassified_plot_path = get_unique_filename(os.path.join(output_folder, "misclassified_vs_selected_genes.png"))
    plt.savefig(misclassified_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Misclassified plot saved to {misclassified_plot_path}")


def plot_explorative_gene_selections(results, top_n=10, output_folder="output"):
    """
    Generate and save plots for exploratory gene selection analysis.
    
    This function evaluates classification performance across different subsets of selected genes
    and ranks them based on test accuracy.
    
    It generates:
    - A bar plot showing the top-ranked gene subsets with their accuracy and standard deviation.
    
    Parameters:
    -----------
    results : dict
        Dictionary where keys are gene subset tuples and values contain classification results.
    top_n : int, optional (default=10)
        Number of top-performing gene subsets to display.
    save_plot : bool, optional (default=True)
        Whether to save the plot as a PNG file.
    output_folder : str, optional (default="output")
        Directory where the plot will be saved.
    
    Outputs:
    --------
    - PNG plot: "top_N_gene_subsets.png" visualizing top gene subset performance.
    """

    subset_accuracies = []

    for gene_subset, result in results.items():
        r2_test, _, _, _ = result[0]
        mean_test_accuracy = np.mean(r2_test)
        std_test_accuracy = np.std(r2_test)
        subset_accuracies.append((gene_subset, mean_test_accuracy, std_test_accuracy))

    subset_accuracies.sort(key=lambda x: x[1], reverse=True)

    top_subsets = subset_accuracies[:top_n]
    top_labels = ["\n".join(subset[0]) for subset in top_subsets]  # Format gene names
    top_accuracies = [subset[1] for subset in top_subsets]
    top_stddevs = [subset[2] for subset in top_subsets]

    plt.figure(figsize=(12, 6))

    bars = plt.bar(range(len(top_subsets)), top_accuracies, yerr=top_stddevs, capsize=5, alpha=0.8, color="skyblue")
    plt.xlabel("Gene Subset", fontsize=14, fontweight="bold")
    plt.ylabel("Balanced Test Accuracy", fontsize=14, fontweight="bold")
    plt.title(f"Top {top_n} Gene Subsets by Accuracy", fontsize=16, fontweight="bold")
    plt.xticks(range(len(top_labels)), top_labels, rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", linewidth=0.7)

    ymin = min(top_accuracies) - max(top_stddevs) * 1.1
    ymax = max(top_accuracies) + max(top_stddevs) * 1.1
    plt.ylim(ymin, ymax)

    legend_labels = [f"{i+1}: {', '.join(top_subsets[i][0])}" for i in range(len(top_subsets))]
    plt.legend(bars, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title=f"Top {top_n} Subsets")

    os.makedirs(output_folder, exist_ok=True)
    plot_path = get_unique_filename(os.path.join(output_folder, f"top_{top_n}_gene_subsets.png"))
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    plt.show()