import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import rcParams
from gSELECT.utils import get_unique_filename
from matplotlib.ticker import FuncFormatter
import os, textwrap, itertools
import pandas as pd
import numpy as np
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and higher-level messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: add timestamp and level
)


def save_dataframe_to_csv(df, output_folder, filename):
    """
    Save a DataFrame to a CSV file in the specified output folder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_folder : str
        Directory for the CSV file.
    filename : str
        Name of the CSV file.
    """
    os.makedirs(output_folder, exist_ok=True)
    csv_path = get_unique_filename(os.path.join(output_folder, filename))
    df.to_csv(csv_path, index=False)
    logging.info(f"CSV written → {csv_path}")
    return csv_path


def save_figure_png(fig, output_folder, filename, dpi=600, bbox_inches="tight"):
    """
    Save a Matplotlib figure as a PNG file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    output_folder : str
        Directory for the PNG file.
    filename : str
        Name of the PNG file.
    dpi : int, default 600
        PNG resolution.
    bbox_inches : str, default "tight"
        Bounding box for saving.
    """
    os.makedirs(output_folder, exist_ok=True)
    png_path = get_unique_filename(os.path.join(output_folder, filename))
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    logging.info(f"PNG written → {png_path}")
    return png_path


def plot_results(
    results,
    output_folder: str = "output",
    dpi: int = 300,
    save_csv: bool = True,
    csv_name: str = "overall_statistics.csv",
    save_png: bool = True,
):
    """
    Summarise and visualise classification performance for multiple gene-selection strategies.

    Produces:
    • Line plot: Test vs Train balanced accuracy per sweep (mean ± SD).
    • Bar plot: Misclassified sample counts (mean ± SD).
    • Optional CSV of aggregated statistics.

    Parameters
    ----------
    results : list of tuples
        Each tuple: (r2_test, r2_train, gene_selection, n_misclassified).
    output_folder : str, default "output"
        Directory for output files.
    dpi : int, default 300
        PNG resolution.
    save_csv : bool, default True
        If True, save summary CSV.
    csv_name : str, default "overall_statistics.csv"
        CSV filename.
    save_png : bool, default True
        If True, save figures as PNG.
    """
    labels = {
        0: "Selected Genes",
        1: "Random Genes",
        2: "All Non-constant Genes",
    }
    colours = {k: c for k, c in zip(labels, ["#1f77b4", "#2ca02c", "#ff7f0e"])}

    os.makedirs(output_folder, exist_ok=True)

    records = []
    for r2_test, r2_train, mode, n_wrong in results:
        mode_lbl = labels.get(mode, f"Mode {mode}")
        for sweep_idx in range(len(r2_test[0])):           # assume same len
            # each element in r2_test is an array/vector across runs
            t_vals  = [run[sweep_idx] for run in r2_test]
            tr_vals = [run[sweep_idx] for run in r2_train]
            wrong   = [run[sweep_idx] for run in n_wrong]

            records.append(
                dict(
                    mode          = mode,
                    mode_label    = mode_lbl,
                    sweep         = sweep_idx,
                    test_mean     = float(np.mean(t_vals)),
                    test_std      = float(np.std(t_vals)),
                    train_mean    = float(np.mean(tr_vals)),
                    train_std     = float(np.std(tr_vals)),
                    wrong_mean    = float(np.mean(wrong)),
                    wrong_std     = float(np.std(wrong)),
                )
            )
    df = pd.DataFrame(records)

    if save_csv:
        save_dataframe_to_csv(df, output_folder, csv_name)

    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(10, fig_h))

    for mode, grp in df.groupby("mode"):
        ax.plot(
            grp.sweep,
            grp.test_mean,
            color=colours.get(mode, "grey"),
            linewidth=2,
            label=f"{labels.get(mode, mode)} (Test)",
        )
        ax.fill_between(
            grp.sweep,
            grp.test_mean - grp.test_std,
            grp.test_mean + grp.test_std,
            alpha=0.15,
            color=colours.get(mode, "grey"),
        )
        ax.plot(
            grp.sweep,
            grp.train_mean,
            color=colours.get(mode, "grey"),
            linewidth=1.5,
            linestyle="--",
            label=f"{labels.get(mode, mode)} (Train)",
        )
        ax.fill_between(
            grp.sweep,
            grp.train_mean - grp.train_std,
            grp.train_mean + grp.train_std,
            alpha=0.10,
            color=colours.get(mode, "grey"),
        )

    ax.set_xlabel("Sweep Number", fontsize=12, weight="bold")
    ax.set_ylabel("Balanced Accuracy", fontsize=12, weight="bold")
    ax.set_title("Balanced Accuracy Across Sweeps", fontsize=14, weight="bold", pad=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save_png:
        save_figure_png(fig, output_folder, "balanced_accuracy_test_train.png", dpi=dpi)
    plt.show()

    bar_data = (
        df.groupby("mode_label")
          .agg(mean_wrong=("wrong_mean", "mean"),
               std_wrong =("wrong_mean", "std"))
          .reset_index()
          .sort_values("mean_wrong", ascending=False)
    )

    fig_h = 3.5 + 0.4 * len(bar_data)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    cmap = get_cmap("Blues")
    colors = [cmap(0.3 + 0.7 * i / max(1, len(bar_data)-1))
              for i in range(len(bar_data))]

    ax.barh(
        bar_data.mode_label,
        bar_data.mean_wrong,
        xerr=bar_data.std_wrong,
        color=colors,
        alpha=0.85,
        edgecolor="none",
        capsize=4,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified Samples (mean ± SD)", fontsize=12, weight="bold")
    ax.set_title("Misclassified Samples by Gene-Selection Strategy",
                 fontsize=14, weight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    plt.tight_layout()

    if save_png:
        save_figure_png(fig, output_folder, "misclassified_samples.png", dpi=dpi)
    plt.show()



def plot_multiple_gene_selections(
    results: dict,
    output_folder: str = "output",
    dpi: int = 600,
    save_csv: bool = True,
    csv_name: str = "selection_size_summary.csv",
    save_png: bool = True,
):
    """
    Compare model performance as a function of panel size (number of selected genes).

    Produces:
    • Line plot: Balanced accuracy vs number of selected genes (mean ± SD).
    • Bar plot: Misclassified sample counts (mean ± SD).
    • Optional CSV of aggregated statistics.

    Parameters
    ----------
    results : dict
        Mapping {panel_size: list_of_run_outputs}, where each run output is a 4-tuple.
    output_folder : str, default "output"
        Directory for output files.
    dpi : int, default 600
        PNG resolution.
    save_csv : bool, default True
        If True, save summary CSV.
    csv_name : str, default "selection_size_summary.csv"
        CSV filename.
    save_png : bool, default True
        If True, save figures as PNG.
    """
    sizes = sorted(results.keys())
    summary_rows = []

    for k in sizes:
        r2_test, r2_train, _, n_wrong = results[k][0]

        summary_rows.append(
            dict(
                panel_size           = k,
                test_mean            = float(np.mean(r2_test)),
                test_std             = float(np.std(r2_test)),
                train_mean           = float(np.mean(r2_train)),
                train_std            = float(np.std(r2_train)),
                misclassified_mean   = float(np.mean(n_wrong)),
                misclassified_std    = float(np.std(n_wrong)),
            )
        )

    df = pd.DataFrame(summary_rows)
    if save_csv:
        save_dataframe_to_csv(df, output_folder, csv_name)

    def fig_ratio(n_points, base_h=4.0):
        """Return a height that grows a little with n_points."""
        return base_h + 0.15 * max(0, n_points - 8)

    fig_h = fig_ratio(len(sizes))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.errorbar(
        df.panel_size,
        df.test_mean,
        yerr=df.test_std,
        marker="o",
        capsize=4,
        linestyle="-",
        linewidth=1.6,
        label="Test",
    )
    ax.errorbar(
        df.panel_size,
        df.train_mean,
        yerr=df.train_std,
        marker="s",
        capsize=4,
        linestyle="--",
        linewidth=1.6,
        label="Train",
    )
    pad = 0.015 * (df[["test_mean", "train_mean"]].values.max()
                   - df[["test_mean", "train_mean"]].values.min())
    ymin, ymax = df[["test_mean", "train_mean"]].min().min() - pad, \
                 df[["test_mean", "train_mean"]].max().max() + pad
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Number of Selected Genes", fontsize=12, weight="bold")
    ax.set_ylabel("Balanced Accuracy",       fontsize=12, weight="bold")
    ax.set_title("Accuracy vs. Number of Selected Genes",
                 fontsize=14, weight="bold", pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    if save_png:
        save_figure_png(fig, output_folder, "accuracy_vs_selected_genes.png", dpi=dpi)
    plt.show()

    fig_h = fig_ratio(len(sizes))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = get_cmap("Blues")
    colours = [cmap(0.35 + 0.55*i/max(1, len(sizes)-1)) for i in range(len(sizes))]

    bars = ax.bar(
        df.panel_size.astype(str),
        df.misclassified_mean,
        yerr=df.misclassified_std,
        capsize=4,
        color=colours,
        alpha=0.85,
        edgecolor="none",
    )

    ax.set_xlabel("Number of Selected Genes", fontsize=12, weight="bold")
    ax.set_ylabel("Misclassified Samples",    fontsize=12, weight="bold")
    ax.set_title("Misclassified Samples vs. Number of Selected Genes",
                 fontsize=14, weight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    plt.tight_layout()

    if save_png:
        save_figure_png(fig, output_folder, "misclassified_vs_selected_genes.png", dpi=dpi)
    plt.show()


def plot_explorative_gene_selections(
    results,
    top_n=10,
    output_folder="output",
    cmap_name="Blues",
    annotate=True,
    dpi=600,
    csv_name="explorative_gene_subset_rankings.csv",
    save_csv=True,
    save_png=True,
    synonym_prefix="S"
):
    """
    Visualise top-performing gene subsets and export their statistics.

    Produces:
    • Horizontal bar chart of top subsets (mean ± SD).
    • Optional CSV ranking all subsets.

    Parameters
    ----------
    results : dict
        Mapping of gene-subset tuples to lists of run outputs.
    top_n : int, default 10
        Number of highest-ranked subsets to display.
    output_folder : str, default "output"
        Directory for output files.
    cmap_name : str, default "Blues"
        Colormap for bars.
    annotate : bool, default True
        Show “mean ± std” text beside each bar.
    dpi : int, default 600
        PNG resolution.
    csv_name : str, default "explorative_gene_subset_rankings.csv"
        CSV filename.
    save_csv : bool, default True
        If True, save ranking CSV.
    save_png : bool, default True
        If True, save figure as PNG.
    synonym_prefix : str, default "S"
        Prefix for subset codes in the legend.
    """
    rows = []
    for genes, run in results.items():
        r2_test = run[0][0] if isinstance(run[0], (tuple, list, np.ndarray)) else run[0]
        scores = np.array(r2_test)
        if scores.ndim > 1:
            scores = scores.ravel()
        rows.append({
            "genes": tuple(genes),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        })

    df = pd.DataFrame(rows).sort_values(["mean", "std"], ascending=[False, True])
    df = df[~df.duplicated(subset=["mean", "std"])].reset_index(drop=True)
    df_top = df.head(top_n)

    if save_csv:
        save_dataframe_to_csv(df, output_folder, csv_name)

    codes = [f"{synonym_prefix}{i+1}" for i in range(len(df_top))]
    code_map = dict(zip(codes, df_top["genes"]))

    fig_height = max(3.0, 0.5 + 0.45 * len(df_top))
    fig_width = 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = get_cmap(cmap_name)
    colors = [cmap(0.3 + 0.7*i/max(1, len(df_top)-1)) for i in range(len(df_top))]

    x_vals = df_top["mean"]
    y_vals = np.arange(len(df_top))

    bars = ax.barh(
        y_vals, x_vals,
        height=0.5,
        color=colors, edgecolor="none", alpha=0.85, zorder=3
    )
    mu_vals = df_top["mean"].values
    std_vals = df_top["std"].clip(upper=0.015)
    ax.errorbar(
        mu_vals,           # actual mean values
        y_vals,
        xerr=std_vals,
        fmt="none",
        ecolor="black",
        elinewidth=0.8,
        capsize=3,
        alpha=0.7,
        zorder=2
    )
    if annotate:
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        padding = 0.001 * x_range  # dynamic padding
    
        for y, mu, sd in zip(y_vals, df_top["mean"], df_top["std"]):
            if sd == 0:
                continue  # optionally skip 0-error bars
            x_txt = mu + sd + padding
            ax.text(
                x_txt, y,
                f"{mu:.3f} ± {sd:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                color="black",
                clip_on=True
            )
    ax.set_yticks(y_vals)
    ax.set_yticklabels(codes, fontfamily="monospace", fontsize=9)
    ax.invert_yaxis()

    pad = 0.01 * (x_vals.max() - x_vals.min() if x_vals.max() != x_vals.min() else 1)
    x_min = (x_vals - std_vals).min()
    x_max = (x_vals + std_vals).max()
    span = x_max - x_min
    zoom_ratio = 0.2  # 20% margin on each side of the true range

    ax.set_xlim(x_min - span * zoom_ratio, x_max + span * zoom_ratio)
    ax.set_xlabel("Balanced Test Accuracy", fontsize=11, weight="bold")
    ax.set_title(f"Top {len(df_top)} Gene Subsets", fontsize=14, weight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    legend_text = "\n".join([
        f"{code}: {', '.join(genes)}" for code, genes in code_map.items()
    ])
    plt.figtext(0.01, -0.02, "Gene Subset Legend:\n" + legend_text,
                ha="left", va="top", fontsize=9, fontfamily="monospace",
                wrap=True, linespacing=1.3)

    plt.tight_layout(rect=(0, 0.18, 1, 1))

    if save_png:
        save_figure_png(fig, output_folder, "explorative_gene_subset_rankings.png", dpi=dpi)
    plt.show()


def plot_all_genes(
    results,
    output_folder="output",
    dpi=600,
    csv_name="all_genes_rankings.csv",
    save_csv=True,
    save_png=True
):
    """
    Visualise performance metrics for all gene-selection strategies.

    Produces:
    • Line plot: Test vs Train balanced accuracy across sweeps.
    • Bar plot: Misclassified sample counts (mean ± SD).
    • Optional CSV of summary statistics.

    Parameters
    ----------
    results : list of tuples
        Each tuple: (r2_test, r2_train, mode, n_misclassified).
    output_folder : str, default "output"
        Directory for output files.
    dpi : int, default 600
        PNG resolution.
    csv_name : str, default "all_genes_rankings.csv"
        CSV filename.
    save_csv : bool, default True
        If True, save summary CSV.
    save_png : bool, default True
        If True, save figures as PNG.
    """
    os.makedirs(output_folder, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid")
    palette = sns.color_palette("deep")

    labels_map = {
        0: "Top-MI genes",
        1: "Random genes",
        2: "All non-constant genes"
    }
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))

    for idx, (r2_test, r2_train, mode, _) in enumerate(results):
        sweeps = np.arange(r2_test.shape[1])
        color = palette[idx % len(palette)]
        label_core = labels_map.get(mode, f"Mode {mode}")

        ax1.plot(
            sweeps, r2_test.squeeze(), lw=2.5, color=color,
            label=f"{label_core} (Test)"
        )
        ax1.plot(
            sweeps, r2_train.squeeze(), lw=2.0, ls="--", color=color,
            label=f"{label_core} (Train)"
        )

    rows = []
    for r2_test, _, mode, _ in results:
        scores = np.array(r2_test)
        if scores.ndim > 1:
            scores = scores.ravel()
        rows.append({
            "mode": mode,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        })
    df = pd.DataFrame(rows).sort_values(["mean", "std"], ascending=[False, True])
    df = df.drop_duplicates(subset=["mean", "std"]).reset_index(drop=True)
    if save_csv:
        save_dataframe_to_csv(df, output_folder, csv_name)

    ax1.set_xlabel("Sweep", labelpad=8, fontweight='bold')
    ax1.set_ylabel("Balanced Accuracy", fontweight='bold')
    
    #set dynamic y-limits
    y_min = min(df["mean"].min() - df["std"].max(), 0.0)
    y_max = max(df["mean"].max() + df["std"].max(), 1.1)
    ax1.set_ylim(y_min, y_max)

    ax1.set_title("Balanced Accuracy Over Sweeps", fontweight='bold', fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

    ax1.legend(
        frameon=False,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35)
    )
    sns.despine(ax=ax1)
    fig1.tight_layout()

    if save_png:
        save_figure_png(fig1, output_folder, "accuracy_vs_sweeps.png", dpi=dpi)
    plt.show()

    means = [nw.mean() for *_, nw in results]
    stds = [nw.std() for *_, nw in results]
    x = np.arange(len(means))
    labels = [labels_map.get(r[2], f"Mode {r[2]}") for r in results]

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    bars = ax2.bar(
        x, means, yerr=stds, capsize=5, width=0.6,
        color=sns.color_palette("deep", n_colors=len(means))
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight='bold')
    ax2.set_ylabel("Misclassified Samples (Mean ± SD)", fontweight='bold')
    ax2.set_title("Misclassified Samples per Strategy", fontweight='bold', fontsize=14)

    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax2)
    fig2.tight_layout()

    if save_png:
        save_figure_png(fig2, output_folder, "misclassified_samples.png", dpi=dpi)
    plt.show()
