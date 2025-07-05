import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import rcParams
from gSELECT.utils import get_unique_filename
import os, textwrap, itertools
import pandas as pd
import numpy as np


def plot_results(
    results,
    output_folder: str = "output",
    dpi: int = 300,
    save_csv: bool = True,
    csv_name: str = "overall_statistics.csv",
    save_png: bool = True,
):
    """
    Summarise and visualise classification performance for several *gene-selection
    strategies* (selected genes, random genes, all non-constant genes, …).

    The function

    1. aggregates **balanced‐accuracy** and **misclassification** metrics from
       ``results``,
    2. writes a tidy CSV table (`save_csv=True`),
    3. produces two publication-ready figures:

       * a **line plot** of *Test* vs *Train* balanced accuracy **per sweep**
         (mean ± 1 SD shading) for each strategy,
       * a **bar plot** of misclassified-sample counts (mean ± SD).

    Parameters
    ----------
    results : list[tuple]
        Each entry must be a 4-tuple  
        ``(r2_test, r2_train, gene_selection, n_misclassified)``  
        where:

        * ``r2_test``  – iterable of balanced-accuracy values (per sweep, per run)\
        * ``r2_train`` – iterable of balanced-accuracy values (per sweep, per run)\
        * ``gene_selection`` – {0, 1, 2, …} identifying the strategy\
        * ``n_misclassified`` – iterable (per sweep, per run)

        Replicates with the **same** ``gene_selection`` code are pooled.
    output_folder : str, default ``"output"``
        Destination directory for all artefacts.
    dpi : int, default ``300``
        Resolution of the saved PNGs.
    save_csv : bool, default ``True``
        If *True*, write the aggregated statistics to CSV.
    csv_name : str, default ``"overall_statistics.csv"``
        File name for the CSV (inside *output_folder*).
    save_png : bool, default ``True``
        If *True*, save the two figures as PNGs.

    Files written
    -------------
    • *overall_statistics.csv* (optional)  
    • *balanced_accuracy_test_train.png*  
    • *misclassified_samples.png*
    """
    # ------------------------------------------------------------------ #
    # 0. Helpers                                                          #
    # ------------------------------------------------------------------ #
    def _unique(path):
        """Append index if a file already exists."""
        i, stem, ext = 1, *os.path.splitext(path)
        while os.path.exists(path):
            path = f"{stem}_{i}{ext}"
            i += 1
        return path

    labels = {
        0: "Selected Genes",
        1: "Random Genes",
        2: "All Non-constant Genes",
    }
    colours = {k: c for k, c in zip(labels, ["#1f77b4", "#2ca02c", "#ff7f0e"])}

    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Pool replicates → long DataFrame                                 #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. CSV summary (mean / std collapsed over sweeps)                  #
    # ------------------------------------------------------------------ #
    if save_csv:
        summary = (
            df.groupby("mode_label")
              .agg(test_mean=("test_mean", "mean"),
                   test_std =("test_mean", "std"),
                   train_mean=("train_mean", "mean"),
                   train_std =("train_mean", "std"),
                   wrong_mean=("wrong_mean", "mean"),
                   wrong_std =("wrong_mean", "std"))
              .reset_index()
        )
        csv_path = _unique(os.path.join(output_folder, csv_name))
        summary.to_csv(csv_path, index=False)
        print(f"CSV written → {csv_path}")

    # ------------------------------------------------------------------ #
    # 3. Balanced-accuracy line plot                                     #
    # ------------------------------------------------------------------ #
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
        acc_png = _unique(os.path.join(output_folder, "balanced_accuracy_test_train.png"))
        plt.savefig(acc_png, dpi=dpi)
        print(f"PNG written → {acc_png}")
    plt.show()

    # ------------------------------------------------------------------ #
    # 4. Misclassified-samples bar plot                                  #
    # ------------------------------------------------------------------ #
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
        mis_png = _unique(os.path.join(output_folder, "misclassified_samples.png"))
        plt.savefig(mis_png, dpi=dpi)
        print(f"PNG written → {mis_png}")
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
    Compare model performance as a function of *panel size* (number of selected
    genes) and export both a line chart (accuracy) and a bar chart
    (misclassification counts).

    Parameters
    ----------
    results : dict
        Mapping ``{ panel_size : list_of_run_outputs }`` where
        ``run_outputs[0]`` is a 4-tuple like
        ``(r2_test, r2_train, _, n_misclassified)``.
    output_folder : str, default "output"
        Destination for the generated files.
    dpi : int, default 600
        Resolution of the PNGs.
    save_csv : bool, default True
        If ``True``, write a CSV containing the aggregated statistics.
    csv_name : str, default "selection_size_summary.csv"
        Filename for the CSV.
    save_png : bool, default True
        If ``True``, save the PNGs in *output_folder*.

    Files written
    -------------
    • *accuracy_vs_selected_genes.png*  
    • *misclassified_vs_selected_genes.png*  
    • *selection_size_summary.csv* (optional)

    Notes
    -----
    • The line plot auto-scales its y-axis (small margin top & bottom).  
    • Bars in the misclassification plot are coloured by a sequential colormap
      for visual ranking.  
    • Both figures have a 16 : 9 aspect ratio that adapts in height to the
      number of points so labels never crowd.
    """
    # ------------------------------------------------------------------ #
    # 1. Gather summary statistics                                       #
    # ------------------------------------------------------------------ #
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

    # save CSV -----------------------------------------------------------
    if save_csv:
        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"CSV written → {csv_path}")

    # ------------------------------------------------------------------ #
    # 2. Dynamic figure sizing helper                                    #
    # ------------------------------------------------------------------ #
    def fig_ratio(n_points, base_h=4.0):
        """Return a height that grows a little with n_points."""
        return base_h + 0.15 * max(0, n_points - 8)

    # ------------------------------------------------------------------ #
    # 3. Accuracy line plot                                              #
    # ------------------------------------------------------------------ #
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

    # aesthetics
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
        os.makedirs(output_folder, exist_ok=True)
        acc_png = os.path.join(output_folder, "accuracy_vs_selected_genes.png")
        plt.savefig(acc_png, dpi=dpi)
        print(f"PNG written → {acc_png}")

    plt.show()

    # ------------------------------------------------------------------ #
    # 4. Misclassification bar plot                                      #
    # ------------------------------------------------------------------ #
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
        mis_png = os.path.join(output_folder, "misclassified_vs_selected_genes.png")
        plt.savefig(mis_png, dpi=dpi)
        print(f"PNG written → {mis_png}")

    plt.show()


def plot_explorative_gene_selections(
    results,
    top_n=10,
    output_folder="output",
    cmap_name="Blues",
    show_delta=False,
    annotate=True,
    dpi=600,
    csv_name="explorative_gene_subset_rankings.csv",
    save_csv=True,
    save_png=True,
    legend_fontsize=9,
    synonym_prefix="S"
):
    """
    Visualise the best-performing gene-subset models and (optionally) export
    their statistics.

    The function
    1. computes *mean* ± *std* of balanced-test accuracy for every subset in
       ``results``,
    2. selects the top-scoring ``top_n`` unique subsets,
    3. renders them as a horizontal bar chart with automatic figure sizing,
       left-margin calculation and per-row spacing that adapts to
       *arbitrarily* long gene IDs and *any* number of subsets,
    4. optionally writes a tidy CSV and/or a high-resolution PNG.

    Results dictionary format
    -------------------------
    ``results = { (gene_id₁, gene_id₂, …): list_of_runs, … }``

    Each *run* is an iterable whose first element (``run[0][0]``) is an array-
    like vector of ``r2_test`` values used to derive the mean and standard
    deviation.

    Parameters
    ----------
    results : dict
        Mapping of gene-subset tuples to lists of run outputs.
    top_n : int, default 10
        Number of highest-ranked subsets to display.
    output_folder : str, default "output"
        Destination directory for the PNG and/or CSV files.
    wrap_width : int, default 14
        Soft-wrap width (in characters) for breaking long gene IDs in the
        y-axis label.
    cmap_name : str, default "Blues"
        Name of a sequential Matplotlib colormap used for the bars.
    show_delta : bool, default False
        If ``True``, plot Δ-accuracy relative to the worst of the displayed
        subsets (useful when absolute accuracies cluster tightly).
    annotate : bool, default True
        Show “mean ± std” text to the right of each bar.
    dpi : int, default 600
        Resolution of the saved PNG.
    csv_name : str, default "explorative_gene_subset_rankings.csv"
        Filename for the CSV that ranks *all* subsets (written only when
        ``save_csv`` is ``True``).
    save_csv : bool, default True
        Write the ranked-subset DataFrame to CSV.
    save_png : bool, default True
        Save the plot as PNG.

    Returns
    -------
    None
        The figure is displayed inline and, depending on the ``save_*`` flags,
        files are written to *output_folder*.

    Notes
    -----
    • Subsets with identical (mean, std) pairs are de-duplicated before
      ranking, ensuring each bar is unique.  
    • Figure height scales with the total number of wrapped text lines;
      left margin is computed from the rendered pixel width of the longest
      label, so even 50 × 4-line labels remain readable.  
    • Bar height consumes 90 % of the vertical slot allocated to each row,
      producing equal visual rhythm regardless of label length.
    """
    # 1. Aggregate stats
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

    # Save CSV if needed
    if save_csv:
        os.makedirs(output_folder, exist_ok=True)
        df.assign(gene_subset=df.genes.apply(lambda g: ";".join(g))) \
          .drop(columns="genes") \
          .to_csv(os.path.join(output_folder, csv_name), index=False)

    # 2. Assign short codes (S1, S2, ...)
    codes = [f"{synonym_prefix}{i+1}" for i in range(len(df_top))]
    code_map = dict(zip(codes, df_top["genes"]))

    # 3. Plot setup
    fig_height = max(3.0, 0.5 + 0.45 * len(df_top))
    fig_width = 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = get_cmap(cmap_name)
    colors = [cmap(0.3 + 0.7*i/max(1, len(df_top)-1)) for i in range(len(df_top))]

    worst = df_top["mean"].min()
    x_vals = df_top["mean"] if not show_delta else df_top["mean"] - worst
    y_vals = np.arange(len(df_top))

    bars = ax.barh(
        y_vals, x_vals,
        height=0.5,
        color=colors, edgecolor="none", alpha=0.85, zorder=3
    )

    # 4. Correct error bars (centered, visually tight)
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

    # 5. Annotations
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

    # 6. Y-ticks
    ax.set_yticks(y_vals)
    ax.set_yticklabels(codes, fontfamily="monospace", fontsize=9)
    ax.invert_yaxis()

    pad = 0.01 * (x_vals.max() - x_vals.min() if x_vals.max() != x_vals.min() else 1)
    x_min = (x_vals - std_vals).min()
    x_max = (x_vals + std_vals).max()
    span = x_max - x_min
    zoom_ratio = 0.2  # 20% margin on each side of the true range

    ax.set_xlim(x_min - span * zoom_ratio, x_max + span * zoom_ratio)
    ax.set_xlabel("Δ Balanced Test Accuracy" if show_delta else "Balanced Test Accuracy",
                  fontsize=11, weight="bold")

    ax.set_title(f"Top {len(df_top)} Gene Subsets", fontsize=14, weight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # 7. Gene legend below
    legend_text = "\n".join([
        f"{code}: {', '.join(genes)}" for code, genes in code_map.items()
    ])
    plt.figtext(0.01, -0.02, "Gene Subset Legend:\n" + legend_text,
                ha="left", va="top", fontsize=legend_fontsize, fontfamily="monospace",
                wrap=True, linespacing=1.3)

    plt.tight_layout(rect=(0, 0.18, 1, 1))

    # 8. Save
    if save_png:
        os.makedirs(output_folder, exist_ok=True)
        base = os.path.join(output_folder, f"top_{len(df_top)}_gene_subsets.png")
        fname = base
        ctr = 1
        while os.path.exists(fname):
            root, ext = os.path.splitext(base)
            fname = f"{root}_{ctr}{ext}"
            ctr += 1
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        print(f"PNG written → {fname}")

    plt.show()