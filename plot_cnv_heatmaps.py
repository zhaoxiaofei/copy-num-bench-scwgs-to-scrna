#!/usr/bin/env python3

"""
Plot heatmaps of CNV caller benchmarking results, and emit a single per-dataset
summary table.

Each heatmap corresponds to one performance metric.
Rows   = datasets  (e.g. BCIS106T_chip1_SAMN48409192_SRR33511671)
Columns = methods   (e.g. copykat_predict, conicsmat, numbat, …)

Cell colour  = mean value across all cells in that dataset×method pair.
Cell text    = "mean ± sd".

Summary table
-------------
Alongside the figures the script writes ONE combined TSV, dataset_summary.tsv,
with one row per dataset and exactly these columns:

    dataset
    n_cells_total
    tumor_purity_cellfraction__dna
    tumor_sample_ploidy_mean__dna
    scRNA_reads_per_cell_mean
    scRNA_reads_per_cell_median
    scWGS_reads_per_cell_mean
    scWGS_reads_per_cell_median

The values are taken from each dataset's own dataset_metrics.tsv (written by
compute_dataset_metrics.py). Those file paths are NOT passed in — they are
inferred from --input_glob: for every evaluation directory the glob touches, the
sibling file <evaluation_dir>/dataset_metrics.tsv is read. Columns absent from a
given file become empty cells; a dataset whose dataset_metrics.tsv is missing is
skipped with a warning.

Usage
-----
    python plot_cnv_heatmaps.py \
        --input_glob "/nfs/wxz/zxf/cnv/copy-num-bench-scwgs-to-scrna/results/BCI*solo-genefull_output/evaluation/*.tsv" \
        --outdir ./heatmaps \
        [--metrics "Pearson Correlation Coefficient,CopyNumber gain F-score"] \
        [--file_pattern "without_preclassified_cells"] \
        [--summary_out ./heatmaps/dataset_summary.tsv]
"""

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Per-dataset summary table specification
# ---------------------------------------------------------------------------
# Columns are taken verbatim from each dataset's dataset_metrics.tsv. Order is
# preserved in the output; a column missing from a given file becomes NaN.
SUMMARY_COLUMNS = [
    "dataset",
    "n_cells_total",
    "tumor_purity_cellfraction__dna",
    "tumor_sample_ploidy_mean__dna",
    "scRNA_reads_per_cell_mean",
    "scRNA_reads_per_cell_median",
    "scWGS_reads_per_cell_mean",
    "scWGS_reads_per_cell_median",
]
# The per-dataset metrics file that lives next to the benchmark eval TSVs.
DATASET_METRICS_BASENAME = "dataset_metrics.tsv"
# Descriptive / non-benchmark files that share the evaluation directory and must
# NOT be fed into the heatmap loader.
NON_BENCHMARK_BASENAMES = {
    DATASET_METRICS_BASENAME,
    "per_cell_metrics.tsv",
    "annotation_agreement.tsv",
    "annotation_discordant_cells.tsv",
    "cnv_event_sizes.tsv",
    "dataset_summary.tsv",
}


# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CNV benchmark heatmaps with dispersion", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--input_glob",
        default="/nfs/wxz/zxf/cnv/copy-num-bench-scwgs-to-scrna/results/*solo-genefull_output/evaluation/*.tsv",
        help="Glob pattern for evaluation TSV files. The per-dataset "
             "dataset_metrics.tsv files used for the summary table are inferred "
             "from the evaluation directories this glob matches.",
    )
    p.add_argument("--outdir", default="./heatmaps", help="Output directory for figures")
    p.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated list of metrics to plot (default: all found metrics)",
    )
    p.add_argument(
        "--file_pattern",
        default=None,
        help="Only include files whose basename contains this substring "
             "(e.g. 'without_preclassified_cells'). Default: include all.",
    )
    p.add_argument(
        "--summary_out",
        default=None,
        help="Path for the single combined summary TSV. "
             "Default: <outdir>/dataset_summary.tsv",
    )
    p.add_argument(
        "--fmt", default="pdf", help="Output figure format: pdf, png, svg (default: pdf)"
    )
    p.add_argument("--dpi", type=int, default=200, help="DPI for raster formats")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Extract dataset name from file path
# ---------------------------------------------------------------------------
def dataset_from_path(filepath: str) -> str:
    """
    Pull the dataset identifier from the directory structure.
    E.g. .../BCIS106T_chip1_SAMN48409192_SRR33511671_solo-genefull_output/...
         → BCIS106T_chip1_SAMN48409192_SRR33511671
    """
    parts = Path(filepath).parts
    for part in parts:
        m = re.match(r"^(\S+?)_solo-genefull_output$", part)
        if m:
            return m.group(1)
    # fallback: use parent directory name minus _solo-genefull_output
    parent = Path(filepath).parent.parent.name
    return parent.replace("_solo-genefull_output", "")


# ---------------------------------------------------------------------------
# 3. Load and concatenate all TSV files (for the heatmaps)
# ---------------------------------------------------------------------------
def load_all(glob_pattern: str, file_pattern) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern, recursive=True))
    if not files:
        sys.exit(f"No files matched: {glob_pattern}")

    # Skip cell_classification files — different schema
    files = [f for f in files if "cell_classification" not in os.path.basename(f)]
    # Skip the descriptive / summary files — different schema
    files = [f for f in files if os.path.basename(f) not in NON_BENCHMARK_BASENAMES]

    if file_pattern:
        files = [f for f in files if file_pattern in os.path.basename(f)]

    print(f"Loading {len(files)} evaluation files …")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
            continue

        # Ensure the expected columns exist
        if "metric" not in df.columns or "value" not in df.columns or "caller" not in df.columns:
            continue

        df["dataset"] = dataset_from_path(fp)
        df["source_file"] = os.path.basename(fp)

        # Resolve a per-row 'final_celltype'.
        #   - If the file distinguishes >1 celltype, trust its per-row 'celltype'.
        #   - If there is only one (or none), fall back to the DNA-derived label
        #     'celltype_dna' when present.
        # Both columns are optional in some eval files, so every access is guarded
        # rather than assumed (a missing column previously raised KeyError, and a
        # file with 3+ celltypes tripped an assert).
        has_celltype = "celltype" in df.columns
        has_celltype_dna = "celltype_dna" in df.columns
        n_celltypes = df["celltype"].nunique(dropna=True) if has_celltype else 0

        if n_celltypes > 1 and has_celltype:
            df["final_celltype"] = df["celltype"]
        elif has_celltype_dna:
            df["final_celltype"] = df["celltype_dna"]
        elif has_celltype:
            df["final_celltype"] = df["celltype"]
        else:
            df["final_celltype"] = np.nan

        frames.append(df)

    if not frames:
        sys.exit("No usable data after loading.")

    data = pd.concat(frames, ignore_index=True)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    # Normalise caller names: strip suffixes like _predict, _predict_hg19 → keep full name
    # but unify duplicates where 'with' vs 'without' preclassified gives same caller string
    data["method"] = data["caller"].str.strip()

    print(f"  {len(data)} rows, {data['dataset'].nunique()} datasets, "
          f"{data['method'].nunique()} methods, {data['metric'].nunique()} metrics")
    return data


# ---------------------------------------------------------------------------
# 3b. Single combined summary table (inferred dataset_metrics.tsv files)
# ---------------------------------------------------------------------------
def infer_metrics_files(input_glob: str):
    """Infer the per-dataset dataset_metrics.tsv paths from --input_glob.

    The benchmark eval TSVs live in each dataset's evaluation/ directory; the
    matching dataset_metrics.tsv is the sibling file in that same directory.
    We expand the glob, collect the unique parent directories, and look for
    dataset_metrics.tsv in each. Returns a list ordered by dataset name.
    """
    matches = sorted(glob.glob(input_glob, recursive=True))
    eval_dirs = []
    seen_dirs = set()
    for m in matches:
        d = os.path.dirname(m)
        if d not in seen_dirs:
            seen_dirs.add(d)
            eval_dirs.append(d)

    # If the glob itself pointed straight at dataset_metrics.tsv files, honour those.
    direct = [m for m in matches
              if os.path.basename(m) == DATASET_METRICS_BASENAME]

    metrics_files = list(direct)
    for d in eval_dirs:
        cand = os.path.join(d, DATASET_METRICS_BASENAME)
        if os.path.exists(cand):
            metrics_files.append(cand)

    # De-duplicate (keep first occurrence) and drop non-existent.
    seen, out = set(), []
    for f in metrics_files:
        if f not in seen and os.path.exists(f):
            seen.add(f)
            out.append(f)
    # Order deterministically by inferred dataset name, then path.
    out.sort(key=lambda f: (dataset_from_path(f), f))
    return out


def build_dataset_summary(metrics_files) -> pd.DataFrame:
    """Read each dataset_metrics.tsv and select SUMMARY_COLUMNS into one table.

    - Missing columns are filled with NaN (kept in the requested order).
    - The `dataset` column is recovered from the file path if absent/blank.
    - dataset_metrics.tsv holds one row per dataset; rows are concatenated and
      sorted by dataset name. Duplicate datasets (same name) are collapsed,
      keeping the first non-null value per column.
    """
    rows = []
    for fp in metrics_files:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"  [summary] SKIP {fp}: {e}")
            continue

        if df.empty:
            df = pd.DataFrame([{}])

        if "dataset" not in df.columns or df["dataset"].isna().all():
            df["dataset"] = dataset_from_path(fp)

        df = df.reindex(columns=SUMMARY_COLUMNS)
        if df["dataset"].isna().all():
            df["dataset"] = dataset_from_path(fp)

        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    out = pd.concat(rows, ignore_index=True)

    # Collapse any accidental duplicate dataset rows: first non-null wins.
    if out["dataset"].duplicated().any():
        out = (out.groupby("dataset", as_index=False, sort=False)
                  .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))

    out = out.sort_values("dataset", kind="stable").reset_index(drop=True)
    return out[SUMMARY_COLUMNS]


def write_dataset_summary(input_glob: str, summary_out: str) -> pd.DataFrame:
    """Infer inputs, build the single combined table, and write it to disk.

    The file is always written (header-only if no dataset_metrics.tsv is found),
    so the output reliably exists for downstream steps.
    """
    metrics_files = infer_metrics_files(input_glob)
    if metrics_files:
        print(f"Building summary from {len(metrics_files)} "
              f"{DATASET_METRICS_BASENAME} file(s) inferred from --input_glob …")
    else:
        print(f"[summary] WARNING: no {DATASET_METRICS_BASENAME} found next to the "
              f"files matched by {input_glob!r}; writing header-only summary.")

    summary = build_dataset_summary(metrics_files)
    os.makedirs(os.path.dirname(summary_out) or ".", exist_ok=True)
    summary.to_csv(summary_out, sep="\t", index=False, na_rep='nan')
    print(f"Dataset summary ({len(summary)} dataset(s)) → {summary_out}")
    return summary


# ---------------------------------------------------------------------------
# 4. Aggregate: mean, Q1, Q3, std per (dataset, method, metric)
# ---------------------------------------------------------------------------
def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    agg = (
        data.groupby(["dataset", "method", "metric"])["value"]
        .agg(["mean", "std", "count",
               lambda x: np.nanpercentile(x, 25),
               lambda x: np.nanpercentile(x, 75)])
        .reset_index()
    )
    agg.columns = ["dataset", "method", "metric", "mean", "std", "n", "q1", "q3"]
    agg["iqr"] = agg["q3"] - agg["q1"]
    return agg


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

# A diverging palette centred on 0.5 for metrics in [0,1]; a diverging palette
# centred on 0 for correlations that can go negative.
CORRELATION_METRICS = {
    "Pearson Correlation Coefficient",
    "Spearman Correlation Coefficient",
}


def short_dataset_label(name: str) -> str:
    """Shorten dataset names for axis labels when they are very long."""
    # Try to extract the core identifier
    # BCIS106T_chip1_SAMN48409192_SRR33511671 → BCIS106T_chip1
    parts = name.split("_")
    if len(parts) >= 2:
        # keep first two tokens, abbreviated SRR
        sample = parts[0]
        chip = parts[1] if len(parts) > 1 else ""
        srr = ""
        for p in parts:
            if p.startswith("SRR"):
                srr = p[-4:]  # last 4 digits
                break
        label = f"{sample}_{chip}"
        if srr:
            label += f"_…{srr}"
        return label
    return name


def plot_heatmap(agg: pd.DataFrame, metric: str, outdir: str, fmt: str, dpi: int):
    """
    Draw one figure per metric.

    The heatmap cell is coloured by the mean value.
    Inside each cell we annotate the mean ± sd.
    """
    sub = agg[agg["metric"] == metric].copy()
    if sub.empty:
        return

    # Pivot to matrix form
    pivot_mean = sub.pivot_table(index="dataset", columns="method", values="mean")
    pivot_q1   = sub.pivot_table(index="dataset", columns="method", values="q1")
    pivot_q3   = sub.pivot_table(index="dataset", columns="method", values="q3")
    pivot_n    = sub.pivot_table(index="dataset", columns="method", values="n")
    pivot_std  = sub.pivot_table(index="dataset", columns="method", values="std")

    if pivot_mean.empty:
        return

    # Sort rows and columns alphabetically for consistency
    pivot_mean = pivot_mean.sort_index(axis=0).sort_index(axis=1)
    pivot_q1   = pivot_q1.reindex_like(pivot_mean)
    pivot_q3   = pivot_q3.reindex_like(pivot_mean)
    pivot_n    = pivot_n.reindex_like(pivot_mean)
    pivot_std  = pivot_std.reindex_like(pivot_mean)

    n_rows, n_cols = pivot_mean.shape

    # --- Choose colourmap ---
    if metric in CORRELATION_METRICS:
        cmap = "RdBu_r"
        vmin, vmax = -1, 1
        center = 0
    elif "ROC-AUC" in metric:
        cmap = "RdYlGn"
        vmin, vmax = 0, 1
        center = 0.5
    elif "Fraction" in metric:
        cmap = "YlGnBu"
        vmin, vmax = 0, 1
        center = None
    else:
        # precision / recall / F-score / accuracy → 0 to 1
        cmap = "YlOrRd"
        vmin, vmax = 0, 1
        center = None

    # --- Build annotation strings ---
    annot = np.full_like(pivot_mean, "", dtype=object)
    for i, ds in enumerate(pivot_mean.index):
        for j, mt in enumerate(pivot_mean.columns):
            m = pivot_mean.iloc[i, j]
            q1 = pivot_q1.iloc[i, j]
            q3 = pivot_q3.iloc[i, j]
            sd = pivot_std.iloc[i, j]
            n  = pivot_n.iloc[i, j]
            if np.isnan(m):
                annot[i, j] = ""
            else:
                # Compact annotation: mean ± sd
                annot[i, j] = (
                    f"{m:.2f} ±{sd:.2f}\n"
                    #f"[{q1:.2f},{q3:.2f}]"
                )

    # --- Figure size ---
    cell_w = max(0.75, min(2.4, 20 / max(n_cols, 1)))
    cell_h = max(0.25, min(1.4, 14 / max(n_rows, 1)))
    fig_w = cell_w * n_cols + 4.5   # extra room for labels + colourbar
    fig_h = cell_h * n_rows + 3.0
    fig_w = max(fig_w, 8)
    fig_h = max(fig_h, 5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- Short labels ---
    # short_rows = [short_dataset_label(d) for d in pivot_mean.index]
    short_rows = [d for d in pivot_mean.index]

    sns.heatmap(
        pivot_mean,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.8,
        linecolor="#e0e0e0",
        annot_kws={"fontsize": 7, "fontfamily": "monospace", "va": "center"},
        cbar_kws={"label": "Mean value", "shrink": 0.75},
        yticklabels=short_rows,
    )

    ax.set_title(metric, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    '''
    # --- Draw mini IQR bars inside each cell ---
    # Map data range → cell coordinate [0,1] within each cell
    for i, ds in enumerate(pivot_mean.index):
        for j, mt in enumerate(pivot_mean.columns):
            m  = pivot_mean.iloc[i, j]
            q1 = pivot_q1.iloc[i, j]
            q3 = pivot_q3.iloc[i, j]
            if np.isnan(m):
                continue

            # Cell centre in data coordinates: (j+0.5, i+0.5)
            cx = j + 0.5
            cy = i + 0.82  # slightly below centre of annotation text

            # Scale bar length: map value range [vmin,vmax] → [-0.4, 0.4] of cell width
            def _map(v):
                return (v - vmin) / (vmax - vmin) * 0.8 - 0.4

            x_q1 = cx + _map(q1)
            x_q3 = cx + _map(q3)
            x_m  = cx + _map(m)

            bar_color = "#333333" if m > (vmin + vmax) / 2 * 0.6 else "#eeeeee"
            alpha = 0.55

            # IQR bar
            ax.plot([x_q1, x_q3], [cy, cy], color=bar_color, linewidth=2.5,
                    solid_capstyle="round", alpha=alpha, zorder=3)
            # Mean tick
            ax.plot([x_m], [cy], marker="|", color=bar_color, markersize=6,
                    markeredgewidth=1.8, alpha=alpha, zorder=4)
    '''
    fig.tight_layout()

    safe_name = re.sub(r"[^\w]+", "_", metric).strip("_")
    out_path = os.path.join(outdir, f"heatmap_{safe_name}.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# 6. Summary overview: one compact figure with all metrics side-by-side
#    (mean across datasets for each method)
# ---------------------------------------------------------------------------
def plot_overview(agg: pd.DataFrame, outdir: str, fmt: str, dpi: int):
    """Grand-mean per method for every metric — a single overview heatmap."""
    overview = (
        agg.groupby(["method", "metric"])["mean"]
        .mean()
        .reset_index()
        .pivot_table(index="metric", columns="method", values="mean")
    )
    if overview.empty:
        return

    overview = overview.sort_index(axis=0).sort_index(axis=1)
    n_rows, n_cols = overview.shape

    fig_w = max(10, n_cols * 1.5 + 3)
    fig_h = max(6, n_rows * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        overview,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.5,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Grand mean", "shrink": 0.6},
    )
    ax.set_title("Overview: grand-mean per method across all datasets",
                  fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Metric", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()

    out_path = os.path.join(outdir, f"overview_grand_mean.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Single combined summary table ---
    # Built first (and independently of the heatmap data) so it is produced even
    # if the benchmark eval TSVs are unusable. The dataset_metrics.tsv inputs are
    # inferred from --input_glob; no separate path argument is needed.
    summary_out = args.summary_out or os.path.join(args.outdir, "dataset_summary.tsv")
    write_dataset_summary(args.input_glob, summary_out)

    # --- Heatmaps (original behaviour) ---
    data = load_all(args.input_glob, args.file_pattern)
    agg  = aggregate(data)

    metrics_to_plot = sorted(data["metric"].unique())
    if args.metrics:
        requested = [m.strip() for m in args.metrics.split(",")]
        metrics_to_plot = [m for m in metrics_to_plot if m in requested]
        if not metrics_to_plot:
            sys.exit(f"None of the requested metrics found. Available:\n"
                     f"  {sorted(data['metric'].unique())}")

    print(f"\nPlotting {len(metrics_to_plot)} metric heatmaps …")
    for metric in metrics_to_plot:
        plot_heatmap(agg, metric, args.outdir, args.fmt, args.dpi)

    print("\nPlotting overview …")
    plot_overview(agg, args.outdir, args.fmt, args.dpi)

    # Also save the aggregated table as TSV for downstream use
    agg_path = os.path.join(args.outdir, "aggregated_results.tsv")
    agg.to_csv(agg_path, sep="\t", index=False, na_rep='nan')
    print(f"\nAggregated data → {agg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

