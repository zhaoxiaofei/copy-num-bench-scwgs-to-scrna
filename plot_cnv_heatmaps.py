#!/usr/bin/env python3

"""
Plot heatmaps of CNV caller benchmarking results, and emit a single per-dataset
summary table.

Each heatmap corresponds to one performance metric.
Rows   = datasets  (e.g. BCIS106T_chip1_SAMN48409192_SRR33511671)
Columns = methods   (e.g. copykat_predict, conicsmat, numbat, …)

Cell colour  = mean value across all cells in that dataset×method pair.
Cell text    = "mean ± sd".

In addition to the per-metric *all-cells* heatmaps, the script can also emit
one *tumor-only* heatmap per metric in --tumor_only_metrics (default: Pearson
Correlation Coefficient, Spearman Correlation Coefficient, CopyNumber gain
ROC-AUC, CopyNumber loss ROC-AUC).  Tumor-cell selection uses a strict
per-row priority:

    1. author-provided sample annotation (`celltype` column), when non-null
    2. DNA-based annotation (`celltype_dna` column)

A row is considered "tumor" when its label matches "tumor" (case-insensitive
substring, so "Tumor", "TUMOR", "tumor_cell" all qualify). Rows for which
neither column has a value are excluded.

Summary table
-------------
Alongside the figures the script writes ONE combined TSV, dataset_summary.tsv,
with one row per dataset and exactly these columns:

    dataset
    n_cells_total
    tumor_purity                     (renamed from tumor_purity_cellfraction__dna)
    sample_mean_ploidy               (renamed from tumor_sample_ploidy_mean__dna)
    scRNA_reads_per_cell_mean
    scRNA_reads_per_cell_median
    scWGS_reads_per_cell_mean
    scWGS_reads_per_cell_median
    ref_and_purity_inference_method               ("pre-given" or "inferred", indicating origin of label)

The values for the first seven columns are taken from each dataset's own
dataset_metrics.tsv (written by compute_dataset_metrics.py). The ref_and_purity_inference_method
column is derived from the per_cell_metrics.tsv file (same directory) by
inspecting the `config_celltype` and `label_dna` columns.
Those file paths are NOT passed in — they are inferred from --input_glob: for
every evaluation directory the glob touches, the sibling files
<evaluation_dir>/dataset_metrics.tsv and <evaluation_dir>/per_cell_metrics.tsv
are read. Columns absent from a given file become empty cells; a dataset whose
dataset_metrics.tsv is missing is skipped with a warning.

Usage
-----
    python plot_cnv_heatmaps.py \
        --input_glob "$PWD/results/BCI*solo-genefull_output/evaluation/*.tsv" \
        --outdir ./heatmaps \
        [--metrics "Pearson Correlation Coefficient,CopyNumber gain F-score"] \
        [--file_pattern "without_preclassified_cells"] \
        [--summary_out ./heatmaps/dataset_summary.tsv] \
        [--tumor_only_metrics "Pearson Correlation Coefficient,Spearman Correlation Coefficient,CopyNumber gain ROC-AUC,CopyNumber loss ROC-AUC"] \
        [--no_tumor_only]
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

name2plotname = {
"A375_HCA00102_PRJNA603321"              : "DNTR-seq_PRJNA603321_A375_noRefCells",
"BCIS106T_chip1_SAMN48409192_SRR33511671": "wellDR-seq_BCIS106T_chip1_SRR33511671_tumorNormalMix",
"BCIS106T_chip2_SAMN48409193_SRR33511670": "wellDR-seq_BCIS106T_chip2_SRR33511670_tumorNormalMix",
"BCIS13T_chip1_SAMN40389282_SRR28357490" : "wellDR-seq_BCIS13T_chip1_SRR28357490_tumorNormalMix",
"BCIS13T_chip2_SAMN40389283_SRR28357489" : "wellDR-seq_BCIS13T_chip2_SRR28357489_tumorNormalMix",
"BCIS28T_chip1_SAMN40389273_SRR28357499" : "wellDR-seq_BCIS28T_chip1_SRR28357499_tumorNormalMix",
"BCIS28T_chip2_SAMN40389274_SRR28357498" : "wellDR-seq_BCIS28T_chip2_SRR28357498_tumorNormalMix",
"BCIS51T_chip1_SAMN48409194_SRR33511669" : "wellDR-seq_BCIS51T_chip1_SRR33511669_tumorNormalMix",
"BCIS51T_chip2_SAMN48409195_SRR33511668" : "wellDR-seq_BCIS51T_chip2_SRR33511668_tumorNormalMix",
"BCIS66T_chip1_SAMN48409190_SRR33511673" : "wellDR-seq_BCIS66T_chip1_SRR33511673_tumorNormalMix",
"BCIS66T_chip2_SAMN48409191_SRR33511672" : "wellDR-seq_BCIS66T_chip2_SRR33511672_tumorNormalMix",
"BCIS70T_chip1_SAMN48409188_SRR33511675" : "wellDR-seq_BCIS70T_chip1_SRR33511675_tumorNormalMix",
"BCIS70T_chip2_SAMN48409189_SRR33511674" : "wellDR-seq_BCIS70T_chip2_SRR33511674_tumorNormalMix",
"BCIS74T_chip1_SAMN48409186_SRR33511677" : "wellDR-seq_BCIS74T_chip1_SRR33511677_tumorNormalMix",
"BCIS74T_chip2_SAMN48409187_SRR33511676" : "wellDR-seq_BCIS74T_chip2_SRR33511676_tumorNormalMix",
"Cellline_mixing_experiment_1_1_SAMN48409183_SRR33511680" : "wellDR-seq_Cellline_mixing_experiment_1_1_SRR33511680_noRefCells",
"Cellline_mixing_experiment_1_2_SAMN48409184_SRR33511679" : "wellDR-seq_Cellline_mixing_experiment_1_2_SRR33511679_noRefCells",
"Cellline_mixing_experiment_2_SAMN48409185_SRR33511678" : "wellDR-seq_Cellline_mixing_experiment_2_SRR33511678_noRefCells",
"ECIS25T_chip1_SAMN40389270_SRR28357502" : "wellDR-seq_ECIS25T_chip1_SRR28357502_tumorNormalMix",
"ECIS25T_chip2_SAMN40389271_SRR28357501" : "wellDR-seq_ECIS25T_chip2_SRR28357501_tumorNormalMix",
"ECIS25T_chip3_SAMN40389272_SRR28357500" : "wellDR-seq_ECIS25T_chip3_SRR28357500_tumorNormalMix",
"ECIS36T_chip1_SAMN40389284_SRR28357488" : "wellDR-seq_ECIS36T_chip1_SRR28357488_tumorNormalMix",
"ECIS36T_chip2_SAMN40389285_SRR28357487" : "wellDR-seq_ECIS36T_chip2_SRR28357487_tumorNormalMix",
"ECIS44T_chip1_SAMN40389277_SRR28357495" : "wellDR-seq_ECIS44T_chip1_SRR28357495_tumorNormalMix",
"ECIS44T_chip2_SAMN40389278_SRR28357494" : "wellDR-seq_ECIS44T_chip2_SRR28357494_tumorNormalMix",
"ECIS44T_chip3_SAMN40389279_SRR28357493" : "wellDR-seq_ECIS44T_chip3_SRR28357493_tumorNormalMix",
"ECIS44T_chip4_SAMN40389280_SRR28357492" : "wellDR-seq_ECIS44T_chip4_SRR28357492_tumorNormalMix",
"ECIS44T_chip5_SAMN40389281_SRR28357491" : "wellDR-seq_ECIS44T_chip5_SRR28357491_tumorNormalMix",
"ECIS48T_chip1_SAMN40389275_SRR28357497" : "wellDR-seq_ECIS48T_chip1_SRR28357497_tumorNormalMix",
"ECIS48T_chip2_SAMN40389276_SRR28357496" : "wellDR-seq_ECIS48T_chip2_SRR28357496_tumorNormalMix",
"ECIS57T_chip1_SAMN48409196_SRR33511667" : "wellDR-seq_ECIS57T_chip1_SRR33511667_tumorNormalMix",
"ECIS57T_chip2_SAMN48409197_SRR33511666" : "wellDR-seq_ECIS57T_chip2_SRR33511666_tumorNormalMix",
"HCT116_PRJNA603321"                     : "DNTR-seq_PRJNA603321_HCT116_noRefCells",
"MDA231_chip1_SAMN40389268_SRR33482482"  : "wellDR-seq_MDA231_chip1_SRR33482482_noRefCells",
"MDA231_chip2_SAMN40389269_SRR33482483"  : "wellDR-seq_MDA231_chip2_SRR33482483_noRefCells",
"scONE-seq_HCT116"                       : "scONE-seq_PRJNA768428_HCT116_noRefCells",
"scONE-seq_HCT116_HUVEC_H9_as_T_N"       : "scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef",
"scONE-seq_NPC43"                        : "scONE-seq_PRJNA768428_NPC43_noRefCells",
"scONE-seq_NPC43_HUVEC_H9_as_T_N"        : "scONE-seq_PRJNA768428_NPC43_asTumor_HUVEC-H9_asRef",
"wellDR3_SAMN48409182_SRR33511681"       : "wellDR-seq_wellDR3_SRR33511681",
}

# ---------------------------------------------------------------------------
# Per-dataset summary table specification
# ---------------------------------------------------------------------------
# Columns are taken verbatim from each dataset's dataset_metrics.tsv. Order is
# preserved in the output; a column missing from a given file becomes NaN.
# The columns 'tumor_purity' and 'sample_mean_ploidy' are renamed from
# 'tumor_purity_cellfraction__dna' and 'tumor_sample_ploidy_mean__dna'
# respectively.
SUMMARY_COLUMNS = [
    "dataset",
    "n_cells_total",
    "tumor_purity",                     # renamed
    "sample_mean_ploidy",               # renamed
    "scRNA_reads_per_cell_mean",
    "scRNA_reads_per_cell_median",
    "scWGS_reads_per_cell_mean",
    "scWGS_reads_per_cell_median",
    "ref_and_purity_inference_method",                     # "pre-given" or "inferred"
]
# The per-dataset metrics file that lives next to the benchmark eval TSVs.
DATASET_METRICS_BASENAME = "dataset_metrics.tsv"
PER_CELL_BASENAME = "per_cell_metrics.tsv"
# Descriptive / non-benchmark files that share the evaluation directory and must
# NOT be fed into the heatmap loader.
NON_BENCHMARK_BASENAMES = {
    DATASET_METRICS_BASENAME,
    PER_CELL_BASENAME,
    "annotation_agreement.tsv",
    "annotation_discordant_cells.tsv",
    "cnv_event_sizes.tsv",
    "dataset_summary.tsv",
}


# ---------------------------------------------------------------------------
# Extra heatmaps sourced from the classification files
# ---------------------------------------------------------------------------
# Tumor/normal classification metrics are NOT in the per-cell eval TSVs; each
# eval writes a sibling '<eval_prefix>.cell_classification.tsv'
# (schema: caller, comparison, metric, value). Each spec below pulls one
# (comparison, metric) cell out of those files and turns it into a heatmap:
#   * 'rna_vs_dna' / 'accuracy'                     -> hard-label agreement of the
#     scRNA-inferred tumor/normal call vs the scWGS (Ginkgo) call.
#   * 'rna_score_vs_dna' / 'aneuploidy_score_auroc' -> threshold-agnostic AUROC of
#     the CONTINUOUS per-cell scRNA aneuploidy score (MADD) against the scWGS
#     gold-standard labels, i.e. how well the RNA CNV profile *ranks* aneuploid
#     (tumor) above near-diploid (reference) cells. This is NOT the 'roc_auc'
#     inside 'rna_vs_dna' (that is an AUROC of hard 0/1 labels == balanced
#     accuracy), which is why the eval script gives it its own comparison.
CLASSIFICATION_BASENAME_SUFFIX = ".cell_classification.tsv"

# (comparison, source_metric, heatmap_metric_name)
CLASSIFICATION_METRIC_SPECS = [
    ("rna_vs_dna", "accuracy",
     "Tumor/normal classification accuracy (scRNA vs scWGS)"),
    ("rna_score_vs_dna", "aneuploidy_score_auroc",
     "Tumor/normal classification ROC-AUC (scRNA aneuploidy score vs scWGS)"),
]

# (2) Emitted per-cell by evaluate_caller_vs_ginkgo.py, already masked to
#     scRNA-normal cells (NaN elsewhere). Name must match the eval script exactly.
DIPLOID_IN_RNA_NORMAL_METRIC = "Fraction of the scWGS genome diploid in scRNA-normal cells"

# Metrics in [0,1] where higher is better -> green-is-good diverging scale.
HIGHER_IS_BETTER_01_METRICS = {
    DIPLOID_IN_RNA_NORMAL_METRIC,
} | {name for _c, _m, name in CLASSIFICATION_METRIC_SPECS}


# ---------------------------------------------------------------------------
# Tumor-only heatmap configuration
# ---------------------------------------------------------------------------
# Metrics to plot a SECOND time, restricted to cells labelled as tumor.  The
# per-row tumor label is resolved with strict author > DNA priority:
#     1. `celltype` (author annotation), when non-null
#     2. `celltype_dna` (DNA-based annotation)
# A row matches "tumor" when its label contains "tumor" (case-insensitive
# substring), so "Tumor", "TUMOR", "tumor_cell" all qualify.
DEFAULT_TUMOR_ONLY_METRICS = [
    "Pearson Correlation Coefficient",
    "Spearman Correlation Coefficient",
    "CopyNumber gain ROC-AUC",
    "CopyNumber loss ROC-AUC",
]
TUMOR_LABEL_PATTERN = "tumor"   # case-insensitive substring match


# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CNV benchmark heatmaps with dispersion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_glob",
        default="./results/*solo-genefull_output/evaluation/*.tsv",
        help="Glob pattern for evaluation TSV files (e.g. ./results/**/evaluation/*.tsv). The per-dataset "
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
        "--tumor_only_metrics",
        default=",".join(DEFAULT_TUMOR_ONLY_METRICS),
        help="Comma-separated list of metrics to ALSO plot on tumor-only cells. "
             "Tumor labels are taken per-row from the author-provided `celltype` "
             "column, falling back to the DNA-based `celltype_dna` column. "
             "Pass an empty string '' to plot none, or use --no_tumor_only.",
    )
    p.add_argument(
        "--no_tumor_only", action="store_true",
        help="Skip the tumor-only heatmaps entirely (default: enabled, see "
             "--tumor_only_metrics for the list).",
    )
    p.add_argument(
        "--fmt", default="pdf",
        help="Output figure format: pdf, png, svg (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=200, help="DPI for raster formats")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Extract dataset name from file path
# ---------------------------------------------------------------------------
def dataset_from_path_1(filepath: str) -> str:
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

def dataset_from_path(filepath: str) -> str:
    ret = dataset_from_path_1(filepath)
    return name2plotname.get(ret, ret)


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

        # ---- Per-row tumor label with strict author > DNA priority ----
        # Used to subset rows to tumor cells for the tumor-only heatmaps. This
        # is a per-row merge (not a per-file fallback like final_celltype), so
        # rows where the author says "Tumor" but DNA says "normal" are kept as
        # tumor, while rows where only DNA has a value still resolve correctly.
        has_celltype = "celltype" in df.columns
        has_celltype_dna = "celltype_dna" in df.columns
        if has_celltype and has_celltype_dna:
            df["tumor_label"] = df["celltype"].combine_first(df["celltype_dna"])
        elif has_celltype:
            df["tumor_label"] = df["celltype"]
        elif has_celltype_dna:
            df["tumor_label"] = df["celltype_dna"]
        else:
            df["tumor_label"] = np.nan

        frames.append(df)

    if not frames:
        sys.exit("No usable data after loading.")

    data = pd.concat(frames, ignore_index=True)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    # Normalise caller names: strip suffixes like _predict, _predict_hg19 → keep full name
    # but unify duplicates where 'with' vs 'without' preclassified gives same caller string
    data["method"] = data["caller"].str.replace(r"_predict", r"_autoInferRef", regex=False).str.strip()

    # ---- Tumor-label coverage (informational) ----
    n_total = len(data)
    label = data["tumor_label"]
    n_with_label = int(label.notna().sum())
    # Use the same robust tumor detection as filter_to_tumor for counting
    tumor_mask = label.astype(str).str.contains(r'(?i)\b(tumor|tumor_|_tumor)\b', na=False) & \
                 ~label.astype(str).str.contains(r'(?i)\b(non|normal)[-_]?tumor\b', na=False)
    n_tumor = int(tumor_mask.sum())
    n_normal = n_with_label - n_tumor
    print(f"  {len(data)} rows, {data['dataset'].nunique()} datasets, "
          f"{data['method'].nunique()} methods, {data['metric'].nunique()} metrics")
    print(f"  tumor_label coverage: {n_with_label}/{n_total} rows have a label "
          f"(tumor={n_tumor}, non-tumor={n_normal}); "
          f"{n_total - n_with_label} rows will be excluded from tumor-only plots.")
    return data


# ---------------------------------------------------------------------------
# 3c. Load tumor/normal classification accuracy (different schema)
# ---------------------------------------------------------------------------
def load_classification(glob_pattern, file_pattern) -> pd.DataFrame:
    """Read the sibling '*.cell_classification.tsv' files and return long rows
    (dataset, method, metric, value) for every spec in
    CLASSIFICATION_METRIC_SPECS. One row per dataset x method x heatmap-metric;
    empty frame if none are found."""
    files = sorted(glob.glob(glob_pattern, recursive=True))
    files = [f for f in files
             if os.path.basename(f).endswith(CLASSIFICATION_BASENAME_SUFFIX)]
    if file_pattern:
        files = [f for f in files if file_pattern in os.path.basename(f)]

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"  [classif] SKIP {fp}: {e}")
            continue
        if not {"caller", "comparison", "metric", "value"}.issubset(df.columns):
            continue
        ds = dataset_from_path(fp)
        for comparison, source_metric, heatmap_name in CLASSIFICATION_METRIC_SPECS:
            sel = df[(df["comparison"] == comparison) &
                     (df["metric"] == source_metric)].copy()
            if sel.empty:
                continue
            sel["dataset"] = ds
            sel["method"]  = sel["caller"].astype(str).str.strip()
            sel["metric"]  = heatmap_name
            sel["value"]   = pd.to_numeric(sel["value"], errors="coerce")
            frames.append(sel[["dataset", "method", "metric", "value"]])

    if not frames:
        specs = ", ".join(f"{c}/{m}" for c, m, _ in CLASSIFICATION_METRIC_SPECS)
        print(f"  [classif] no rows for any of [{specs}] found in "
              f"*{CLASSIFICATION_BASENAME_SUFFIX}; those heatmaps will be skipped.")
        return pd.DataFrame(columns=["dataset", "method", "metric", "value"])

    out = pd.concat(frames, ignore_index=True)
    print(f"  [classif] loaded {len(out)} classification row(s) across "
          f"{out['metric'].nunique()} metric(s) from {out['dataset'].nunique()} dataset(s).")
    return out


# ---------------------------------------------------------------------------
# 3b. Single combined summary table (inferred dataset_metrics.tsv and per_cell_metrics.tsv)
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


def infer_per_cell_files(input_glob: str):
    """Same as infer_metrics_files but for per_cell_metrics.tsv."""
    matches = sorted(glob.glob(input_glob, recursive=True))
    eval_dirs = []
    seen_dirs = set()
    for m in matches:
        d = os.path.dirname(m)
        if d not in seen_dirs:
            seen_dirs.add(d)
            eval_dirs.append(d)

    direct = [m for m in matches
              if os.path.basename(m) == PER_CELL_BASENAME]

    per_cell_files = list(direct)
    for d in eval_dirs:
        cand = os.path.join(d, PER_CELL_BASENAME)
        if os.path.exists(cand):
            per_cell_files.append(cand)

    seen, out = set(), []
    for f in per_cell_files:
        if f not in seen and os.path.exists(f):
            seen.add(f)
            out.append(f)
    out.sort(key=lambda f: (dataset_from_path(f), f))
    return out


def compute_tumor_origin_and_purity(per_cell_file: str):
    """
    Read a per_cell_metrics.tsv and determine:
      - origin: "cell_line_metadata" if config_celltype provides definitive labels,
                else "scWGS_CNV_calls" if label_dna provides labels,
                else np.nan.
      - purity: fraction of cells that are tumor (case‑insensitive substring "tumor")
                among cells that have a definitive label in the chosen column.

    Returns:
        tuple(origin, purity)
    """
    try:
        df = pd.read_csv(per_cell_file, sep="\t")
    except Exception:
        return np.nan, np.nan

    has_config = "config_celltype" in df.columns
    has_label_dna = "label_dna" in df.columns

    if not has_config and not has_label_dna:
        return np.nan, np.nan

    # Determine which column to use for tumor labels
    use_config = False
    if has_config:
        # If any non-Unknown value exists in config_celltype, use it
        config_vals = df["config_celltype"].astype(str)
        non_unknown = config_vals[~config_vals.str.lower().str.contains("unknown", na=False)]
        if not non_unknown.empty:
            use_config = True

    if use_config:
        col = "config_celltype"
        origin = "cell_line_metadata"
        # Exclude Unknown / NA cells for purity calculation
        mask = df[col].notna() & (~df[col].astype(str).str.lower().str.contains("unknown", na=False))
        valid = df.loc[mask, col]
    elif has_label_dna:
        col = "label_dna"
        origin = "scWGS_CNV_calls"
        mask = df[col].notna()
        valid = df.loc[mask, col]
    else:
        return np.nan, np.nan

    if valid.empty:
        return origin, np.nan

    # Purity = fraction of valid cells whose label contains "tumor"
    purity = (valid.astype(str).str.contains("tumor", case=False, na=False).sum()) / len(valid)
    return origin, purity


def build_dataset_summary(metrics_files, per_cell_files) -> pd.DataFrame:
    """Read each dataset_metrics.tsv and select SUMMARY_COLUMNS into one table.

    - Missing columns are filled with NaN (kept in the requested order).
    - The `dataset` column is recovered from the file path if absent/blank.
    - dataset_metrics.tsv holds one row per dataset; rows are concatenated and
      sorted by dataset name. Duplicate datasets (same name) are collapsed,
      keeping the first non-null value per column.
    - The columns 'tumor_purity_cellfraction__dna' and
      'tumor_sample_ploidy_mean__dna' are renamed to 'tumor_purity' and
      'sample_mean_ploidy' respectively.
    - The 'ref_and_purity_inference_method' column is derived from per_cell_metrics.tsv.
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

        # Rename columns (we will override tumor_purity later with computed value)
        rename_map = {
            "tumor_purity_cellfraction__dna": "tumor_purity",
            "tumor_sample_ploidy_mean__dna": "sample_mean_ploidy",
        }
        df.rename(columns=rename_map, inplace=True, errors="ignore")

        # Ensure we have the columns we want; missing ones become NaN
        df = df.reindex(columns=SUMMARY_COLUMNS)
        if df["dataset"].isna().all():
            df["dataset"] = dataset_from_path(fp)

        df["dataset"] = df["dataset"].replace(name2plotname)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    out = pd.concat(rows, ignore_index=True)

    # Collapse any accidental duplicate dataset rows: first non-null wins.
    if out["dataset"].duplicated().any():
        out = (out.groupby("dataset", as_index=False, sort=False)
                  .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))

    out = out.sort_values("dataset", kind="stable").reset_index(drop=True)

    # ---- Compute origin and purity from per_cell files ----
    origin_dict = {}
    purity_dict = {}
    for pcf in per_cell_files:
        ds = dataset_from_path(pcf)
        origin, purity = compute_tumor_origin_and_purity(pcf)
        if pd.isna(origin):
            continue
        origin_dict[ds] = origin
        if not pd.isna(purity):
            purity_dict[ds] = purity

    # Override tumor_purity with computed values
    if purity_dict:
        out["tumor_purity"] = out["dataset"].map(purity_dict)

    # Set ref_and_purity_inference_method with mapped values ("pre-given" or "inferred")
    if origin_dict:
        out["ref_and_purity_inference_method"] = out["dataset"].map(origin_dict)
    else:
        out["ref_and_purity_inference_method"] = np.nan

    return out[SUMMARY_COLUMNS]


def write_dataset_summary(input_glob: str, summary_out: str) -> pd.DataFrame:
    """Infer inputs, build the single combined table, and write it to disk.

    The file is always written (header-only if no dataset_metrics.tsv is found),
    so the output reliably exists for downstream steps.

    Returns:
        The built summary DataFrame.
    """
    metrics_files = infer_metrics_files(input_glob)
    per_cell_files = infer_per_cell_files(input_glob)

    if metrics_files:
        print(f"Building summary from {len(metrics_files)} "
              f"{DATASET_METRICS_BASENAME} file(s) inferred from --input_glob …")
    else:
        print(f"[summary] WARNING: no {DATASET_METRICS_BASENAME} found next to the "
              f"files matched by {input_glob!r}; writing header-only summary.")

    summary = build_dataset_summary(metrics_files, per_cell_files)
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
    # iqr removed as it was unused
    return agg


# ---------------------------------------------------------------------------
# 4b. Filter rows to tumor cells (author > DNA priority)
# ---------------------------------------------------------------------------
def filter_to_tumor(data: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose `tumor_label` matches "tumor" (case-insensitive
    substring). Rows missing both `celltype` and `celltype_dna` are dropped
    because their tumor status is unknown.

    The `tumor_label` column is built in `load_all` with strict per-row
    priority: author `celltype` > DNA `celltype_dna`.
    """
    label = data["tumor_label"]
    # Match "tumor" as a whole word, optionally with underscores, but exclude "non-tumor" / "normal"
    mask = label.astype(str).str.contains(r'(?i)\b(tumor|tumor_|_tumor)\b', na=False) & \
           ~label.astype(str).str.contains(r'(?i)\b(non|normal)[-_]?tumor\b', na=False)
    out = data[mask].copy()
    print(f"  [tumor-only] {len(out)}/{len(data)} rows kept "
          f"({len(out) / max(len(data), 1) * 100:.1f}%) after filtering to tumor cells.")
    return out


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


# revised by https://www.doubao.com/thread/xb758d1c4c8fc84f988eba87d7023fd40
#    origin: https://www.doubao.com/chat/38434352775086082
# Core fix: align top-right of label to column center
def heatmap_prettify(ax):
    """
    Align the TOP-RIGHT corner of each rotated x-axis label
    to the vertical centerline of its heatmap column.
    """
    ax.tick_params(
        axis="x",
        rotation=20,
        labelsize=9,
    )

    for label in ax.get_xticklabels():
        label.set_ha("right")           # right edge → tick x (column center)
        label.set_va("top")             # top edge → tick y (axis baseline)
        label.set_rotation_mode("anchor")  # rotate around the (ha, va) anchor point

    return ax


def plot_heatmap(
    agg: pd.DataFrame,
    metric: str,
    outdir: str,
    fmt: str,
    dpi: int,
    title_suffix: str = "",
    filename_suffix: str = "",
    purity_map: dict = None,
):
    """
    Draw one figure per metric.

    The heatmap cell is coloured by the mean value.
    Inside each cell we annotate the mean ± sd.

    Parameters
    ----------
    title_suffix    : appended to the figure title (e.g. " (tumor cells only)")
    filename_suffix : appended to the output filename before the extension
                      (e.g. "_tumor_only")
    purity_map      : dict {dataset: tumor_purity} used to decide if an asterisk
                      is appended to the dataset name (purity < 0.2).
    """
    if purity_map is None:
        purity_map = {}

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

    # --- Choose colourmap and scaling ---
    if metric in CORRELATION_METRICS:
        cmap = "RdBu_r"
        vmin, vmax = -1, 1
        center = 0
    elif metric in HIGHER_IS_BETTER_01_METRICS:
        # metrics in [0,1] where higher is better -> green-is-good diverging
        cmap = "RdYlGn"
        vmin, vmax = 0, 1
        center = 0.5
    elif "ROC-AUC" in metric:
        cmap = "RdYlGn"
        vmin, vmax = 0, 1
        center = 0.5
    else:
        # Unknown metric: use a sequential colormap scaled to the data range,
        # without assuming a direction or fixed bounds.
        cmap = "viridis"
        # compute global min/max from the pivot (ignore NaN)
        data_min = pivot_mean.min().min()
        data_max = pivot_mean.max().max()
        if np.isnan(data_min) or np.isnan(data_max):
            vmin, vmax = 0, 1   # fallback
        else:
            vmin, vmax = data_min, data_max
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
            elif np.isnan(sd):
                # Single observation (e.g. one classification accuracy per
                # dataset x method): no dispersion to show.
                annot[i, j] = f"{m:.2f}"
            else:
                # Compact annotation: mean ± sd
                annot[i, j] = (
                    f"{m:.2f} ±{sd:.2f}"
                    #f"\n[{q1:.2f},{q3:.2f}]"
                )

    # --- Figure size ---
    cell_w = max(0.75, min(2.4, 16 / max(n_cols, 1)))
    cell_h = max(0.25, min(1.4, 10 / max(n_rows, 1)))
    fig_w = cell_w * n_cols + 4.8   # extra room for labels + colourbar
    fig_h = cell_h * n_rows + 3.0
    fig_w = max(fig_w, 8)
    fig_h = max(fig_h, 5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- Row labels (dataset names), with asterisk if purity < 20% ---
    row_labels = []
    asterisk_needed = False
    for ds in pivot_mean.index:
        purity = purity_map.get(ds, np.nan)
        if pd.notna(purity) and purity < 0.2:
            label = ds + "*"
            asterisk_needed = True
        else:
            label = ds
        row_labels.append(label)

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
        annot_kws={"fontsize": 8, "fontfamily": "monospace", "va": "center"},
        cbar_kws={"label": "Mean value", "shrink": 0.75},
        yticklabels=row_labels,
    )

    ax.set_title(metric + title_suffix, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    ax = heatmap_prettify(ax)

    # --- Footnote for asterisk ---
    if asterisk_needed:
        fig.text(0.02, 0.02, "* tumor purity < 20%", fontsize=8, ha="left", va="bottom")

    fig.tight_layout()

    safe_name = re.sub(r"[^\w]+", "_", metric).strip("_")
    out_path = os.path.join(outdir, f"heatmap_{safe_name}{filename_suffix}.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# 5b. Tumor-only heatmaps: one extra figure per metric in `metrics`
# ---------------------------------------------------------------------------
def plot_tumor_only_heatmaps(
    data: pd.DataFrame,
    metrics,
    outdir: str,
    fmt: str,
    dpi: int,
    purity_map: dict = None,
):
    """Generate one heatmap per metric in `metrics`, restricted to tumor cells.

    Annotation priority is enforced in `filter_to_tumor`: per-row,
        author `celltype` > DNA-based `celltype_dna`.
    """
    tumor = filter_to_tumor(data)
    if tumor.empty:
        print("[tumor-only] No tumor rows after filtering; skipping all "
              "tumor-only heatmaps.")
        return

    agg_tumor = aggregate(tumor)
    available = set(tumor["metric"].unique())

    plotted = 0
    for metric in metrics:
        if metric not in available:
            print(f"[tumor-only] WARNING: {metric!r} had no tumor rows at all; "
                  f"skipping this tumor-only heatmap.")
            continue
        plot_heatmap(
            agg_tumor,
            metric,
            outdir,
            fmt,
            dpi,
            title_suffix=" (tumor cells only)",
            filename_suffix="_tumor_only",
            purity_map=purity_map,
        )
        plotted += 1

    if plotted == 0:
        print("[tumor-only] None of the requested metrics had tumor rows; "
              "no tumor-only heatmap written.")
    else:
        # Drop the aggregated TSV for the tumor subset alongside the all-cells
        # aggregated_results.tsv written by main(), so downstream steps can pick
        # it up too.
        agg_path = os.path.join(outdir, "aggregated_results_tumor_only.tsv")
        agg_tumor.to_csv(agg_path, sep="\t", index=False, na_rep='nan')
        print(f"[tumor-only] aggregated data → {agg_path} ({plotted} metric(s))")


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

    # Use data-driven scaling with a neutral sequential colormap
    data_min = overview.min().min()
    data_max = overview.max().max()
    if np.isnan(data_min) or np.isnan(data_max):
        vmin, vmax = 0, 1
    else:
        vmin, vmax = data_min, data_max

    fig_w = max(10, n_cols * 1.5 + 3)
    fig_h = max(6, n_rows * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        overview,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        linewidths=0.5,
        linecolor="#e0e0e0",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Grand mean", "shrink": 0.6},
    )
    ax.set_title("Overview: grand-mean per method across all datasets",
                  fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel("Metric", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax = heatmap_prettify(ax)
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
    summary = write_dataset_summary(args.input_glob, summary_out)

    # Build purity map (dataset -> tumor_purity) for asterisk annotation.
    purity_map = {}
    if not summary.empty and "tumor_purity" in summary.columns:
        for _, row in summary.iterrows():
            ds = row["dataset"]
            pur = row["tumor_purity"]
            if pd.notna(pur):
                purity_map[ds] = pur

    # --- Heatmaps (original behaviour + two extra heatmaps) ---
    data = load_all(args.input_glob, args.file_pattern)

    # Tumor/normal classification accuracy lives in the sibling
    # *.cell_classification.tsv files (different schema); fold it in as an extra
    # metric so it becomes one more heatmap. The diploid-in-scRNA-normal metric
    # arrives automatically via load_all (it is a per-cell metric).
    clf = load_classification(args.input_glob, args.file_pattern)
    if not clf.empty:
        data = pd.concat([data, clf], ignore_index=True)

    agg  = aggregate(data)

    metrics_to_plot = sorted(data["metric"].unique())
    if args.metrics:
        requested = [m.strip() for m in args.metrics.split(",")]
        # Keep only metrics that exist, but warn about missing ones
        available = set(metrics_to_plot)
        missing = [m for m in requested if m not in available]
        if missing:
            print(f"WARNING: The following requested metrics were not found and will be skipped: {missing}")
        metrics_to_plot = [m for m in metrics_to_plot if m in requested]
        if not metrics_to_plot:
            print("No requested metrics available; skipping all heatmaps.")
            return

    print(f"\nPlotting {len(metrics_to_plot)} metric heatmaps …")
    for metric in metrics_to_plot:
        plot_heatmap(agg, metric, args.outdir, args.fmt, args.dpi,
                     purity_map=purity_map)

    print("\nPlotting overview …")
    plot_overview(agg, args.outdir, args.fmt, args.dpi)

    # --- Tumor-only heatmaps (additional figures) ---
    if not args.no_tumor_only:
        tumor_metrics = [m.strip() for m in args.tumor_only_metrics.split(",")
                         if m.strip()]
        if tumor_metrics:
            print(f"\nPlotting {len(tumor_metrics)} tumor-only metric heatmap(s) …")
            plot_tumor_only_heatmaps(data, tumor_metrics,
                                      args.outdir, args.fmt, args.dpi,
                                      purity_map=purity_map)
        else:
            print("\n[tumor-only] --tumor_only_metrics is empty; "
                  "no tumor-only heatmaps written.")

    # Also save the aggregated table as TSV for downstream use
    agg_path = os.path.join(args.outdir, "aggregated_results.tsv")
    agg.to_csv(agg_path, sep="\t", index=False, na_rep='nan')
    print(f"\nAggregated data → {agg_path}")
    print("Done.")


if __name__ == "__main__":
    main()
