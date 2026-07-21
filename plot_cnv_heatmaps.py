#!/usr/bin/env python3

'''
https://sorryios.ai/chat/c7f1600f-8e4c-4d0e-804f-edbb44344127
  exclude HG19 performance results
'''


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
    tumor_purity_from_config          (from config_celltype only; NA if not available)
    tumor_purity_from_scWGS           (from label_dna, always when available)
    sample_mean_ploidy                (renamed from tumor_sample_ploidy_mean__dna)
    scRNA_reads_per_cell_mean
    scRNA_reads_per_cell_median
    scWGS_reads_per_cell_mean
    scWGS_reads_per_cell_median
    ref_and_purity_inference_method   ("cell_line_metadata" if config_celltype supplied the primary purity, else NA)

The values for the first seven columns are taken from each dataset's own
dataset_metrics.tsv (written by compute_dataset_metrics.py). The ref_and_purity_inference_method
column AND the two tumor_purity columns are derived from the per_cell_metrics.tsv
file (same directory) by inspecting the `config_celltype` and `label_dna`
columns. BOTH purities are computed independently — `tumor_purity_from_config` is
ONLY from `config_celltype` (NA if missing), while `tumor_purity_from_scWGS` is
always from `label_dna`. Thus, the scWGS-inferred purity is always visible, and
the primary purity (if available from metadata) is shown separately.
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

LOW_TUMOR_PURITY = "$^A$"
NO_NORMAL_SET = "$^B$"

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
# The column 'sample_mean_ploidy' is renamed from 'tumor_sample_ploidy_mean__dna'.
# 'tumor_purity_from_config' is set ONLY from `config_celltype` (pre-given /
# cell-line metadata). If that column is missing or has no usable non-Unknown
# labels, the value remains NA (no fallback to DNA).
# 'tumor_purity_from_scWGS' is ALWAYS computed from `label_dna` alone, so both
# purities are exposed side-by-side.
SUMMARY_COLUMNS = [
    "dataset",
    "n_cells_total",
    "tumor_purity_from_config",          # from config_celltype only, NA if absent
    "tumor_purity_from_scWGS",           # from label_dna (always, when available)
    "sample_mean_ploidy",                # renamed
    "scRNA_reads_per_cell_mean",
    "scRNA_reads_per_cell_median",
    "scWGS_reads_per_cell_mean",
    "scWGS_reads_per_cell_median",
    "ref_and_purity_inference_method",   # "cell_line_metadata" if config supplied the primary purity, else NA
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

# Files whose basename contains this substring are excluded wherever
# --input_glob results are consumed (heatmap eval TSVs, classification TSVs,
# and the dataset_metrics.tsv / per_cell_metrics.tsv inference for the
# summary table).
EXCLUDED_FILENAME_SUBSTRING = "_hg19_with"


def _exclude_hg19_with(files):
    """Drop any path whose basename contains EXCLUDED_FILENAME_SUBSTRING."""
    kept = [f for f in files
            if EXCLUDED_FILENAME_SUBSTRING not in os.path.basename(f)]
    n_excluded = len(files) - len(kept)
    if n_excluded:
        print(f"  [filter] excluding {n_excluded} file(s) whose name contains "
              f"{EXCLUDED_FILENAME_SUBSTRING!r}.")
    return kept


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
    ("rna_score_vs_dna", "aneuploidy_score_auroc", # Tumor_normal_classification_ROC_AUC_scRNA_aneuploidy_score_vs_scWGS_aneuploidy_status
     "Tumor/normal classification ROC-AUC (scRNA aneuploidy score vs scWGS aneuploidy status)"),
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

# What types of plots to generate
PLOT_TYPES = [
    "all",
    "overview",
    "main",
    "supp",
]

# ---------------------------------------------------------------------------
# Main-text figure: grid of per-(metric, method) swarmplots
# ---------------------------------------------------------------------------
# One dot per dataset. Dot colour = tumor-purity range (scWGS-derived);
# dot shape = study / scDNA-scRNA co-sequencing technology. Rows of the grid
# are performance metrics, columns are scRNA-seq-based CNV calling methods.
#
# Row order below is deliberate: the tumor-only correlation/ROC-AUC metrics
# first (require filtering to tumor cells via `filter_to_tumor`), then the
# three all-cells metrics (coverage fractions + the classification ROC-AUC,
# which is folded into `data` via `load_classification` in main()).
# NOTE: Spearman Correlation Coefficient is intentionally excluded from this
# grid (kept in the original per-metric heatmaps) since it correlates very
# closely with Pearson CC and was judged redundant for this summary figure.
SWARM_GRID_TUMOR_ONLY_METRICS = [
    "Pearson Correlation Coefficient",
    "CopyNumber gain ROC-AUC",
    "CopyNumber loss ROC-AUC",
]
SWARM_GRID_ALL_CELLS_METRICS = [
    "Fraction_of_the_cells_with_inferred_copy_numbers",
    "Fraction_of_the_exome_with_inferred_copy_numbers",
    "Tumor_normal_classification_ROC_AUC_scRNA_aneuploidy_score_vs_scWGS_aneuploidy_status",
]
# Row labels shown on the figure (kept short; the underscore-heavy raw metric
# names above are display-unfriendly).
SWARM_GRID_ROW_DISPLAY_NAMES = {
    "Pearson Correlation Coefficient": "Tumor-only PCC",
    "CopyNumber gain ROC-AUC": "Tumor-only gain\nROC-AUC",
    "CopyNumber loss ROC-AUC": "Tumor-only loss\nROC-AUC",
    "Fraction_of_the_cells_with_inferred_copy_numbers": "Fraction of cells\nwith inferred CNs",
    "Fraction_of_the_exome_with_inferred_copy_numbers": "Fraction of exome\nwith inferred CNs",
    "Tumor_normal_classification_ROC_AUC_scRNA_aneuploidy_score_vs_scWGS_aneuploidy_status":
        "Tumor/normal\nclassification ROC-AUC\n\ni.e.,\nscRNA aneuploidy score\nvs\nscWGS aneuploidy status",
}
# Full ordered row list for the swarm grid (union of the two lists above, in
# the order rows are drawn top-to-bottom).
SWARM_GRID_METRICS = SWARM_GRID_TUMOR_ONLY_METRICS + SWARM_GRID_ALL_CELLS_METRICS

# Hard cap on the number of methods (grid columns); above this the grid gets
# unreadably wide, so we still plot but raise a loud, hard-to-miss warning.
MAX_METHODS_FOR_SWARM_GRID = 15

# Tumor-purity (scWGS-derived) colour bins. Edges are inclusive on the left,
# exclusive on the right, except the last bin which includes 1.0.
PURITY_BIN_EDGES = [0.0, 0.2, 0.5, 0.8, 1.0 + 1e-9]
PURITY_BIN_LABELS = ["<20%", "20-50%", "50-80%", ">80%"]
PURITY_BIN_UNKNOWN_LABEL = "Unknown"
# Same blue -> orange -> red hue progression as before (low -> high purity),
# but pushed to much higher HSV saturation (now ~73-99%, vs. ~50-87% in the
# original ColorBrewer-derived palette) so the four bins are more easily told
# apart at a glance, especially the two middle bins which were the most
# washed-out before.
PURITY_BIN_COLORS = {
    "<20%":     "#0b5ea8",
    "20-50%":   "#3aa8d8",
    "50-80%":   "#ff8c1a",
    ">80%":     "#d0021b",
    PURITY_BIN_UNKNOWN_LABEL: "#999999",
}

# Study / co-sequencing technology, parsed from the plotname prefix in
# `name2plotname` (e.g. "wellDR-seq_BCIS106T_..." -> "wellDR-seq"). Marker
# shapes chosen to be distinguishable at small size and in greyscale.
# Dict order here ALSO controls swarm-plotting draw order in
# `plot_metric_method_swarm_grid` (wellDR-seq drawn first, then scONE-seq,
# then DNTR-seq, per request), since that function iterates this dict in
# insertion order.
TECHNOLOGY_MARKERS = {
    "wellDR-seq": "^",  # triangle
    "scONE-seq": "X", # filled X
    "DNTR-seq": "s", # square
}
TECHNOLOGY_MARKER_UNKNOWN = "D"   # diamond, for any prefix not in the map above
TECHNOLOGY_UNKNOWN_LABEL = "Other"

# Method (grid column) names to always exclude from the swarm grid, regardless
# of whether they're present in the underlying data. Applied when building
# `col_methods` in `plot_metric_method_swarm_grid`.
SWARM_GRID_EXCLUDED_METHODS = {
    "copykat_cellline_autoInferRef",
    "infercna_autoInferRef_hg19",
    "infercna_hg19",
}

# Only keep the 'chip1' replicate for wellDR-seq datasets that were split
# across multiple chips (chip1, chip2, chip3, ...), to avoid one tumor sample
# contributing multiple near-duplicate points to the same panel. Matched as a
# literal substring against the (already name2plotname-remapped) dataset
# name; wellDR-seq datasets with NO chip token at all (e.g. "wellDR3",
# "Cellline_mixing_experiment_...") are also excluded under this rule, since
# they don't contain the literal substring "chip1" either.
WELLDR_SEQ_CHIP1_SUBSTRING = "chip1"


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
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    p.add_argument(
        "--plots", default=",".join(PLOT_TYPES),
        help="The types of plots that will be generated.",
    )
    
    # NEW: CLI argument for no normal cells glob
    p.add_argument(
        "--no_normal_cells_glob",
        default="./results/*_bams/rna/*.cluster_stat_zero_ref_cells.rds",
        help=f"Glob for sentinel files written by scripts/identify_normal_cell_subset.R when NO normal-cell cluster was found. Datasets matching this will have '{NO_NORMAL_SET}' appended to their names in plots and TSVs."
    )

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

# --- NEW HELPER FUNCTIONS FOR NO-NORMAL-CELLS DETECTION ---
def dataset_from_zero_ref_path(filepath: str) -> str:
    """Recover the dataset identifier from a sentinel file produced by
    scripts/identify_normal_cell_subset.R when NO normal-cell cluster was
    found (the script fell back to the top-scoring cluster and warned).
    """
    raw = None
    for part in Path(filepath).parts:
        m = re.match(r"^(.+?)_(bams|input)$", part)
        if m:
            raw = m.group(1)
            break
    if raw is None:
        raw = Path(filepath).parent.parent.name
    return name2plotname.get(raw, raw)

def build_no_normal_cells_set(no_normal_cells_glob) -> set:
    """Return the set of (name2plotname-remapped) dataset names for which
    scripts/identify_normal_cell_subset.R could NOT identify a normal-cell
    cluster and fell back to the top-scoring cluster (issuing its warning).
    """
    if not no_normal_cells_glob:
        return set()
    sentinels = sorted(glob.glob(no_normal_cells_glob, recursive=True))
    found = {dataset_from_zero_ref_path(s) for s in sentinels}
    if found:
        print(f"[no-normal-cells] {len(found)} dataset(s) had no normal-cell "
              f"cluster identified by identify_normal_cell_subset.R and will "
              f"be marked with '{NO_NORMAL_SET}': {sorted(found)}")
    return found

def _suffix_dataset_column(df: pd.DataFrame, no_normal_cells_set: set) -> pd.DataFrame:
    f"""Return a COPY of `df` with '{NO_NORMAL_SET}' appended to the `dataset` value of
    every dataset in `no_normal_cells_set`. The input is never mutated."""
    if not no_normal_cells_set or "dataset" not in df.columns:
        return df
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).map(
        lambda d: d + f"{NO_NORMAL_SET}" if d in no_normal_cells_set else d
    )
    return out


# ---------------------------------------------------------------------------
# 3. Load and concatenate all TSV files (for the heatmaps)
# ---------------------------------------------------------------------------
def load_all(glob_pattern: str, file_pattern) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern, recursive=True))
    if not files:
        sys.exit(f"No files matched: {glob_pattern}")

    # Skip files matching the global exclusion substring
    files = _exclude_hg19_with(files)

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
    files = _exclude_hg19_with(files)
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
            sel["method"]  = sel["caller"].astype(str).str.strip().str.replace(r'_predict', r'_autoInferRef')
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
    matches = _exclude_hg19_with(matches)
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
    matches = _exclude_hg19_with(matches)
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


def compute_tumor_purities(per_cell_file: str):
    """
    Read a per_cell_metrics.tsv and compute the tumor purity from BOTH
    inference methods INDEPENDENTLY (so that both purities are always
    available side-by-side in the summary table):

      - config_purity: fraction of cells whose label contains "tumor"
                       (case-insensitive substring) among the non-Unknown,
                       non-NA values in `config_celltype` (pre-given /
                       cell-line metadata).
      - dna_purity:    fraction of cells whose label contains "tumor"
                       (case-insensitive substring) among the non-NA values
                       in `label_dna` (inferred from scWGS CNV calls).

    Returns:
        tuple(config_purity, dna_purity)
        Either value is np.nan if the corresponding column is missing or
        has no usable labels.
    """
    try:
        df = pd.read_csv(per_cell_file, sep="\t")
    except Exception:
        return np.nan, np.nan

    def _purity(colname: str, exclude_unknown: bool = False):
        if colname not in df.columns:
            return np.nan
        valid = df[colname]
        mask = valid.notna()
        if exclude_unknown:
            mask &= ~valid.astype(str).str.lower().str.contains("unknown", na=False)
        valid = valid[mask]
        if valid.empty:
            return np.nan
        return (valid.astype(str).str.contains("tumor", case=False, na=False).sum()) / len(valid)

    config_purity = _purity("config_celltype", exclude_unknown=True)
    dna_purity    = _purity("label_dna",       exclude_unknown=False)

    return config_purity, dna_purity


def build_dataset_summary(metrics_files, per_cell_files) -> pd.DataFrame:
    """Read each dataset_metrics.tsv and select SUMMARY_COLUMNS into one table.

    - Missing columns are filled with NaN (kept in the requested order).
    - The `dataset` column is recovered from the file path if absent/blank.
    - dataset_metrics.tsv holds one row per dataset; rows are concatenated and
      sorted by dataset name. Duplicate datasets (same name) are collapsed,
      keeping the first non-null value per column.
    - The column 'sample_mean_ploidy' is renamed from
      'tumor_sample_ploidy_mean__dna' if present.
    - The 'tumor_purity_from_config' column is set ONLY from `config_celltype`
      (pre-given / cell-line metadata). If that column is missing or has no
      usable non-Unknown labels, the value remains NA (no fallback to DNA).
    - The 'tumor_purity_from_scWGS' column is ALWAYS set to the value computed
      from `label_dna` alone (NaN if that column is missing / empty), so the
      scWGS-inferred purity is always visible next to the primary one.
    - The 'ref_and_purity_inference_method' column reports the source that
      supplied the primary purity: "cell_line_metadata" if config_celltype was
      usable, otherwise NA (since there is no fallback).
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

        # Rename only the ploidy column; the purity columns will be overridden later.
        rename_map = {
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

    # ---- Compute BOTH purities from per_cell files (independent of each other) ----
    config_purity_dict = {}
    dna_purity_dict = {}
    origin_dict = {}
    for pcf in per_cell_files:
        ds = dataset_from_path(pcf)
        config_purity, dna_purity = compute_tumor_purities(pcf)
        if not pd.isna(config_purity):
            config_purity_dict[ds] = config_purity
            origin_dict[ds] = "cell_line_metadata"
        else:
            origin_dict[ds] = "scWGS_CNV_calls"
        # DNA purity is stored regardless, but it does NOT influence origin_dict.
        if not pd.isna(dna_purity):
            dna_purity_dict[ds] = dna_purity

    # tumor_purity_from_scWGS: always reflects the label_dna computation
    out["tumor_purity_from_scWGS"] = out["dataset"].map(dna_purity_dict)

    # tumor_purity_from_config: only from config_celltype (no fallback)
    out["tumor_purity_from_config"] = out["dataset"].map(config_purity_dict)

    # ref_and_purity_inference_method: only set if config was available
    if origin_dict:
        out["ref_and_purity_inference_method"] = out["dataset"].map(origin_dict)
    else:
        out["ref_and_purity_inference_method"] = np.nan

    return out[SUMMARY_COLUMNS]


# MODIFIED: Added no_normal_cells_set parameter
def write_dataset_summary(input_glob: str, summary_out: str, no_normal_cells_set: set = None) -> pd.DataFrame:
    """Infer inputs, build the single combined table, and write it to disk.

    The file is always written (header-only if no dataset_metrics.tsv is found),
    so the output reliably exists for downstream steps.
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
    
    # MODIFIED: Write a '{NO_NORMAL_SET}'-suffixed copy to disk, but return the clean summary
    summary_out_df = _suffix_dataset_column(summary, no_normal_cells_set or set())
    summary_out_df.to_csv(summary_out, sep="\t", index=False, na_rep='N/A')
    
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
# 4c. Per-dataset metadata for the swarm-grid figure (technology, purity bin)
# ---------------------------------------------------------------------------
def technology_from_dataset(dataset_name: str) -> str:
    """Recover the study / scWGS-scRNA co-sequencing technology from a
    (already-remapped) plotname, e.g. 'wellDR-seq_BCIS106T_chip1_…' -> 'wellDR-seq'.

    Falls back to TECHNOLOGY_UNKNOWN_LABEL if the name doesn't start with one
    of the known prefixes in TECHNOLOGY_MARKERS (e.g. if a dataset slipped
    through without a name2plotname remap).
    """
    prefix = str(dataset_name).split("_", 1)[0]
    if prefix in TECHNOLOGY_MARKERS:
        return prefix
    return TECHNOLOGY_UNKNOWN_LABEL


def should_include_dataset_for_swarm_grid(dataset_name: str) -> bool:
    """Per-dataset inclusion rule for the swarm-grid figure only (does not
    affect the original heatmaps). Currently implements: for wellDR-seq
    datasets, keep only the 'chip1' replicate (literal substring match on the
    dataset name) so a tumor sample split across multiple chips doesn't
    contribute multiple near-duplicate points to the same panel. wellDR-seq
    datasets with no chip token at all (e.g. 'wellDR3',
    'Cellline_mixing_experiment_...') are excluded too, since they don't
    contain the literal substring 'chip1'. Non-wellDR-seq datasets are always
    kept (this rule is a no-op for them)."""
    if technology_from_dataset(dataset_name) != "wellDR-seq":
        return True
    return WELLDR_SEQ_CHIP1_SUBSTRING in str(dataset_name)


def purity_bin_label(purity) -> str:
    """Map a scalar tumor_purity_from_scWGS value to one of PURITY_BIN_LABELS,
    or PURITY_BIN_UNKNOWN_LABEL if NaN / unresolvable."""
    if purity is None or (isinstance(purity, float) and np.isnan(purity)):
        return PURITY_BIN_UNKNOWN_LABEL
    try:
        purity = float(purity)
    except (TypeError, ValueError):
        return PURITY_BIN_UNKNOWN_LABEL
    for lo, hi, label in zip(PURITY_BIN_EDGES[:-1], PURITY_BIN_EDGES[1:], PURITY_BIN_LABELS):
        if lo <= purity < hi:
            return label
    return PURITY_BIN_UNKNOWN_LABEL


def _normalize_metric_key(name: str) -> str:
    """Loose key for matching metric names across naming conventions used in
    this codebase, e.g.:
        'Pearson Correlation Coefficient'                       (spaces)
        'Fraction_of_the_cells_with_inferred_copy_numbers'      (underscores)
        'Tumor/normal classification ROC-AUC (scRNA … vs scWGS)' (slashes,
                                                    parens, hyphens, spaces)
    Lower-cases and strips every non-alphanumeric character, so the
    swarm-grid lookup below still finds a metric even if it's spelled with
    different separators/punctuation than the constant lists above."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_metric_name(requested: str, available_metrics) -> str:
    """Find the actual metric-name string present in `available_metrics` that
    corresponds to `requested`, tolerating space/underscore/case differences.
    Returns `requested` unchanged if no match is found (the caller will then
    correctly report it as missing rather than silently mis-plotting)."""
    if requested in available_metrics:
        return requested
    target_key = _normalize_metric_key(requested)
    for m in available_metrics:
        if _normalize_metric_key(m) == target_key:
            return m
    return requested


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


# MODIFIED: Added no_normal_cells_set parameter
def plot_heatmap(
    agg: pd.DataFrame,
    metric: str,
    outdir: str,
    fmt: str,
    dpi: int,
    title_suffix: str = "",
    filename_suffix: str = "",
    purity_map: dict = None,
    no_normal_cells_set: set = None,
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
    purity_map      : dict {dataset: tumor_purity_from_scWGS} used to decide if
                      an asterisk is appended to the dataset name (purity < 0.2).
    no_normal_cells_set : set of dataset names where no normal cells were found,
                          causing '{NO_NORMAL_SET}' to be appended to the dataset name.
    """
    if purity_map is None:
        purity_map = {}
        
    # MODIFIED: Initialize no_normal_cells_set
    if no_normal_cells_set is None:
        no_normal_cells_set = set()

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

    # --- Row labels (dataset names), with asterisk if scWGS purity < 20% ---
    # MODIFIED: Added '{NO_NORMAL_SET}' suffix logic for no_normal_cells_set
    row_labels = []
    asterisk_needed = False
    double_asterisk_needed = False
    for ds in pivot_mean.index:
        label = ds
        if ds in no_normal_cells_set:
            label += f"{NO_NORMAL_SET}"
            double_asterisk_needed = True
        purity = purity_map.get(ds, np.nan)
        if pd.notna(purity) and purity < 0.2:
            label += LOW_TUMOR_PURITY
            asterisk_needed = True
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

    # --- Footnotes for asterisks ---
    # MODIFIED: Added footnote for double asterisk
    if double_asterisk_needed:
        fig.text(0.02, 0.02, f"{NO_NORMAL_SET} no normal-cell cluster identified by identify_normal_cell_subset.R (fell back to top-scoring cluster)", fontsize=8, ha="left", va="bottom")
    if asterisk_needed:
        fig.text(0.02, 0.04, f"{LOW_TUMOR_PURITY} tumor purity (scWGS) < 20%", fontsize=8, ha="left", va="bottom")

    fig.tight_layout()

    safe_name = re.sub(r"[^\w]+", "_", metric).strip("_")
    out_path = os.path.join(outdir, f"heatmap_{safe_name}{filename_suffix}.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# 5b. Tumor-only heatmaps: one extra figure per metric in `metrics`
# ---------------------------------------------------------------------------
# MODIFIED: Added no_normal_cells_set parameter
def plot_tumor_only_heatmaps(
    data: pd.DataFrame,
    metrics,
    outdir: str,
    fmt: str,
    dpi: int,
    purity_map: dict = None,
    no_normal_cells_set: set = None,
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
        # MODIFIED: Pass no_normal_cells_set to plot_heatmap
        plot_heatmap(
            agg_tumor,
            metric,
            outdir,
            fmt,
            dpi,
            title_suffix=" (tumor cells only)",
            filename_suffix="_tumor_only",
            purity_map=purity_map,
            no_normal_cells_set=no_normal_cells_set,
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
        # MODIFIED: Write suffixed copy to TSV
        agg_tumor_out = _suffix_dataset_column(agg_tumor, no_normal_cells_set)
        agg_tumor_out.to_csv(agg_path, sep="\t", index=False, na_rep='N/A')
        print(f"[tumor-only] aggregated data → {agg_path} ({plotted} metric(s))")


# ---------------------------------------------------------------------------
# 5c. Main-text figure: grid of per-(metric, method) swarmplots
# ---------------------------------------------------------------------------
def build_swarm_grid_data(data: pd.DataFrame, purity_map: dict) -> pd.DataFrame:
    """Assemble the long-format table that feeds `plot_metric_method_swarm_grid`.

    One row per (dataset, method, metric) with the aggregated mean value for
    that combination, plus two per-dataset metadata columns:
        purity_bin  -> one of PURITY_BIN_LABELS / PURITY_BIN_UNKNOWN_LABEL
        technology  -> one of TECHNOLOGY_MARKERS keys / TECHNOLOGY_UNKNOWN_LABEL

    The SWARM_GRID_TUMOR_ONLY_METRICS rows are computed on tumor-only cells
    (via `filter_to_tumor`); the remaining SWARM_GRID_ALL_CELLS_METRICS rows
    are computed on all cells. Metric names are resolved leniently
    (`resolve_metric_name`) so a space/underscore mismatch between the metric
    names requested here and the actual `metric` values in the eval TSVs does
    not silently drop a row.

    A dataset-level filter (`should_include_dataset_for_swarm_grid`) and a
    method-level filter (`SWARM_GRID_EXCLUDED_METHODS`) are both applied here,
    up front, before any tumor-filtering, aggregation, or the >15-method
    warning check below — so that warning reflects the method count that will
    actually appear in the figure, not a pre-exclusion count. Both filters are
    local to the swarm-grid figure and do not affect the original per-metric
    heatmaps.
    """
    all_datasets = data["dataset"].unique()
    keep_datasets = [d for d in all_datasets if should_include_dataset_for_swarm_grid(d)]
    dropped_datasets = sorted(set(all_datasets) - set(keep_datasets))
    if dropped_datasets:
        print(f"[swarm-grid] Excluding {len(dropped_datasets)} dataset(s) per the "
              f"wellDR-seq chip1-only rule: {dropped_datasets}")
    data = data[data["dataset"].isin(keep_datasets)].copy()

    excluded_methods_present = sorted(set(data["method"].unique()) & SWARM_GRID_EXCLUDED_METHODS)
    if excluded_methods_present:
        print(f"[swarm-grid] Excluding {len(excluded_methods_present)} method column(s) "
              f"per SWARM_GRID_EXCLUDED_METHODS: {excluded_methods_present}")
    data = data[~data["method"].isin(SWARM_GRID_EXCLUDED_METHODS)].copy()

    available_all = set(data["metric"].unique())
    tumor_data = filter_to_tumor(data)
    available_tumor = set(tumor_data["metric"].unique())

    agg_all = aggregate(data)
    agg_tumor = aggregate(tumor_data) if not tumor_data.empty else pd.DataFrame(
        columns=["dataset", "method", "metric", "mean", "std", "n", "q1", "q3"])

    frames = []
    for requested_metric in SWARM_GRID_METRICS:
        is_tumor_only = requested_metric in SWARM_GRID_TUMOR_ONLY_METRICS
        source_agg = agg_tumor if is_tumor_only else agg_all
        available = available_tumor if is_tumor_only else available_all

        resolved = resolve_metric_name(requested_metric, available)
        sub = source_agg[source_agg["metric"] == resolved].copy()
        if sub.empty:
            scope = "tumor-only" if is_tumor_only else "all-cells"
            print(f"[swarm-grid] WARNING: {requested_metric!r} ({scope}) had no "
                  f"data at all (resolved lookup name: {resolved!r}); this row "
                  f"of the swarm grid will be empty.")
            continue

        sub["metric"] = requested_metric   # normalize back to the requested/display key
        frames.append(sub[["dataset", "method", "metric", "mean"]])

    if not frames:
        print("[swarm-grid] No requested metrics resolved to any data; "
              "skipping the swarm-grid figure entirely.")
        return pd.DataFrame(columns=["dataset", "method", "metric", "mean",
                                      "purity_bin", "technology"])

    long_df = pd.concat(frames, ignore_index=True)

    # --- Enforce / warn on the method cap ---
    n_methods = long_df["method"].nunique()
    if n_methods > MAX_METHODS_FOR_SWARM_GRID:
        methods_list = ", ".join(sorted(long_df["method"].unique()))
        print("\n" + "!" * 78)
        print(f"!!! WARNING: {n_methods} CNV calling methods found, which exceeds the "
              f"recommended maximum of {MAX_METHODS_FOR_SWARM_GRID} for the "
              f"metric-by-method swarmplot grid.")
        print("!!! The figure will still be generated, but it will likely be too "
              "wide / crowded to read.")
        print(f"!!! Consider filtering the input (e.g. via --metrics or by "
              f"pre-subsetting methods) to <= {MAX_METHODS_FOR_SWARM_GRID} methods.")
        print(f"!!! Methods found: {methods_list}")
        print("!" * 78 + "\n")

    # --- Attach per-dataset purity bin and technology ---
    long_df["purity_bin"] = long_df["dataset"].map(
        lambda ds: purity_bin_label(purity_map.get(ds, np.nan)))
    long_df["technology"] = long_df["dataset"].map(technology_from_dataset)

    return long_df


def plot_metric_method_swarm_grid(
    data: pd.DataFrame,
    purity_map: dict,
    outdir: str,
    fmt: str,
    dpi: int,
):
    """Main-text figure: a grid of seaborn swarmplots, one panel per
    (metric row, method column). Each dot is one dataset; dot colour encodes
    the scWGS-derived tumor-purity bin, dot marker shape encodes the
    scWGS-scRNA co-sequencing technology / study. The overall figure area is
    square, per the figure spec.
    """
    long_df = build_swarm_grid_data(data, purity_map)
    if long_df.empty:
        print("[swarm-grid] No data to plot; skipping the swarm-grid figure.")
        return

    # Rows: only metrics that actually resolved to data, in the canonical order.
    row_metrics = [m for m in SWARM_GRID_METRICS if m in set(long_df["metric"].unique())]
    missing_rows = [m for m in SWARM_GRID_METRICS if m not in row_metrics]
    if missing_rows:
        print(f"[swarm-grid] The following requested rows had no data and will "
              f"be omitted from the grid: {missing_rows}")
    if not row_metrics:
        print("[swarm-grid] No rows resolved to data; skipping the swarm-grid figure.")
        return

    # Columns: methods sorted alphabetically for a stable, reproducible layout.
    # (SWARM_GRID_EXCLUDED_METHODS is already applied upstream, in
    # build_swarm_grid_data, so long_df here is already filtered.)
    col_methods = sorted(long_df["method"].unique())
    if not col_methods:
        print("[swarm-grid] All methods were excluded; skipping the swarm-grid figure.")
        return
    n_rows, n_cols = len(row_metrics), len(col_methods)

    # --- Square figure sizing. Panel WIDTH is fixed (keeps column-label
    #     spacing consistent regardless of grid shape); panel HEIGHT instead
    #     stretches to fill whatever vertical room is available once the
    #     canvas side is set by the width-driven extent, clamped to a
    #     sensible [MIN_PANEL_H, MAX_PANEL_H] range. This avoids the large
    #     blank band that a FIXED panel height left above the grid whenever
    #     the grid was much wider than it was tall (many columns, few rows) —
    #     the canvas was square, but a fixed-height grid inside it wasn't. ---
    panel_w = 1.1     # inches, fixed panel width
    MIN_PANEL_H = 1.0  # inches, floor so panels never get unreadably squat
    MAX_PANEL_H = 4.0  # inches, ceiling so panels don't get absurdly stretched
                        # when n_rows is very small relative to n_cols
    # Inches reserved for the shared legend strip at the bottom. Sized with
    # real clearance on both sides of the legend text itself (not just the
    # bare minimum up to the grid's nominal edge): axes with no drawn swarm
    # points (an empty metric x method cell) can render a tight bbox that
    # dips slightly below the nominal grid bottom edge, so the legend needs
    # headroom to spare, not a hairline fit.
    legend_h = 1.25
    label_pad_left = 1.8   # inches reserved for row labels (metric names) on the left
    label_pad_top = 0.8    # inches reserved for column labels (method names) on top

    grid_w = panel_w * n_cols
    # Candidate canvas side if WIDTH is the binding dimension (the common case
    # here: many method columns, few metric rows). Compute the panel height
    # that would exactly fill the remaining vertical space on that canvas,
    # then clamp it — this is what lets the grid actually reach the canvas
    # edges instead of leaving blank space above/below it.
    width_driven_side = label_pad_left + grid_w
    avail_h_if_width_binding = width_driven_side - label_pad_top - legend_h
    panel_h = avail_h_if_width_binding / n_rows if n_rows > 0 else MIN_PANEL_H
    panel_h = max(MIN_PANEL_H, min(MAX_PANEL_H, panel_h))
    grid_h = panel_h * n_rows

    # The figure canvas is a square whose side is set by whichever of the two
    # candidate extents (label_pad_left + grid_w) or (label_pad_top + grid_h +
    # legend_h) is larger — using the ALREADY-STRETCHED grid_h from above, so
    # if MAX_PANEL_H was hit (grid still shorter than the width-driven side,
    # e.g. very few rows relative to columns) the canvas correctly falls back
    # to being sized by width with the residual blank space this time
    # genuinely unavoidable at readable panel proportions, rather than being
    # silently absorbed. Margins are anchored at FIXED absolute-inch offsets
    # from the canvas edges (not derived by subtracting content size from
    # canvas size), so the grid always sits flush against the left/top label
    # area and the fractions stay valid regardless of which extent was
    # binding. NOTE: this figure is saved WITHOUT bbox_inches="tight" (see the
    # savefig call below) — tight-bbox cropping would re-fit the output to the
    # content's own (non-square) bounding box and defeat this entirely.
    side = max(width_driven_side, label_pad_top + grid_h + legend_h)
    fig_w = fig_h = side

    left_frac = label_pad_left / side
    right_frac = left_frac + grid_w / side
    bottom_frac = legend_h / side
    top_frac = bottom_frac + grid_h / side

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=left_frac,
        right=min(right_frac, 0.995),
        top=min(top_frac, 0.98),
        bottom=bottom_frac,
        wspace=0.15, hspace=0.25,
    )

    # --- Shared y-limits per row (metric), so dots across methods in the
    #     same row are visually comparable. ---
    row_ylims = {}
    for metric in row_metrics:
        vals = long_df.loc[long_df["metric"] == metric, "mean"].dropna()
        if vals.empty:
            row_ylims[metric] = (0.0, 1.0)
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if metric in CORRELATION_METRICS:
            lo, hi = -1.0, 1.0
        elif "ROC-AUC" in metric or "ROC_AUC" in metric or metric in SWARM_GRID_ALL_CELLS_METRICS:
            lo, hi = 0.0, 1.0
        else:
            pad = max(0.02, (hi - lo) * 0.1)
            lo, hi = lo - pad, hi + pad
        row_ylims[metric] = (lo, hi)

    for i, metric in enumerate(row_metrics):
        for j, method in enumerate(col_methods):
            ax = fig.add_subplot(gs[i, j])
            cell = long_df[(long_df["metric"] == metric) & (long_df["method"] == method)]

            if not cell.empty:
                # seaborn's swarmplot has no native categorical->marker-shape
                # channel, only hue->colour. To get colour=purity_bin AND
                # shape=technology in one panel, draw one swarmplot call per
                # technology subgroup, each with its own `marker`.
                # Limitation: swarm-jitter packing is then computed
                # independently within each technology subgroup rather than
                # jointly across the whole cell, so two dots of different
                # technologies at a near-identical y-value may render closer
                # together (or overlap) than a single joint swarm call would
                # have allowed. With the low point-per-cell counts typical
                # here (one dot per dataset) this is a minor, expected
                # trade-off rather than a bug.
                for tech, marker in list(TECHNOLOGY_MARKERS.items()) + \
                        [(TECHNOLOGY_UNKNOWN_LABEL, TECHNOLOGY_MARKER_UNKNOWN)]:
                    sub = cell[cell["technology"] == tech]
                    if sub.empty:
                        continue
                    if i % 4 > 0 or j % 4 > 0:
                        warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
                    sns.swarmplot(
                        data=sub,
                        x=np.zeros(len(sub)),
                        y="mean",
                        hue="purity_bin",
                        hue_order=PURITY_BIN_LABELS + [PURITY_BIN_UNKNOWN_LABEL],
                        palette=PURITY_BIN_COLORS,
                        marker=marker,
                        ax=ax,
                        size=6.5,
                        linewidth=0.4,
                        edgecolor="black",
                        legend=False,
                    )
                    warnings.filterwarnings("default", category=UserWarning, module="seaborn")

            ax.set_ylim(*row_ylims[metric])
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([])
            ax.set_xlabel("")

            if j == 0:
                display_name = SWARM_GRID_ROW_DISPLAY_NAMES.get(metric, metric)
                ax.set_ylabel(display_name, fontsize=8, rotation=0,
                               ha="right", va="center", labelpad=8)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.tick_params(axis="y", labelsize=6)

            if i == 0:
                ax.set_title(method.replace('_', '\n'), fontsize=8.5) #(, rotation=35, ha="left", va="bottom")
            elif i == len(row_metrics) - 1:
                ax.set_xlabel(method.replace('_', '\n'), fontsize=8.5) #(, rotation=35, ha="left", va="bottom")

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.suptitle(
        "scRNA-seq CNV caller performance across datasets, methods, and metrics",
        fontsize=12, fontweight="bold", y=0.995,
    )

    # --- Shared legend: purity-bin colour swatches + technology marker shapes ---
    from matplotlib.lines import Line2D
    legend_handles = []
    for label in PURITY_BIN_LABELS + [PURITY_BIN_UNKNOWN_LABEL]:
        legend_handles.append(Line2D(
            [0], [0], marker="o", linestyle="", markersize=7,
            markerfacecolor=PURITY_BIN_COLORS[label], markeredgecolor="black",
            markeredgewidth=0.4, label=f"Purity {label}"))
    for tech, marker in TECHNOLOGY_MARKERS.items():
        legend_handles.append(Line2D(
            [0], [0], marker=marker, linestyle="", markersize=7,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.8, label=str(tech).replace('wellDR-seq', 'wellDR-seq_chip1')))
    # Only show the "Other" (unrecognized-technology) legend entry if at
    # least one plotted datapoint actually falls into that bucket; otherwise
    # it's a dead legend entry that never appears in the figure.
    if (long_df["technology"] == TECHNOLOGY_UNKNOWN_LABEL).any():
        legend_handles.append(Line2D(
            [0], [0], marker=TECHNOLOGY_MARKER_UNKNOWN, linestyle="", markersize=7,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.8, label=TECHNOLOGY_UNKNOWN_LABEL))

    # Anchor the legend a small margin above the absolute figure bottom edge
    # (not flush at y=0.0), so there is real clearance below the legend text
    # as well as above it, within the legend_h zone reserved earlier.
    legend_y_anchor = 0.15 / side   # ~0.15in margin below the legend itself
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y_anchor),
    )

    out_path = os.path.join(outdir, f"swarm_grid_metric_by_method.{fmt}")
    # Intentionally NOT bbox_inches="tight" here: this figure's square-canvas
    # sizing is computed explicitly above, and tight-bbox would crop back to
    # the (non-square) bounding box of the drawn content, undoing it.
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(out_path + '.png', dpi=dpi)
    plt.close(fig)
    print(f"  → {out_path}")
    if n_cols > MAX_METHODS_FOR_SWARM_GRID:
        print(f"  (NOTE: this figure has {n_cols} method columns, above the "
              f"recommended {MAX_METHODS_FOR_SWARM_GRID} — see warning above.)")


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


def is_in_plots(kw, plots_str, kw2='all'):
    plots = plots_str.split(',')
    return (kw2 in plots) or (kw in plots)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    # NEW: Build the set of datasets with no normal cells identified
    no_normal_cells_set = build_no_normal_cells_set(args.no_normal_cells_glob)

    # --- Single combined summary table ---
    # Built first (and independently of the heatmap data) so it is produced even
    # if the benchmark eval TSVs are unusable. The dataset_metrics.tsv inputs are
    # inferred from --input_glob; no separate path argument is needed.
    summary_out = args.summary_out or os.path.join(args.outdir, "dataset_summary.tsv")
    # MODIFIED: Pass no_normal_cells_set
    summary = write_dataset_summary(args.input_glob, summary_out, no_normal_cells_set=no_normal_cells_set)

    # Build purity map from scWGS-derived tumor_purity_from_scWGS.
    purity_map = {}
    if not summary.empty and "tumor_purity_from_scWGS" in summary.columns:
        for _, row in summary.iterrows():
            ds = row["dataset"]
            pur = row["tumor_purity_from_scWGS"]
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

    if is_in_plots('supp', args.plots):
        print(f"\nPlotting {len(metrics_to_plot)} metric heatmaps …")
        for metric in metrics_to_plot:
            # MODIFIED: Pass no_normal_cells_set
            plot_heatmap(agg, metric, args.outdir, args.fmt, args.dpi,
                         purity_map=purity_map,
                         no_normal_cells_set=no_normal_cells_set)

    if is_in_plots('overview', args.plots):
        print("\nPlotting overview …")
        plot_overview(agg, args.outdir, args.fmt, args.dpi)

    if is_in_plots('supp', args.plots):
        # --- Tumor-only heatmaps (additional figures) ---
        if not args.no_tumor_only:
            tumor_metrics = [m.strip() for m in args.tumor_only_metrics.split(",")
                             if m.strip()]
            if tumor_metrics:
                print(f"\nPlotting {len(tumor_metrics)} tumor-only metric heatmap(s) …")
                # MODIFIED: Pass no_normal_cells_set
                plot_tumor_only_heatmaps(data, tumor_metrics,
                                          args.outdir, args.fmt, args.dpi,
                                          purity_map=purity_map,
                                          no_normal_cells_set=no_normal_cells_set)
            else:
                print("\n[tumor-only] --tumor_only_metrics is empty; "
                      "no tumor-only heatmaps written.")

    if is_in_plots('main', args.plots):
        # --- Main-text figure: grid of per-(metric, method) swarmplots ---
        print("\nPlotting metric-by-method swarm grid …")
        plot_metric_method_swarm_grid(data, purity_map, args.outdir, args.fmt, args.dpi)

    # Also save the aggregated table as TSV for downstream use
    agg_path = os.path.join(args.outdir, "aggregated_results.tsv")
    # MODIFIED: Write suffixed copy to TSV
    agg_out = _suffix_dataset_column(agg, no_normal_cells_set)
    agg_out.to_csv(agg_path, sep="\t", index=False, na_rep='N/A')
    print(f"\nAggregated data → {agg_path}")
    print("Done.")


if __name__ == "__main__":
    main()

