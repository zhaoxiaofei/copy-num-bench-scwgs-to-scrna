# Benchmark scRNA-seq CNV Callers

## Overview

This repository provides a benchmarking pipeline to evaluate and select the best scRNA-seq-based Copy Number Variation (CNV) callers. Currently, the pipeline evaluates seven different tools: **inferCNA**, **inferCNV**, **CopyKat**, **SCEVAN**, **Numbat**, **CaSpER**, and **CONICSmat**.

To establish highly accurate, cell-specific ground truths for these scRNA-seq-based callers, this repository relies on single-cell Whole Genome Sequencing (scWGS) data from scWGS-scRNA co-sequencing experiments (where each individual cell is sequenced by both modalities). It selects the optimal scWGS-based CNV caller(s) utilizing benchmarking results from [copy-num-bench-scwgs](https://github.com/zhaoxiaofei/copy-num-bench-scwgs).

## Installation

To set up the environment and install all necessary dependencies, navigate to this repository's root directory and run the provided installation script:
```bash
# Clone the repo and navigate into it (if you haven't already)
# Install conda and micromamba (if you havent's already)
bash -evx install-end2end.sh cnb_scrna1 ginkgo_env1 # cnb: copy-num-bench, scrna: single-cell RNA-seq
```

Next, run `micromamba activate ginkgo_env1` (if you haven't already), and then install the scWGS-based CNV caller Ginkgo from [https://github.com/zhaoxiaofei/ginkgo](https://github.com/zhaoxiaofei/ginkgo) and build the hg19 genome (the build files will be used by Ginkgo). Then, modify the `config_template.yaml` file accordingly to match the path of Ginkgo in your file system.

## Usage

Follow these steps to configure and run the benchmarking workflow:

1. **Set the Configuration File:** Define your target YAML configuration file. 
   *(Example: `YAML=coseq_configs/config_BCIS106T_chip1_SAMN48409192_SRR33511671.yaml`)*
2. **Prepare the Data:** Download the FASTQ files specified in your `$YAML` file, and edit the `$YAML` paths if necessary to match your local environment.
   * *Note:* For instructions on generating the FASTQ files specified in the config, please refer to the documentation provided by the authors of the datasets. Examples include [scONE-seq-data-processing](https://github.com/0YuLei0/scONE-seq-data-processing) and [wellDR-seq](https://github.com/navinlabcode/wellDR-seq).
3. **Allocate Resources:** Specify the number of CPU cores available for the pipeline.
   *(Example: `NUM_CORES=64`)*
4. **Execute Snakemake:** Run the pipeline using the following command:
```bash
# run: micromamba activate cnb_scrna1 (if you haven't already)
snakemake \
  --configfile config_template.yaml ${YAML} \
  -s snakemake_pipeline/new_workflow.snake \
  --rerun-incomplete \
  --printshellcmd \
  --cores ${NUM_CORES}
```

## Results

Upon completion, the benchmarked performance metrics and visualizations will be generated and stored in the following directory structure:

`results/<dataset_name>/evaluation/`

Inside these directories, you will find:
* **TSV files:** Raw, tabular data containing the benchmarking results.
* **PDF/PNG files:** Visual plots illustrating the performance comparisons.

## Plotting CNV Heatmaps (`plot_cnv_heatmaps.py`)

After the Snakemake pipeline finishes, you can summarize and visualize the
benchmarking results across all datasets/methods with the standalone Python
script [`plot_cnv_heatmaps.py`](./plot_cnv_heatmaps.py) located at the
repository root.

### What it produces

For every run the script writes the following artifacts into `--outdir`
(`./heatmaps` by default):

* **Per-metric heatmaps** — one figure per benchmark metric (e.g.
  *Pearson Correlation Coefficient*, *CopyNumber gain ROC-AUC*).  
  Rows = datasets, columns = scRNA-seq CNV callers (inferCNA, inferCNV,
  CopyKat, SCEVAN, Numbat, CaSpER, CONICSmat). Each cell shows
  `mean ± sd` across all cells in that dataset×method pair, with the color
  encoding the mean.
  Caller columns are ordered by each tool's exact publication date and the
  column labels carry the publication year (e.g. `inferCNV 2014`,
  `SCEVAN 2023`), matching the caller figures of the gDNA benchmark
  repository.
* **Tumor-only heatmaps** — additional heatmaps restricted to tumor cells
  (labelled via the `celltype` column, falling back to `celltype_dna`) for
  the metrics listed in `--tumor_only_metrics`.
* **Overview heatmap** — a single grand-mean overview across all
  dataset×method pairs.
* **Main-text swarm grid** — a grid of per-(metric, method) swarmplots, one
  dot per dataset, colored by scWGS-derived tumor purity and shaped by
  scDNA-scRNA co-sequencing technology.
* **`dataset_summary.tsv`** — a single combined per-dataset table with the
  columns `dataset`, `n_cells_total`, `tumor_purity_from_config`,
  `tumor_purity_from_scWGS`, `sample_mean_ploidy`,
  `scRNA_reads_per_cell_mean`, `scRNA_reads_per_cell_median`,
  `scWGS_reads_per_cell_mean`, `scWGS_reads_per_cell_median`,
  `ref_and_purity_inference_method`.
* **`aggregated_results.tsv`** — the long-form mean ± sd table that backs
  every heatmap (handy for downstream statistical testing).
* **`stats.pairwise.tsv` / `stats.friedman.tsv` / `stats.json`** — the
  pairwise statistical tests (see the next section), run by default on the
  same filtered method set as the figures.
* **`stats.pairwise.tex`** — a copy-paste-ready booktabs LaTeX table of the
  pairwise comparisons; every row carries exactly four statistics, in this
  order and nothing else: the effective sample size **n** (number of
  independent materials — the patients / cell lines behind the datasets),
  the Holm-adjusted **p** (two-sided Wilcoxon signed-rank), the effect size
  **r** (matched-pairs rank-biserial correlation; positive = the reference
  performs better) and the **95% CI** of r (percentile bootstrap over the
  independent materials).

### Statistical tests (`stat_tests.py`)

`stat_tests.py` (repository root, next to `plot_cnv_heatmaps.py`) adds the
same statistical testing layer that the scWGS benchmark repository
[copy-num-bench-scwgs](https://github.com/zhaoxiaofei/copy-num-bench-scwgs)
uses, adapted to the co-sequencing design:

* **Design:** every method is evaluated on the same cells of the same
  datasets, so per-cell metrics are paired (blocked) by cell. The
  independent experimental unit is the biological **material** (patient /
  cell line) behind the datasets — derived from the dataset name, so chips of
  one patient (`BCIS106T_chip1/chip2`, `ECIS44T_chip1-5`, …), one cell line
  across technologies (`HCT116` in DNTR-seq and scONE-seq) and
  reference-cell configuration variants of one dataset collapse into ONE
  unit. `--stats_cluster_key dataset` analyses at the per-dataset level
  (sensitivity analysis); `none` reverts to the naive per-cell tests
  (discouraged: pseudoreplication — see the module docstring for the
  measured demonstration of the failure mode).
* **Tests:** Friedman omnibus per metric across methods (on per-material
  medians); two-sided Wilcoxon signed-rank + exact sign test post-hoc on the
  per-material medians of the paired per-cell differences (reference method
  vs. every other; `--stats_all_pairs` for all pairs); Holm-Bonferroni
  correction within each metric family.
* **Effect size:** matched-pairs rank-biserial correlation r (positive = the
  reference performs better) with a 95% percentile-bootstrap CI obtained by
  resampling the materials; the paired common-language effect size and the
  median difference with its CI stay in the TSVs.
* **Diagnostics:** per comparison, the ICC of the paired differences within
  materials, the design effect, the effective sample size and the
  naive-vs-cluster P-inflation ratio — the degree of dependence is measured,
  not assumed away.

By default `plot_cnv_heatmaps.py` runs the tests (disable with `--no_stats`)
and writes the tables next to the figures. The reference method ("scenario
A") defaults to `infercnv`; change it with `--stats_reference`. Standalone
use on the raw evaluation TSVs:

```bash
python stat_tests.py -i 'results/*solo-genefull_output/evaluation/*.tsv' \
    -o heatmaps/stats --reference infercnv
python stat_tests.py -i aggregated_per_cell_long.tsv -o heatmaps/stats  # unified long TSV
python stat_tests.py -o heatmaps/stats --latex-table   # re-print the table
python test_stat_tests.py                             # self-test
```

### Ordering / winner / top-2 tests (`winner_analysis.py`)

The pairwise table answers "does the reference outperform method b?";
`winner_analysis.py` (repository root, next to `stat_tests.py`) answers the
three questions a reference-based comparison cannot — it needs **no
reference method** and treats all callers symmetrically:

1. **Does an approximate ordering of the callers exist?** Friedman omnibus
   per metric (H0: the methods are exchangeable across materials) plus
   Kendall's W with a 95% bootstrap CI and the bootstrap rank-stability
   (mean Spearman ρ between the observed ordering and the orderings of
   resampled materials; reproduction rate of the leading group). The
   ordering is "approximate" when it is supported (Friedman P ≤ α) but at
   least one adjacent pair of the ordering is not separable at the
   family-wise level.
2. **Does a unique winner exist?** Methods are ordered by their Friedman
   mean rank over the per-material method medians; the step-down procedure
   walks the ordered methods from the top and cuts at the first separable
   adjacent pair. A unique winner exists iff M1 vs M2 is separable
   (Holm-adjusted two-sided P ≤ α and rank-biserial r > 0).
3. **Does a top-2 group exist?** iff M1 vs M2 is NOT separable AND M2 vs M3
   IS separable (two statistically indistinguishable leaders ahead of every
   remaining method). The general verdict covers any leading-group size
   (g = 1 unique winner, g = 2 top-2, g = k no separation).

Cross-checks: the Nemenyi critical-difference grouping on the Friedman mean
ranks; the leading group's internal homogeneity and its separation from
below verified on **all** cross pairs (not only the adjacent ones); lower-
is-better metrics (e.g. runtime) can be declared with
`--lower-is-better METRIC` so that "larger is better" always holds after
orientation.

Outputs (prefix `-o`, default `heatmaps/winner`): `winner.order.tsv`
(per metric and method: rank, mean rank, median, dominance counts,
group memberships), `winner.stepdown.tsv` (one row per adjacent step-down
comparison with n, p, p_holm, r and the 95% bootstrap CI of r plus the
per-metric verdict), `winner.stepdown.tex` (booktabs table; every row
carries exactly n, p, r and the 95% CI of r) and `winner.json` (settings,
per-metric verdicts, consensus across metrics). By default
`plot_cnv_heatmaps.py` runs this analysis next to the pairwise tests
(disable with `--no_winner_analysis`; it also runs when the pairwise table
was skipped because of a bogus `--stats_reference`).

```bash
python winner_analysis.py -i 'results/*solo-genefull_output/evaluation/*.tsv' \
    -o heatmaps/winner
python winner_analysis.py -i aggregated_per_cell_long.tsv -o heatmaps/winner
python winner_analysis.py -o heatmaps/winner --latex-table  # re-print the table
python test_stat_tests.py                                  # self-test (Parts A-F)
```

### Comparison with the best - Hsu's MCB (`mcb.py`)

`mcb.py` (repository root, next to `stat_tests.py` / `winner_analysis.py`) adds
the classical interval view of the same reference-free questions. Per metric
(= per Fig. 5 swarm-grid panel) it builds, for every method configuration i,
a **simultaneous 95% confidence interval** for

    theta_i - max_{j != i} theta_j        ("method i versus the best of the others")

on the same material-level units as the other analyses: an interval entirely
below 0 = significantly **inferior** to the best; entirely above 0 = the
**unique winner** (possible only for the sample-best method); brackets 0 =
**indistinguishable from the best**, i.e. a member of the leading group (a
group of size 2 is the "top-2" of the manuscript's observations — the
interval-based counterpart of the step-down leading group). The intervals are
Tukey-style projections of studentized **cluster bootstrap max-|t|** bands
(every replicate recomputes its own per-pair standard error; seeded and
reproducible); a studentized-range cross-check and the Friedman omnibus
accompany every family, and the per-metric verdicts plus the consensus across
metrics are written to the JSON.

Outputs (prefix `-o`, default `heatmaps/mcb`): `mcb.tsv` (one row per
(metric, method): gap to the best, its simultaneous MCB interval, the
MCB-adjusted one-sided p, the rank-biserial r versus the best competitor with
its bootstrap CI, verdicts), `mcb.tex` (booktabs table, one section per
metric; every row carries **exactly n, p, r and the 95% CI** of the gap to
the best) and `mcb.json`. By default `plot_cnv_heatmaps.py` runs this
analysis next to the winner analysis (disable with `--no_mcb_analysis`; it
needs no reference method, so it also runs when the pairwise table was
skipped because of a bogus `--stats_reference`). Lower-is-better metrics are
declared with `--lower-is-better` exactly as in `winner_analysis.py`.

```bash
python mcb.py -i 'results/*solo-genefull_output/evaluation/*.tsv' -o heatmaps/mcb
python mcb.py -i aggregated_per_cell_long.tsv -o heatmaps/mcb --lower-is-better runtime
python mcb.py -o heatmaps/mcb --latex-table     # re-print the table
python test_stat_tests.py                       # Part F covers the MCB layer
```

### Fig. 5 source data

`plot_cnv_heatmaps.py` writes `heatmaps/fig5_source_data.tsv` plus
`fig5_source_data.meta.json` next to the swarm grid: one row per
(dataset x method x metric) with the plotted mean value, the scWGS-derived
tumor-purity bin (dot colour) and the co-sequencing protocol (dot marker),
with the row/column display names in the sidecar — the data necessary and
sufficient to re-plot Fig. 5 without re-running any caller. The scWGS
companion repository's `entire_pipeline.sh` collects this table (and the
scRNA MCB families) into the submission package's `source_data/` directory.

### Prerequisites

The script only needs Python 3 with `matplotlib`, `numpy`, `pandas`, and
`seaborn`. The easiest way to satisfy these is to activate the same conda
environment the rest of the pipeline uses:

```bash
micromamba activate cnb_scrna1
```

### Inputs the script expects

The script reads the TSVs that the Snakemake pipeline writes under
`results/<dataset>_<quant>_output/evaluation/`. The default glob
`./results/*solo-genefull_output/evaluation/*.tsv` matches the
10x STARsolo `GeneFull` quantification used by the standard workflow.
For every evaluation directory the glob touches, the script also reads
two sibling files automatically — `dataset_metrics.tsv` and
`per_cell_metrics.tsv` (no separate flag needed).

A few descriptive TSVs that share the same directory
(`annotation_agreement.tsv`, `annotation_discordant_cells.tsv`,
`cnv_event_sizes.tsv`, `dataset_summary.tsv`) are skipped automatically.
Any file whose basename contains the substring `_hg19_with` is also
excluded, so the hg19-based repeats are never mixed into the hg38 figures.

### Basic usage

From the repository root, after the Snakemake pipeline has finished:

```bash
python plot_cnv_heatmaps.py \
  --input_glob "./results/*solo-genefull_output/evaluation/*.tsv" \
  --outdir ./heatmaps
```

### Common options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--input_glob` | `./results/*solo-genefull_output/evaluation/*.tsv` | Glob for the evaluation TSVs. Per-dataset `dataset_metrics.tsv`/`per_cell_metrics.tsv` files are inferred from the directories this glob matches. |
| `--outdir` | `./heatmaps` | Output directory (created if missing). |
| `--metrics` | all metrics found | Comma-separated subset to plot (e.g. `"Pearson Correlation Coefficient,CopyNumber gain F-score"`). |
| `--file_pattern` | (none) | Restrict to files whose basename contains this substring (e.g. `without_preclassified_cells` or `with_preclassified_cells`). |
| `--summary_out` | `<outdir>/dataset_summary.tsv` | Path for the combined summary TSV. |
| `--tumor_only_metrics` | `Pearson Correlation Coefficient,Spearman Correlation Coefficient,CopyNumber gain ROC-AUC,CopyNumber loss ROC-AUC` | Metrics that also get a tumor-only heatmap. Pass `""` (empty string) to disable, or use `--no_tumor_only`. |
| `--no_tumor_only` | off | Skip tumor-only heatmaps entirely. |
| `--fmt` | `pdf` | Figure format: `pdf`, `png`, or `svg`. |
| `--dpi` | `300` | DPI for raster formats (`png`). |
| `--plots` | `all,overview,main,supp` | Comma-separated subset of plot types to produce. Use `all` to keep the default; pass e.g. `supp,overview` to skip the swarm grid. |
| `--no_normal_cells_glob` | `./results/*_bams/rna/*.cluster_stat_zero_ref_cells.rds` | Glob for sentinel `.rds` files written by `scripts/identify_normal_cell_subset.R` when no normal-cell cluster was found. Matching datasets are flagged with `$^B$` in plots and TSVs. |

### Examples

Plot only the "without preclassified cells" benchmark TSVs, write PNGs at
150 DPI, and put the summary table alongside the figures:

```bash
python plot_cnv_heatmaps.py \
  --input_glob "./results/*solo-genefull_output/evaluation/*.tsv" \
  --file_pattern "without_preclassified_cells" \
  --outdir ./heatmaps_without_preclassified \
  --fmt png --dpi 150 \
  --summary_out ./heatmaps_without_preclassified/dataset_summary.tsv
```

Restrict the heatmaps to two specific metrics and drop the tumor-only
panels:

```bash
python plot_cnv_heatmaps.py \
  --input_glob "./results/*solo-genefull_output/evaluation/*.tsv" \
  --metrics "Pearson Correlation Coefficient,CopyNumber gain ROC-AUC" \
  --no_tumor_only \
  --outdir ./heatmaps_pearson_gainAUC
```

Generate only the main-text swarm grid (no heatmaps):

```bash
python plot_cnv_heatmaps.py \
  --input_glob "./results/*solo-genefull_output/evaluation/*.tsv" \
  --plots main \
  --outdir ./heatmaps_main
```

## Acknowledgments

This repository uses the codebase at [colomemaria/benchmark_scrnaseq_cnv_callers](https://github.com/colomemaria/benchmark_scrnaseq_cnv_callers) as its code-structure template.
