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
