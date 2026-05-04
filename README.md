# Benchmark scRNA-seq CNV Callers

## Overview

This repository provides a benchmarking pipeline to evaluate and select the best scRNA-seq-based Copy Number Variation (CNV) callers. Currently, the pipeline evaluates seven different tools: **inferCNA**, **inferCNV**, **CopyKat**, **SCEVAN**, **Numbat**, **CaSpER**, and **CONICSmat**.

To establish highly accurate, cell-specific ground truths for these scRNA-seq-based callers, this repository relies on single-cell Whole Genome Sequencing (scWGS) data from scWGS-scRNA co-sequencing experiments (where each individual cell is sequenced by both modalities). It selects the optimal scWGS-based CNV caller(s) utilizing benchmarking results from [copy-num-bench-scwg](https://github.com/zhaoxiaofei/copy-num-bench-scwg).

## Installation

To set up the environment and install all necessary dependencies, navigate to this repository's root directory and run the provided installation script:
```bash
# Clone the repo and navigate into it (if you haven't already)
bash -evx install-end2end.sh
```

Next, install the scWGS-based CNV caller Ginkgo from [https://github.com/zhaoxiaofei/ginkgo](https://github.com/zhaoxiaofei/ginkgo) and build the hg19 genome (the build files will be used by Ginkgo). Then, modify the `config_template.yaml` file accordingly to match the path of Ginkgo in your file system.

## Usage

Follow these steps to configure and run the benchmarking workflow:

1. **Set the Configuration File:** Define your target YAML configuration file. 
   *(Example: `YAML=configs/config_BCIS106T_chip1_SAMN48409192_SRR33511671.yaml`)*
2. **Prepare the Data:** Download the FASTQ files specified in your `$YAML` file, and edit the `$YAML` paths if necessary to match your local environment.
   * *Note:* For instructions on generating the FASTQ files specified in the config, please refer to the documentation provided by the authors of the datasets. Examples include [scONE-seq-data-processing](https://github.com/0YuLei0/scONE-seq-data-processing) and [wellDR-seq](https://github.com/navinlabcode/wellDR-seq).
3. **Allocate Resources:** Specify the number of CPU cores available for the pipeline.
   *(Example: `NUM_CORES=64`)*
4. **Execute Snakemake:** Run the pipeline using the following command:
```bash
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

## Acknowledgments

This repository uses the codebase at [colomemaria/benchmark_scrnaseq_cnv_callers](https://github.com/colomemaria/benchmark_scrnaseq_cnv_callers) as its code-structure template.
