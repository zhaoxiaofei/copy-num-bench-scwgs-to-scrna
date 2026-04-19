This directory contains the partial config files.
Each partial config file contains the assigned valuse for the following variables from the related dataset:
dataset, scWGS\_scRNA\_prefix, and cellname\_celltype\_scWGS\_scRNA\_tup\_list.
You should download the relevant dataset specified in its associated partial config file.
It is recommended to use https://sra-explorer.info/ to facilitate the download.

To execute a workflow, please run snakemake in this manner:
`bash
cd $PWD && snakemake -s ../snakemake_pipeline/new_workflow.snake --configfile ../config\_template.yaml --configfile <a_file_in_this_directory>.yaml ...
`

