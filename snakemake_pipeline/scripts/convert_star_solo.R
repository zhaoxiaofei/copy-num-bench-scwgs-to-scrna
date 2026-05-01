#!/usr/bin/env Rscript

# https://chat.deepseek.com/a/chat/s/701a7b85-b3ea-46a4-9dc3-0bfae91cf72d
# https://chat.deepseek.com/share/44q5iyc6zl4yd095j0

# Snakemake input/output objects
mtx_gene        <- snakemake@input[["mtx_gene"]]
barcodes_gene   <- snakemake@input[["barcodes_gene"]]
features_gene   <- snakemake@input[["features_gene"]]
mtx_genefull    <- snakemake@input[["mtx_genefull"]]
barcodes_genefull <- snakemake@input[["barcodes_genefull"]]
features_genefull <- snakemake@input[["features_genefull"]]

out_gene        <- snakemake@output[["gene_tsv"]]
out_genefull    <- snakemake@output[["genefull_tsv"]]

# Load required packages (install if missing)
packages <- c("Matrix", "data.table")
for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg, repos = "https://cloud.r-project.org")
        library(pkg, character.only = TRUE)
    }
}

# Function to convert STAR solo output to cell x gene CSV
convert_star_solo <- function(mtx_file, barcodes_file, features_file, output_tsv) {
    # Read sparse matrix (genes x cells)
    mat <- readMM(mtx_file)
    # Read barcodes (cells) and features (genes)
    cells <- fread(barcodes_file, header = FALSE)[[1]]
    genes <- fread(features_file, header = FALSE)[[2]]
    rownames(mat) <- genes
    colnames(mat) <- cells

    df <- as.data.frame(as.matrix(mat))
    df <- cbind(V1 = rownames(df), df)

    # Write as CSV (genes = rows, cells = columns)
    fwrite(df, file = output_tsv, row.names = FALSE, sep = "\t")
    message("Written: ", output_tsv, " (", nrow(df), " genes x ", ncol(df)-1, " cells)")
}

# Convert both feature sets
convert_star_solo(mtx_gene, barcodes_gene, features_gene, out_gene)
convert_star_solo(mtx_genefull, barcodes_genefull, features_genefull, out_genefull)

message("All conversions completed.")
