# ------------------------------------------------------------------------------
# Run CONICSmat - Part 1: obtain gene positions (LOCAL GTF VERSION)
# ------------------------------------------------------------------------------
# Revised with the prompts at https://www.doubao.com/chat/38421549834942978

log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

# ------------------------------------------------------------------------------
print("Load libraries")
# ------------------------------------------------------------------------------

library(CONICSmat)
library(rtracklayer)  # newly adeed to read GTF (which can be gzipped)
library(dplyr)        # newly added to manipulate data
library(stringr)

# ------------------------------------------------------------------------------
print("Get input parameters from snakemake")
# ------------------------------------------------------------------------------

input_expression_df <- snakemake@input$expression_df

input_species <- snakemake@params$species
if(is.null(input_species)){
  input_species <- "human"
}
print(paste("Running CONICSmat for the following organism:", input_species))

output_gene_pos_df <- snakemake@output$gene_pos_df

input_gtf <- snakemake@input$gtf  # In snakemake rule: add input: gtf="path/to/gtf"

# ------------------------------------------------------------------------------
print("Load and process GTF file locally")
# ------------------------------------------------------------------------------

expr_df <- as.matrix(read.table(input_expression_df, sep="\t", header=T, row.names=1, check.names=F))

print("Reading GTF... This may take a while.")
gtf_gr <- import(input_gtf)

gtf_genes <- gtf_gr[gtf_gr$type == "gene"]

# Manually debugged by looking at the following link:
# https://github.com/Neurosurgery-Brain-Tumor-Center-DiazLab/CONICS/blob/053cf267cc819694096d923f17081937ed1a53ea/CONICSmat/R/GetPositions.R#L12
# In CONICSmat, getGenePositions typically returns: gene_ID< hgnc_symbol, chromosome_name, start_position, end_position
gene_pos_df <- as.data.frame(gtf_genes) %>%
  select(
    chromosome_name = seqnames,
    start_position = start,
    end_position = end,
    gene_name,
    gene_id,
    strand
  ) %>%
  mutate(chromosome_name = str_remove(as.character(chromosome_name), "^chr"))
# str_remove: from https://www.doubao.com/chat/38421803244434178

gene_pos_df$hgnc_symbol <- gene_pos_df$gene_name

if (input_species == "mouse") {
   # If your input is GeneID，please unc-comment
   # gene_pos_df$hgnc_symbol <- gene_pos_df$gene_id
}

input_genes <- rownames(expr_df)

# matching, note: match will keep the order in input_genes
matched_indices <- match(input_genes, gene_pos_df$hgnc_symbol)
gene_pos <- gene_pos_df[matched_indices, ]

# check lost genes
lost_genes <- sum(is.na(gene_pos$hgnc_symbol))
print(paste("Total genes in input:", length(input_genes)))
print(paste("Genes successfully mapped to GTF:", length(input_genes) - lost_genes))
print(paste("Genes missing (will be NA):", lost_genes))

# cleanup CONICSmat does not need gene_id/gene_name so keep the core columns
# BUT the first column cannot be missed!
gene_pos <- gene_pos[, c("gene_id", "hgnc_symbol", "chromosome_name", "start_position", "end_position")]

# ------------------------------------------------------------------------------
print("Save results")
# ------------------------------------------------------------------------------

write.table(gene_pos, file = output_gene_pos_df, sep = "\t", row.names = FALSE)

# ------------------------------------------------------------------------------
print("SessionInfo:")
# ------------------------------------------------------------------------------
sessionInfo()

