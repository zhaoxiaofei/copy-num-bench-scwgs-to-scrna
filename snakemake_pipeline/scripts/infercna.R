# ------------------------------------------------------------------------------
# Run infercna for CNV inference from scRNA-seq data
# infercna infers copy-number aberrations from scRNA-seq by comparing expression
# profiles to a reference set of normal cells
# ------------------------------------------------------------------------------
log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

# ------------------------------------------------------------------------------
print("Load libraries")
# ------------------------------------------------------------------------------
library(infercna)
library(data.table)

# ------------------------------------------------------------------------------
print("Get input parameters from snakemake")
# ------------------------------------------------------------------------------
input_matrix <- snakemake@input$matrix
input_annot <- snakemake@input$annot
input_ref_groups <- snakemake@input$ref_groups

# Set genome version (human or mouse)
input_genome <- snakemake@params$genome
if(is.null(input_genome)){
  input_genome <- "hg19"
}
print(paste("Running infercna with genome:", input_genome))

output_file <- snakemake@output$cnv_file

# ------------------------------------------------------------------------------
print("Execute infercna")
# ------------------------------------------------------------------------------
# Load dataset matrix
data_matrix <- fread(input_matrix)

# Format into a matrix
gene_names <- data_matrix$V1
data_matrix$V1 <- NULL
data_matrix <- as.matrix(data_matrix)
rownames(data_matrix) <- gene_names

# Create output directory
output_dir <- dirname(output_file)
dir.create(output_dir, recursive = TRUE)

# Set reference cells
if(is.null(input_annot)){
  print("No reference cells defined, infercna will use all cells as reference")
  ref_cells <- NULL
} else {
  print("Reference cells for infercna defined")
  # Extract reference cells
  annotation <- fread(input_annot, header = FALSE)
  # Read reference groups (saved in one tsv file)
  ref_groups <- read.table(input_ref_groups, header = TRUE)
  ref_cells <- annotation$V1[annotation$V2 %in% ref_groups$ref_groups]
  if(length(ref_cells) == 0){
    warning("No reference cells found, using NULL")
    ref_cells <- NULL
  } else {
    print(paste("Using", length(ref_cells), "reference cells"))
  }
}

# Set the genome for infercna
useGenome(input_genome)
retrieveGenome()

# Run infercna
cna_result <- infercna(
  m = data_matrix,
  refCells = ref_cells,
  n = 5000,
  noise = 0.1,
  center.method = "median",
  isLog = FALSE,
  verbose = TRUE
)

# Save the CNA matrix (gene x cell with log2 ratios)
write.table(
  cna_result,
  file = output_file,
  sep = "\t",
  quote = FALSE,
  col.names = TRUE,
  row.names = TRUE
)

print(paste("infercna output saved to:", output_file))

# ------------------------------------------------------------------------------
print("SessionInfo:")
# ------------------------------------------------------------------------------
sessionInfo()
