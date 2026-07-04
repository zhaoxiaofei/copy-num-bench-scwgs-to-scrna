# ------------------------------------------------------------------------------
# Run copykat (both modes possible, specifying reference cells or not)
# ------------------------------------------------------------------------------

log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

# ------------------------------------------------------------------------------
print("Load libraries")
# ------------------------------------------------------------------------------

library(copykat)
library(data.table)

# ------------------------------------------------------------------------------
print("Get input parameters from snakemake")
# ------------------------------------------------------------------------------

input_file<-snakemake@input$matrix

#Potentially getting reference annotations (not required)
input_annotations<-snakemake@input$annot
input_ref_groups<-snakemake@input$ref_groups

#Set genome version (human or mouse)
input_genome<-snakemake@params$genome
if(is.null(input_genome)){
  input_genome<-"hg20"
} 
print(paste("Running CopyKat with the following genome:",input_genome))

output_file<-snakemake@output$cnv_file
  
# ------------------------------------------------------------------------------
print("Execute the copyKat in the chosen settings.")
# ------------------------------------------------------------------------------

#Load dataset matrix (annotation file not required in this modus)
data_matrix<-fread(input_file)
#Format into a matrix
gene_names<-data_matrix$V1
data_matrix$V1<-NULL
data_matrix<-as.matrix(data_matrix)
rownames(data_matrix)<-gene_names

#Define reference cells in case they are specified
if(is.null(input_annotations)){
  print("No reference cells defined, copyKat will identify cancer cells automatically")
  
  ref_cells <- ""
  
} else {
  print("Reference cells for copyKat defined")
  
  #Extract reference cells
  annotation<-fread(input_annotations, header=FALSE)
  
  #Read reference groups (saved in one tsv file)
  ref_groups<-read.table(input_ref_groups,header=TRUE)
  
  ref_cells <- annotation$V1[annotation$V2 %in% ref_groups$ref_groups]
  
}

#Create the output directory and define it as a working directory
output_dir<-dirname(output_file)
dir.create(output_dir,recursive=TRUE) # gives a warning if the directory exists already
setwd(output_dir)

#Extract name of the dataset
#dataset_name<-gsub("output_","",unlist(strsplit(output_dir,split="/"))[2])
dataset_name <- snakemake@params$dataset

#Run copyKat
copykat.test <- tryCatch({
                copykat(rawmat=data_matrix, #2d matrix with gene expression counts
                        id.type="S", #gene id type (symbol or ensemble)
                        cell.line="yes", #if data is from pure cell line
                        ngene.chr=5,
                        LOW.DR = 0.05,
                        UP.DR = 0.1,
                        win.size = 25, #minimal window size for segmentation
                        norm.cell.names = ref_cells, #not specifying the reference cells
                        sam.name=dataset_name, #sample name used for output files
                        distance="euclidean",
                        output.seg="FALSE",
                        plot.genes="TRUE",
                        genome = input_genome,
                        n.cores=32)
}, error = function(e) {
  message("MCMC segmentation failed with error: ", e$message)
  message("Falling back to custom MCMC...")

  # https://www.doubao.com/chat/38432855834525186
  # do not throw out an error if only one cluster is found

  original_CNA_MCMC <- copykat:::CNA.MCMC
  patched_CNA_MCMC <- function(clu, fttmat, bins, cut.cor, n.cores) {

      valid_clu <- clu[!is.na(clu)]

      if (length(valid_clu) == 0) {
        message("WARNING: clu is empty/NA. Assuming all cells belong to a single clone.")
        clu <- rep(1, ncol(fttmat))
        names(clu) <- colnames(fttmat)
      } else if (length(unique(valid_clu)) == 1) {
        message("WARNING: only 1 cluster detected. Proceeding with single-clone segmentation.")
        if (length(clu) != ncol(fttmat)) {
          clu <- rep(unique(valid_clu), ncol(fttmat))
          names(clu) <- colnames(fttmat)
        }
      }

      original_CNA_MCMC(clu = clu, fttmat = fttmat, bins = bins, 
                        cut.cor = cut.cor, n.cores = n.cores)
  }

  assignInNamespace("CNA.MCMC", patched_CNA_MCMC, ns = "copykat")

  fallback_result <- tryCatch({
                copykat(rawmat = data_matrix, #2d matrix with gene expression counts
                        id.type = "S", #gene id type (symbol or ensemble)
                        cell.line="yes", #if data is from pure cell line
                        ngene.chr = 5,
                        LOW.DR = 0.05,
                        UP.DR = 0.1,
                        win.size = 25, #minimal window size for segmentation
                        norm.cell.names = ref_cells, #not specifying the reference cells
                        sam.name = dataset_name, #sample name used for output files
                        distance = "euclidean",
                        output.seg = "FALSE",
                        plot.genes = "TRUE",
                        genome = input_genome,
                        n.cores = 32)
  }, error = function(e2) {
    message("Custom MCMC fall-back also failed: ", e2$message)
    stop("Both methods failed.")
  })
  return(fallback_result)
}
)

# ------------------------------------------------------------------------------
print("SessionInfo:")
# ------------------------------------------------------------------------------
sessionInfo()
