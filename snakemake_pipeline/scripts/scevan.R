# ------------------------------------------------------------------------------
# Run SCEVAN (both modes possible, specifying reference cells or not)
# ------------------------------------------------------------------------------

log <- file(snakemake@log[[1]], open="wt")
sink(log)
sink(log, type="message")

# ------------------------------------------------------------------------------
print("Load libraries")
# ------------------------------------------------------------------------------

library(SCEVAN)
library(data.table)

# ------------------------------------------------------------------------------
print("Get input parameters from snakemake")
# ------------------------------------------------------------------------------

input_file<-snakemake@input$matrix

#Potentially getting reference annotations (not required)
input_annotations<-snakemake@input$annot
input_ref_groups<-snakemake@input$ref_groups

#Boolean option to find subclones
input_clones<-as.logical(snakemake@params$find_clones)

#Set genome version (human or mouse)
input_organism<-snakemake@params$organism
if(is.null(input_organism)){
  input_organism<-"human"
} 
print(paste("Running SCEVAN for the following organism:",input_organism))

output_file<-snakemake@output$cnv_file
output_pred<-snakemake@output$pred_file

subclone1_file <- output_file
clone1_file <- gsub("_subclone1_CN.seg", "_Clonal_CN.seg", subclone1_file)

expected_CNAmtx_file <- gsub("_subclone1_CN.seg", "_CNAmtxSubclones.RData", subclone1_file)
actual_CNAmtx_file <- gsub("_subclone1_CN.seg", "_CNAmtx.RData", subclone1_file)

# ------------------------------------------------------------------------------
print("Execute the SCEVAN in the chosen settings.")
# ------------------------------------------------------------------------------

#Load dataset as prepared for inferCNV
data_matrix<-fread(input_file)
#Format into a matrix
gene_names<-data_matrix$V1
data_matrix$V1<-NULL
data_matrix<-as.matrix(data_matrix)
rownames(data_matrix)<-gene_names

#Define reference cells in case they are specified
if(is.null(input_annotations)){
  print("No reference cells defined, SCEVAN will identify cancer cells automatically")
  
  ref_cells <- NULL
  
} else {
  print("Reference cells for SCEVAN defined")
  
  #Extract reference cells
  annotation<-fread(input_annotations,header=FALSE)
  
  #Read reference groups (saved in one tsv file)
  ref_groups<-read.table(input_ref_groups,header=TRUE)
  
  ref_cells <- annotation$V1[annotation$V2 %in% ref_groups$ref_groups]
  
}

#Issue with . in cell names for SCEVAN subclonal analysis
if(input_clones){
  colnames(data_matrix)<- gsub("\\.","_",colnames(data_matrix))
  ref_cells <- gsub("\\.","_",ref_cells)
}

#Create the SCEVAN output directory and define it as a working directory
#Remove also the last directory from the path as the "output" directory will
#be created automatically by SCEVAN
output_dir<-dirname(dirname(output_file))
dir.create(output_dir,recursive=TRUE) # gives a warning if the directory exists already
old_wd <- getwd()
setwd(output_dir)

#Extract name of the dataset
#dataset_name<-gsub("output_","",unlist(strsplit(output_dir,split="/"))[2])
dataset_name<-snakemake@params$dataset

# https://www.doubao.com/chat/38419084615184642
if (FALSE) {
scevan_output<-SCEVAN::pipelineCNA(data_matrix, 
                                   sample = dataset_name, 
                                   par_cores = 1,
                                   norm_cell=ref_cells,
                                   SUBCLONES = input_clones, 
                                   plotTree = FALSE,
                                   organism = input_organism)
}
scevan_output <- tryCatch({
    message("Try first run with (SUBCLONES = ", input_clones, ")...")
    SCEVAN::pipelineCNA(
      data_matrix, 
      sample = dataset_name,
      par_cores = 1,
      norm_cell = ref_cells,
      SUBCLONES = input_clones,
      plotTree = FALSE,
      organism = input_organism
    )
  }, error = function(e) {
    message("The first run failed, failure message: ", e$message)
    message("Back to safe mode with (SUBCLONES = FALSE), rerunning...")
    SCEVAN::pipelineCNA(
      data_matrix,
      sample = dataset_name,
      par_cores = 1,
      norm_cell = ref_cells,  
      SUBCLONES = FALSE,
      plotTree = FALSE,
      organism = input_organism
    )
})

#Save cell classification as it is not saved automatically in the directory
#Remark: output_pred file will be saved in the SCEVAN output directory
write.table(scevan_output,file=paste0("output/",basename(output_pred)),
            sep="\t",quote=FALSE)

# ------------------------------------------------------------------------------
print("SessionInfo:")
# ------------------------------------------------------------------------------
sessionInfo()

setwd(old_wd)

safe_cp <- function(src_file, dest_file) {
  if (file.exists(dest_file)) {
    print(paste0("The destination file `", dest_file, "` already exists, so skip changing the destination file. "))
  } else if (file.exists(src_file)) {
    copy_success <- file.copy(src_file, dest_file, overwrite=FALSE)
    if (copy_success) {
        message(paste0("Copied successfully: `", src_file, "` -> `", dest_file, "`"))
    } else {
        stop(paste0("FAILED to copy: `", src_file, "` -> `", dest_file, "`"))
    }
  } else {
     stop(paste("Both observed and expected SCEVAN output not found:", src_file, dest_file))
  }
}

safe_cp(clone1_file, subclone1_file)
safe_cp(actual_CNAmtx_file, expected_CNAmtx_file)

