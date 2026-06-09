suppressPackageStartupMessages({
  library(Seurat)
  library(infercnv)
  library(dplyr)
  library(ggplot2)
})
set.seed(0)

args <- commandArgs(trailingOnly = TRUE) # e.g., args[1] = "cell_by_gene_GeneFull.tsv"
filename <- args[1]
## ---- 1. Read + transpose + cluster ----
raw    <- read.table(filename, header = TRUE, row.names = 1,
                     sep = "\t", check.names = FALSE)
counts <- (as.matrix(raw)); storage.mode(counts) <- "integer"  # gene x cell

seu <- CreateSeuratObject(counts, min.cells = 3, min.features = 200)
seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, nfeatures = 2000, verbose = FALSE)
seu <- ScaleData(seu, verbose = FALSE)
seu <- RunPCA(seu, npcs = 30, verbose = FALSE)
seu <- FindNeighbors(seu, dims = 1:30, verbose = FALSE)
seu <- FindClusters(seu, resolution = 0.8, verbose = FALSE)

## ============================================================
## 2. Define "confident reference" lineage markers (immune + stromal)
##    ↓↓↓ MUST adjust for your species/tissue; this is a human set ↓↓↓
## ============================================================
normal_markers <- list(
  Tcell       = c("CD3D","CD3E","CD2","CD8A","IL7R"),
  Bcell       = c("CD79A","CD79B","MS4A1","CD19"),
  NK          = c("NKG7","GNLY","KLRD1"),
  Myeloid     = c("CD68","LYZ","CD14","FCGR3A","AIF1"),
  Fibroblast  = c("COL1A1","COL1A2","DCN","PDGFRB","LUM"),
  Endothelial = c("PECAM1","VWF","CLDN5","CDH5")
)
panimmune <- c("PTPRC")  # CD45, pan-immune, inspect separately

print('Seurat row names are as follows.')
head(rownames(seu), 20)

## Keep only genes that actually exist in the matrix, to avoid AddModuleScore errors
normal_markers <- lapply(normal_markers, intersect, rownames(seu))
normal_markers <- normal_markers[lengths(normal_markers) > 0]

## ============================================================
## 3. Score each cell for reference-lineage identity
## ============================================================
seu <- AddModuleScore(seu, features = normal_markers,
                      name = "normalScore_", ctrl = 50)
score_cols <- grep("^normalScore_", colnames(seu@meta.data), value = TRUE)

meta <- seu@meta.data
meta$cell  <- rownames(meta)
meta$group <- paste0("cluster_", as.character(Idents(seu)))
## Per cell, take the max across lineage scores = "how much it looks like some reference cell type"
meta$normal_max <- apply(meta[, score_cols, drop = FALSE], 1, max)
## CD45 expression (normalized) as hard corroborating evidence for immune cells
if ("PTPRC" %in% rownames(seu)) {
  meta$CD45 <- FetchData(seu, "PTPRC")[, 1]
} else meta$CD45 <- NA_real_

## ============================================================
## 4. Decide reference at the CLUSTER level (never call it per-cell)
## ============================================================
cluster_stat <- meta %>%
  group_by(group) %>%
  summarise(mean_normal = mean(normal_max),
            mean_CD45   = mean(CD45, na.rm = TRUE),
            n           = n(),
            .groups = "drop") %>%
  arrange(desc(mean_normal))
print(cluster_stat)

## Rule: clusters with clearly positive reference-lineage score (>0 means above background) become reference
NORMAL_SCORE_CUT <- 0.5       # AddModuleScore already subtracts background, so >0 = enriched
ref_clusters <- cluster_stat$group[cluster_stat$mean_normal > NORMAL_SCORE_CUT]

## Fallback: if no cluster passes, take the highest-scoring one and warn for manual review
if (length(ref_clusters) == 0) {
  ref_clusters <- cluster_stat$group[1]
  warning("No cluster is clearly enriched for reference markers; fell back to the top-scoring cluster. Inspect the heatmap manually!")
  saveRDS(cluster_stat, file=paste0(filename, ".cluster_stat_zero_ref_cells.rds"))
  has_ref = FALSE
} else {
  saveRDS(cluster_stat, file=paste0(filename, ".cluster_stat_nonzero_ref_cells.rds"))
  has_ref = TRUE
}
# writeLines(paste0("n_ref_clusters=", length(ref_clusters)), con=paste0(filename, ".n_clusters.txt"))

message("Marker-based reference groups: ", paste(ref_clusters, collapse = ", "))

## ============================================================
## 5. UMAP visualization (add this after ref_clusters is defined)
## ============================================================
set.seed(0)  # for reproducible UMAP

# Compute UMAP using the same PCA dimensions as clustering
seu <- RunUMAP(seu, dims = 1:30, verbose = FALSE)

# --- Plot 1: Clusters (from Seurat) ---
pdf(paste0(filename, ".umap_clusters.pdf"), width = 8, height = 6)
print(DimPlot(seu, reduction = "umap", label = TRUE, repel = TRUE) +
        ggtitle("UMAP - Seurat Clusters"))
dev.off()

# --- Plot 2: Tumor vs Normal (based on your marker-based reference) ---
# Add the label to the Seurat object for easy plotting
meta$label <- ifelse(meta$group %in% ref_clusters, "reference", "tumor")
seu$label <- meta$label

pdf(paste0(filename, ".umap_label.pdf"), width = 8, height = 6)
print(DimPlot(seu, reduction = "umap", group.by = "label",
              cols = c("tumor" = "red", "reference" = "blue")) +
        ggtitle("UMAP - Tumor vs Normal (Marker-based)"))
dev.off()

# Optional: Save UMAP coordinates to a file for external use
umap_coords <- Embeddings(seu, "umap")
write.table(umap_coords, file = paste0(filename, ".umap_coords.txt"),
            sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

## ------------------- REVISED OUTPUT SECTION -------------------
# Create binary labels: reference = reference clusters, tumor = everything else
meta$label <- ifelse(meta$group %in% ref_clusters, "reference", "tumor")
meta$real_label <- ifelse(((meta$group %in% ref_clusters) & has_ref), "reference", "tumor")

# 1. Write annotation file (cell + tumor/reference label)
annot <- meta[, c("cell", "group")]
write.table(annot,
            file = paste0(filename, ".group_annotation.txt"),   # or .txt as you prefer
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)

annot <- meta[, c("cell", "label")]
write.table(annot,
            file = paste0(filename, ".annotation.txt"),   # or .txt as you prefer
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)

annot <- meta[, c("cell", "real_label")]
write.table(annot,
            file = paste0(filename, ".annotation.txt.maybe_zero_refs"),
            sep = "\t",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE)

# 2. Write reference cell list (only cells from reference clusters)
writeLines(meta$cell[meta$group %in% ref_clusters],
           con = paste0(filename, ".reference_cells.txt"))

writeLines(meta$cell[!(meta$group %in% ref_clusters)],
           con = paste0(filename, ".tumor_cells.txt"))

# 3. Save ref_clusters for downstream use (optional)
saveRDS(ref_clusters, file = paste0(filename, ".ref_clusters.rds"))

