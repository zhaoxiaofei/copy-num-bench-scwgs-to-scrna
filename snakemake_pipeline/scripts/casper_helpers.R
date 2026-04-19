# Revised from https://www.doubao.com/chat/38421549834942978 
# Please install these packages via 
# if (!require("BiocManager", quietly = TRUE)) { install.packages("BiocManager") }
# BiocManager::install("PackageName") 
# # where PackageName refers to TxDb.Hsapiens.UCSC.hg19.knownGene, ..., GenomicRanges

library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(org.Hs.eg.db)
library(GenomicRanges)
library(dplyr)

generateAnnotation_offline_v01 <- function(
    genes,
    centromere,
    id_type = "hgnc_symbol",
    genome = "hg38",
    ishg19 = FALSE
) {
    if (ishg19) {
        genome <- "hg19"
    }
    # Validate genome version
    if (!genome %in% c("hg38", "hg19")) {
        stop("Only 'hg38' and 'hg19' are supported")
    }

    # Load appropriate TxDb database
    if (genome == "hg38") {
        txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
    } else {
        txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
    }

    # Get gene coordinates from local TxDb
    gene_gr <- genes(txdb)
    gene_df <- as.data.frame(gene_gr) %>%
        dplyr::select(entrez_id = gene_id, Chr = seqnames, start, end) %>%
        mutate(Chr = gsub("chr", "", Chr))

    # Map input genes to Entrez IDs
    if (id_type == "hgnc_symbol") {
        gene_map <- select(
            org.Hs.eg.db,
            keys = genes,
            keytype = "SYMBOL",
            columns = c("SYMBOL", "ENTREZID")
        )
        colnames(gene_map) <- c("Gene", "entrez_id")
        gene_map$GeneSymbol <- gene_map$Gene
    } else if (id_type == "ensembl_gene_id") {
        gene_map <- select(
            org.Hs.eg.db,
            keys = genes,
            keytype = "ENSEMBL",
            columns = c("ENSEMBL", "ENTREZID", "SYMBOL")
        )
        colnames(gene_map) <- c("Gene", "entrez_id", "GeneSymbol")
    } else {
        stop("Only 'hgnc_symbol' and 'ensembl_gene_id' are supported")
    }

    # Merge coordinates with gene mappings
    annotation <- gene_df %>%
        inner_join(gene_map, by = "entrez_id") %>%
        filter(Chr %in% c(as.character(1:22), "X")) %>%
        arrange(Chr, start)

    # Calculate gene midpoint (Position) - FIXED: calculated before centromere check
    annotation$Position <- (annotation$start + annotation$end) / 2

    # Approximate cytoband (p/q arm) using centromere positions
    annotation$cytoband <- NA
    for (k in 1:nrow(centromere)) {
        chr_target <- gsub("chr", "", centromere$V1[k])
        centromere_start <- centromere$V2[k]
        centromere_end <- centromere$V3[k]

        chr_idx <- which(annotation$Chr == chr_target)
        if (length(chr_idx) == 0) next

        # Assign p arm (before centromere) or q arm (after centromere)
        annotation$cytoband[chr_idx][annotation$Position[chr_idx] < centromere_start] <- paste0(chr_target, "p")
        annotation$cytoband[chr_idx][annotation$Position[chr_idx] > centromere_end] <- paste0(chr_target, "q")
    }

    # Remove genes without cytoband assignment
    annotation <- annotation[!is.na(annotation$cytoband), ]

    # Mark genes in centromere regions
    annotation$isCentromer <- "no"
    for (k in 1:nrow(centromere)) {
        chr_target <- gsub("chr", "", centromere$V1[k])
        centromere_start <- centromere$V2[k]
        centromere_end <- centromere$V3[k]

        hit_idx <- which(
            annotation$Chr == chr_target &
            annotation$Position >= centromere_start &
            annotation$Position <= centromere_end
        )
        annotation$isCentromer[hit_idx] <- "yes"
    }

    # Reorder columns to match original CaSpER format
    annotation <- annotation %>%
        dplyr::select(Gene, GeneSymbol, Chr, start, end, Position, cytoband, isCentromer)

    # Add new_positions (index within cytoband)
    annotation <- annotation %>%
        group_by(cytoband) %>%
        mutate(new_positions = row_number()) %>%
        ungroup()

    # Final formatting
    annotation <- as.data.frame(annotation)
    rownames(annotation) <- NULL

    return(annotation)
}


library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(org.Hs.eg.db)
library(GenomicRanges)
library(dplyr)
library(AnnotationDbi)  # Explicitly load to ensure select() is available

generateAnnotation_offline <- function(
    genes,
    centromere,
    id_type = "hgnc_symbol",
    genome = "hg38",
    ishg19 = FALSE
) {
    if (ishg19) { genome <- "hg19"; }
    # Validate genome version
    if (!genome %in% c("hg38", "hg19")) {
        stop("Only 'hg38' and 'hg19' are supported")
    }

    # Load appropriate TxDb database
    if (genome == "hg38") {
        txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
    } else {
        txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
    }

    # Get gene coordinates from local TxDb (suppress harmless strand warning)
    gene_gr <- suppressMessages(genes(txdb))  # FIX 1: Suppress warning
    gene_df <- as.data.frame(gene_gr) %>%
        dplyr::select(entrez_id = gene_id, Chr = seqnames, start, end) %>%  # Explicit dplyr::select
        mutate(Chr = gsub("chr", "", Chr))

    # Map input genes to Entrez IDs
    if (id_type == "hgnc_symbol") {
        # FIX 2: Use AnnotationDbi::select() explicitly to avoid dplyr conflict
        gene_map <- AnnotationDbi::select(
            org.Hs.eg.db,
            keys = genes,
            keytype = "SYMBOL",
            columns = c("SYMBOL", "ENTREZID")
        )
        colnames(gene_map) <- c("Gene", "entrez_id")
        gene_map$GeneSymbol <- gene_map$Gene
    } else if (id_type == "ensembl_gene_id") {
        # FIX 2: Use AnnotationDbi::select() explicitly
        gene_map <- AnnotationDbi::select(
            org.Hs.eg.db,
            keys = genes,
            keytype = "ENSEMBL",
            columns = c("ENSEMBL", "ENTREZID", "SYMBOL")
        )
        colnames(gene_map) <- c("Gene", "entrez_id", "GeneSymbol")
    } else {
        stop("Only 'hgnc_symbol' and 'ensembl_gene_id' are supported")
    }

    # Merge coordinates with gene mappings
    annotation <- gene_df %>%
        inner_join(gene_map, by = "entrez_id") %>%
        filter(Chr %in% c(as.character(1:22), "X")) %>%
        arrange(Chr, start)

    # Calculate gene midpoint (Position)
    annotation$Position <- (annotation$start + annotation$end) / 2

    # Approximate cytoband (p/q arm) using centromere positions
    annotation$cytoband <- NA
    for (k in 1:nrow(centromere)) {
        chr_target <- gsub("chr", "", centromere$V1[k])
        centromere_start <- centromere$V2[k]
        centromere_end <- centromere$V3[k]

        chr_idx <- which(annotation$Chr == chr_target)
        if (length(chr_idx) == 0) next

        # Assign p arm (before centromere) or q arm (after centromere)
        annotation$cytoband[chr_idx][annotation$Position[chr_idx] < centromere_start] <- paste0(chr_target, "p")
        annotation$cytoband[chr_idx][annotation$Position[chr_idx] > centromere_end] <- paste0(chr_target, "q")
    }

    # Remove genes without cytoband assignment
    annotation <- annotation[!is.na(annotation$cytoband), ]

    # Mark genes in centromere regions
    annotation$isCentromer <- "no"
    for (k in 1:nrow(centromere)) {
        chr_target <- gsub("chr", "", centromere$V1[k])
        centromere_start <- centromere$V2[k]
        centromere_end <- centromere$V3[k]

        hit_idx <- which(
            annotation$Chr == chr_target &
            annotation$Position >= centromere_start &
            annotation$Position <= centromere_end
        )
        annotation$isCentromer[hit_idx] <- "yes"
    }

    # Reorder columns to match original CaSpER format
    annotation <- annotation %>%
        dplyr::select(Gene, GeneSymbol, Chr, start, end, Position, cytoband, isCentromer)

    # Add new_positions (index within cytoband)
    annotation <- annotation %>%
        group_by(cytoband) %>%
        mutate(new_positions = row_number()) %>%
        ungroup()

    # Final formatting
    annotation <- as.data.frame(annotation)
    rownames(annotation) <- NULL

    return(annotation)
}
