#!/usr/bin/env bash

conda=micromamba # please change to mamba or conda if you want
envname="$1"
ginkgo_env="$2"

if [ -z "${envname}" ]; then envname=cnb_scrna1 ; fi
if [ -z "${ginkgo_env}" ]; then ginkgo_env=ginkgo_env1 ; fi

conda_install_params="-c conda-forge -c bioconda"

eval "$(${conda} shell hook --shell bash)"

scalop_prereqs="bioconductor-clusterprofiler bioconductor-homo.sapiens bioconductor-org.hs.eg.db bioconductor-bsgenome.hsapiens.ucsc.hg19"
casper_prereqs="bioconductor-txdb.hsapiens.ucsc.hg19.knowngene bioconductor-txdb.hsapiens.ucsc.hg38.knowngene"
# cellsnp-lite is required by numbat, CTAT libs have been built by STAR version 2.7.4a, subread has featureCounts
${conda} env create -y --name ${envname} sra-tools snakemake bwa samtools bcftools star=2.7.4a r-base \
    r-devtools r-remotes r-stringr r-seurat r-numbat r-scevan bioconductor-infercnv cellsnp-lite \
    bioconductor-rgraphviz bioconductor-gostats bioconductor-fgsea bioconductor-ggtree bioconductor-rtracklayer \
    subread ucsc-liftover \
    pyreadr \
    numpy scipy matplotlib seaborn pandas scikit-learn weightedstats \
    $scalop_prereqs $casper_prereqs

${conda} activate ${envname}

pip install wcorr # weighted correlations

Rscript -e 'library(devtools); install_github("jlaffy/scalop")' # required by infercna
Rscript -e 'library(devtools); install_github("jlaffy/infercna")'
Rscript -e 'library(devtools); install_github("akdess/CaSpER")'
Rscript -e 'library(devtools); install_github("navinlabcode/copykat")'
Rscript -e 'library(devtools); install_github("diazlab/CONICS/CONICSmat")'

git clone https://github.com/akdess/BAFExtract.git && cd BAFExtract && make

# Install ginkgo, the scWGS-based CNV caller for constructing single-cell CNV ground-truth
${conda} create -y -n ${ginkgo_env} bioconductor-ctc bioconductor-dnacopy r-devtools r-inline r-gplots r-scales r-plyr r-ggplot2 r-gridExtra r-fastcluster r-heatmap3 \
  parallel # bwa, bowtie, wgsim are not needed

# Download databases
mkdir -p data/annotations/
wget -c https://hgdownload.cse.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz && mv hg19ToHg38.over.chain.gz data/annotations/
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_27/gencode.v27.annotation.gtf.gz && mv gencode.v27.annotation.gtf.gz data/annotations/
gunzip -f data/annotations/gencode.v27.annotation.gtf.gz
wget -c https://github.com/broadinstitute/inferCNV_examples/raw/master/__gene_position_data/gencode_v19_gene_pos.txt && mv gencode_v19_gene_pos.txt data/annotations/
wget -c https://github.com/Neurosurgery-Brain-Tumor-Center-DiazLab/CONICS/raw/master/chromosome_arm_positions_grch38.txt && mv chromosome_arm_positions_grch38.txt data/annotations/
wget -c https://data.broadinstitute.org/Trinity/CTAT/cnv/hg38_gencode_v27.txt && mv hg38_gencode_v27.txt data/annotations/

pushd data/annotations/
wget -c https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/GRCh38_gencode_v22_CTAT_lib_Mar012021.plug-n-play.tar.gz
tar -xvf GRCh38_gencode_v22_CTAT_lib_Mar012021.plug-n-play.tar.gz
popd

# wget ftp://ftp.ensembl.org/pub/release-105/gtf/homo_sapiens/Homo_sapiens.GRCh38.105.gtf.gz

# run_gene_annot
Rscript -e '
# BiocManager::install("rtracklayer") # Already installed with $conda

library(rtracklayer)

gtf <- import("data/annotations/gencode.v27.annotation.gtf")

genes <- gtf[gtf$type == "gene"]

annotation <- data.frame(
    Gene = genes$gene_name, # not gene_id
    chr = as.character(seqnames(genes)),
    start = start(genes),
    end = end(genes),
    stringsAsFactors = FALSE
)

saveRDS(annotation, "data/annotations/casper_gene_annotation_GRCh38.rds")

output_gene_annot <- ("data/annotations/casper_gene_annotation_GRCh38.tsv")
write.table(annotation,file=output_gene_annot,
            sep="\t",quote=FALSE,row.names = FALSE)

'

### https://sorryios.ai/chat/dd8f4c23-9f1e-450e-af0b-a8f7128a02db
###

### BAFExtract (for CaSpER)  ###

# 1) Build the binary
mkdir -p tools && cd tools
git clone https://github.com/akdess/BAFExtract.git || true
cd BAFExtract
make                     # produces bin/BAFExtract
cd ../..

# 2) Genome list (hg38 chromosome sizes)
mkdir -p data/annotations/genome_input_BAFExtract_hg38
cd data/annotations/genome_input_BAFExtract_hg38

# Option A: prebuilt from the BAFExtract authors
wget -c -O hg38_genome_list "https://www.dropbox.com/s/rq7v67tiou1qwwg/hg38.list?dl=1"
# Option B: regenerate yourself (requires UCSC 'fetchChromSizes' from kent-tools)
# fetchChromSizes hg38 > hg38_genome_list

# 3) Genome pileup directory (per-chromosome preprocessed FASTAs)
wget -c -O hg38.zip "https://www.dropbox.com/s/ysrcfcnk7z8gyit/hg38.zip?dl=1"
unzip -o hg38.zip        # unpacks into ./hg38/  -> this is BAFEXTRACT_GENOME
cd ../../..

### Numbat / Eagle files ###

mkdir -p data/annotations tools && cd tools

# 1) Eagle v2.4.1 (prebuilt static binary, ships the genetic map too)
wget -c https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/Eagle_v2.4.1.tar.gz
tar -xzf Eagle_v2.4.1.tar.gz     # -> Eagle_v2.4.1/eagle  + Eagle_v2.4.1/tables/
cd ..

# 2) Genetic map (comes with Eagle; just symlink/copy into annotations)
cp tools/Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz \
   data/annotations/genetic_map_hg38_withX.txt.gz

cd data/annotations

# 3) SNP VCF (cellsnp SNP list, hg38)
wget -c "https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download" \
     -O genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz

# 4) 1000G reference panel (BCF files) for Eagle phasing
wget -c -d \
  --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36" \
  --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
  --header="Accept-Language: en-US,en;q=0.5" \
  --header="Connection: keep-alive" \
  --header="DNT: 1" \
  --header="Sec-Fetch-Dest: document" \
  --header="Sec-Fetch-Mode: navigate" \
  --header="Sec-Fetch-Site: none" \
  --header="Sec-Fetch-User: ?1" \
  --header="Upgrade-Insecure-Requests: 1" \
  --referer="https://pklab.med.harvard.edu" \
  --content-disposition \
  --continue \
  "https://pklab.med.harvard.edu/teng/data/1000G_hg38.zip"

# This is the old link: wget https://pklab.org/teng/data/1000G_hg38.zip
unzip -o 1000G_hg38.zip        # -> 1000G_hg38/

cd ../..

pushd data/annotations
for ctat in 'GRCh37_gencode_v19_CTAT_lib_Mar012021' 'GRCh38_gencode_v22_CTAT_lib_Mar012021' 'GRCh38_gencode_v44_CTAT_lib_Oct292023'; do
    wget -c https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/${ctat}.plug-n-play.tar.gz && tar -xvf ${ctat}.plug-n-play.tar.gz
done
popd

