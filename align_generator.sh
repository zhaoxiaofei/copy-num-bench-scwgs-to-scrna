#!/bin/bash

### Revised at https://sorryios.ai/chat/8d42e5b8-fd4a-41ec-afe2-c21173df5f29

# This script takes as input:
#  input_A --infile: a file-path with two or three tab-separated columns, denoting the filepath-prefix
#      of BAM file (first column) as output and one or two FASTQ files (last two columns) as input
#  input_B --hg19: a file-path denoting HG19 (with chrom name beginning with chr) FASTA refrence genome
#  input_C --grch38: a file-path denoting GRCH38 (with chrom name beginning with chr) FASTA reference genome
#  input_D --type: either the character D or R, denoting DNA or RNA as input in input_A
# Then, for each row in input_A, this script generates a runner script to the stdout.
# The runner script uses BWA-MEM (DNA) or STAR (RNA) to align the input FASTQ file(s) to each reference
#     genome (HG19 and GRCh38) and uses samtools sort/index to generate the corresponding
#     coordinate-sorted and indexed output BAM file.
#
# Additional optional arguments (defaults from the original script are used when omitted):
#  --bwa-threads       : threads for BWA-MEM            (default: 4)
#  --star-threads      : threads for STAR               (default: 4)
#  --sort-threads      : threads for samtools sort       (default: 4)
#  --star-hg19-dir     : STAR genome index dir for hg19
#  --star-hg19-gtf     : GTF annotation for hg19
#  --star-grch38-dir   : STAR genome index dir for GRCh38
#  --star-grch38-gtf   : GTF annotation for GRCh38
#  --sjdb-overhang     : STAR sjdbOverhang               (default: 149)
#
# Usage:
#   bash align_generator.sh --infile samples.tsv --type D > run_align.sh
#   bash align_generator.sh --infile samples.tsv --type R > run_align.sh
#   bash align_generator.sh --infile samples.tsv --hg19 /path/hg19.fa --grch38 /path/grch38.fa --type R > run_align.sh

set -euo pipefail

# -------------------------- DEFAULT VALUES (from original script) --------------------------
HG19="/stor/zxf/cnv/refs/hg19.fa"
GRCH38="/stor/zxf/cnv/refs/grch38/GRCh38_masked_v2_decoy_gene.fasta"
SEQ_TYPE=""
INFILE=""

# https://www.doubao.com/thread/wf56d5e8e5e51e0e9
# BWA_MEM_K=$((10*1000*1000) # https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html
BWA_MEM_K=$((100*1000*1000)) # PMC6168605, https://github.com/CCDG/Pipeline-Standardization/blob/master/PipelineStandard.md
STAR_DETERMINISTIC_ARGS="--runRNGseed 0 --outSAMorder Paired"

BWA_THREADS=4
STAR_THREADS=4
SORT_THREADS=4
SJDB_OVERHANG=149

STAR_HG19_DIR="/nfs/public/zxf/neoguider/database/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa.star.idx"
STAR_HG19_GTF="/nfs/public/zxf/neoguider/database/GRCh37_gencode_v19_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf"
STAR_GRCH38_DIR="/stor/zxf/cnv/cnvguider-sc-rna/db/GRCh38_gencode_v44_CTAT_lib_Oct292023.plug-n-play/ctat_genome_lib_build_dir/ref_genome.fa.star.idx"
STAR_GRCH38_GTF="/stor/zxf/cnv/cnvguider-sc-rna/db/GRCh38_gencode_v44_CTAT_lib_Oct292023.plug-n-play/ctat_genome_lib_build_dir/ref_annot.gtf"

# -------------------------- ARGUMENT PARSING --------------------------
usage() {
    cat >&2 <<EOF
Usage: $0 --infile <sample_table> --type <D|R> [options]

Required:
  --infile          Tab-separated file. Columns: BAM_prefix  FASTQ_R1  [FASTQ_R2]
  --type            D (DNA, uses BWA-MEM) or R (RNA, uses STAR)

Optional (defaults from original script used when omitted):
  --hg19            HG19 FASTA reference        [default: $HG19]
  --grch38          GRCh38 FASTA reference       [default: $GRCH38]
  --bwa-threads     BWA-MEM threads              [default: $BWA_THREADS]
  --star-threads    STAR threads                 [default: $STAR_THREADS]
  --sort-threads    samtools sort threads         [default: $SORT_THREADS]
  --star-hg19-dir   STAR index dir for hg19      [default: $STAR_HG19_DIR]
  --star-hg19-gtf   GTF annotation for hg19      [default: $STAR_HG19_GTF]
  --star-grch38-dir STAR index dir for GRCh38    [default: $STAR_GRCH38_DIR]
  --star-grch38-gtf GTF annotation for GRCh38    [default: $STAR_GRCH38_GTF]
  --sjdb-overhang   STAR sjdbOverhang            [default: $SJDB_OVERHANG]
  -h, --help        Show this help message
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --infile)          INFILE="$2";           shift 2 ;;
        --hg19)            HG19="$2";             shift 2 ;;
        --grch38)          GRCH38="$2";           shift 2 ;;
        --type)            SEQ_TYPE="$2";         shift 2 ;;
        --bwa-threads)     BWA_THREADS="$2";      shift 2 ;;
        --star-threads)    STAR_THREADS="$2";     shift 2 ;;
        --sort-threads)    SORT_THREADS="$2";     shift 2 ;;
        --star-hg19-dir)   STAR_HG19_DIR="$2";    shift 2 ;;
        --star-hg19-gtf)   STAR_HG19_GTF="$2";    shift 2 ;;
        --star-grch38-dir) STAR_GRCH38_DIR="$2";  shift 2 ;;
        --star-grch38-gtf) STAR_GRCH38_GTF="$2";  shift 2 ;;
        --sjdb-overhang)   SJDB_OVERHANG="$2";    shift 2 ;;
        -h|--help)         usage ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage
            ;;
    esac
done

# -------------------------- VALIDATION --------------------------
if [ -z "$INFILE" ]; then
    echo "Error: --infile is required" >&2
    usage
fi
if [ ! -f "$INFILE" ]; then
    echo "Error: Input file '$INFILE' not found" >&2
    exit 1
fi
if [[ "$SEQ_TYPE" != "D" && "$SEQ_TYPE" != "R" ]]; then
    echo "Error: --type must be 'D' (DNA) or 'R' (RNA), got '${SEQ_TYPE:-<empty>}'" >&2
    usage
fi

# -------------------------- GENERATE RUNNER SCRIPT TO STDOUT --------------------------
echo "#!/bin/bash"
echo "set -euo pipefail"
echo ""

# If RNA mode, emit STAR index generation commands for both references
if [ "$SEQ_TYPE" = "R" ]; then
    cat <<EOF
# --- Generate STAR hg19 index if missing ---
if [ ! -d "${STAR_HG19_DIR}" ]; then
    mkdir -p "${STAR_HG19_DIR}"
    STAR --runMode genomeGenerate \\
         --genomeDir "${STAR_HG19_DIR}" \\
         --genomeFastaFiles "${HG19}" \\
         --sjdbGTFfile "${STAR_HG19_GTF}" \\
         --sjdbOverhang ${SJDB_OVERHANG} \\
         --runThreadN ${STAR_THREADS} \\
else
    echo "Skipping STAR hg19 index (exists)"
fi

# --- Generate STAR GRCh38 index if missing ---
if [ ! -d "${STAR_GRCH38_DIR}" ]; then
    mkdir -p "${STAR_GRCH38_DIR}"
    STAR --runMode genomeGenerate \\
         --genomeDir "${STAR_GRCH38_DIR}" \\
         --genomeFastaFiles "${GRCH38}" \\
         --sjdbGTFfile "${STAR_GRCH38_GTF}" \\
         --sjdbOverhang ${SJDB_OVERHANG} \\
         --runThreadN ${STAR_THREADS} \\
else
    echo "Skipping STAR GRCh38 index (exists)"
fi

EOF
fi

# -------------------------- PROCESS EACH ROW --------------------------
while IFS=$'\t' read -r bam_prefix fq1 fq2 || [ -n "$bam_prefix" ]; do
    # Skip empty lines and comments
    [[ -z "$bam_prefix" || "$bam_prefix" == \#* ]] && continue

    # Trim whitespace
    bam_prefix="$(echo "$bam_prefix" | xargs)"
    fq1="$(echo "$fq1" | xargs)"
    fq2="$(echo "${fq2:-}" | xargs)"

    # Define output BAM paths for both references
    hg19_bam="${bam_prefix}_hg19.bam"
    grch38_bam="${bam_prefix}_grch38.bam"

    if [ "$SEQ_TYPE" = "D" ]; then
        # ======================== DNA: BWA-MEM ========================
        if [ -n "$fq2" ]; then
            fq_args="\"${fq1}\" \"${fq2}\""
        else
            fq_args="\"${fq1}\""
        fi

        # --- hg19 ---
        cat <<EOF
# --- DNA alignment to hg19: ${hg19_bam} ---
if [ ! -f "${hg19_bam}.bai" ]; then
    bwa mem -K ${BWA_MEM_K} -t ${BWA_THREADS} "${HG19}" ${fq_args} \\
        | samtools sort -@ ${SORT_THREADS} -o "${hg19_bam}" - \\
    && samtools index "${hg19_bam}"
else
    echo "Skipping ${hg19_bam} (exists)"
fi

EOF

        # --- grch38 ---
        cat <<EOF
# --- DNA alignment to GRCh38: ${grch38_bam} ---
if [ ! -f "${grch38_bam}.bai" ]; then
    bwa mem -K ${BWA_MEM_K} t ${BWA_THREADS} "${GRCH38}" ${fq_args} \\
        | samtools sort -@ ${SORT_THREADS} -o "${grch38_bam}" - \\
    && samtools index "${grch38_bam}"
else
    echo "Skipping ${grch38_bam} (exists)"
fi

EOF

    else
        # ======================== RNA: STAR ========================
        if [ -n "$fq2" ]; then
            read_files_in="\"${fq1}\" \"${fq2}\""
        else
            read_files_in="\"${fq1}\""
        fi

        hg19_star_prefix="${bam_prefix}_hg19_"
        hg19_star_sorted="${hg19_star_prefix}Aligned.sortedByCoord.out.bam"
        grch38_star_prefix="${bam_prefix}_grch38_"
        grch38_star_sorted="${grch38_star_prefix}Aligned.sortedByCoord.out.bam"

        # --- hg19 ---
        cat <<EOF
# --- RNA alignment to hg19: ${hg19_bam} ---
if [ ! -f "${hg19_bam}.bai" ]; then
    STAR --readFilesCommand zcat \\
         --genomeLoad LoadAndKeep \\
         --genomeDir "${STAR_HG19_DIR}" \\
         --readFilesIn ${read_files_in} \\
         --outFileNamePrefix "${hg19_star_prefix}" \\
         --outSAMtype BAM SortedByCoordinate \\
         --quantMode GeneCounts \\
         --runThreadN ${STAR_THREADS} \\
         $STAR_DETERMINISTIC_ARGS \\
    && mv "${hg19_star_sorted}" "${hg19_bam}" \\
    && samtools index "${hg19_bam}"
else
    echo "Skipping ${hg19_bam} (exists)"
fi

EOF

        # --- grch38 ---
        cat <<EOF
# --- RNA alignment to GRCh38: ${grch38_bam} ---
if [ ! -f "${grch38_bam}.bai" ]; then
    STAR --readFilesCommand zcat \\
         --genomeLoad LoadAndKeep \\
         --genomeDir "${STAR_GRCH38_DIR}" \\
         --readFilesIn ${read_files_in} \\
         --outFileNamePrefix "${grch38_star_prefix}" \\
         --outSAMtype BAM SortedByCoordinate \\
         --quantMode GeneCounts \\
         --runThreadN ${STAR_THREADS} \\
         $STAR_DETERMINISTIC_ARGS \\
    && mv "${grch38_star_sorted}" "${grch38_bam}" \\
    && samtools index "${grch38_bam}"
else
    echo "Skipping ${grch38_bam} (exists)"
fi

EOF
    fi
done < "$INFILE"

