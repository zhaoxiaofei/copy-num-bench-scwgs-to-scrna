#!/bin/bash
set -euo pipefail

# ===== 用户配置区域 =====
REF_HG19="/stor/zxf/cnv/refs/hg19.fa"
REF_HG38="/stor/zxf/cnv/refs/grch38/GRCh38_masked_v2_decoy_gene.fasta"
CHAIN_FILE="../data/annotations/hg19ToHg38.over.chain.gz"
FASTQ_R1="/stor/zxf/cnv/real_tumor_data/1from0.datdir/SRR30713454_1.fastq.gz"
FASTQ_R2="/stor/zxf/cnv/real_tumor_data/1from0.datdir/SRR30713454_2.fastq.gz"
THREADS=8
OUTDIR="./liftover_test"
mkdir -p "$OUTDIR"

# 选择坐标转换工具：crossmap 或 liftOver
CONVERT_TOOL="liftOver"

# ===== Custom code =====
USE_SEQ_LEN=false
LIFTED_BAM="$OUTDIR/hg19_lifted_sorted.bam"

# ===== 检查依赖 =====
command -v bwa      >/dev/null 2>&1 || { echo "错误: bwa 未安装"; exit 1; }
command -v samtools >/dev/null 2>&1 || { echo "错误: samtools 未安装"; exit 1; }
command -v bedtools >/dev/null 2>&1 || { echo "错误: bedtools 未安装"; exit 1; }

if [ "$CONVERT_TOOL" == "crossmap" ]; then
    command -v CrossMap.py >/dev/null 2>&1 \
        || { echo "错误: CrossMap 未安装 (可用 pip install CrossMap)"; exit 1; }
elif [ "$CONVERT_TOOL" == "liftOver" ]; then
    command -v liftOver >/dev/null 2>&1 \
        || { echo "错误: liftOver 未安装"; exit 1; }
    [ -f "$CHAIN_FILE" ] \
        || { echo "错误: 链文件 $CHAIN_FILE 不存在"; exit 1; }
else
    echo "错误: CONVERT_TOOL 必须为 crossmap 或 liftOver"
    exit 1
fi

if false; then
# ===== 步骤 1: 比对到 hg19 =====
echo "[1/5] 比对到 hg19 ..."
bwa mem -t "$THREADS" "$REF_HG19" "$FASTQ_R1" "$FASTQ_R2" \
    | samtools view -F 4 -bS - \
    | samtools sort -@ "$THREADS" -o "$OUTDIR/hg19_sorted.bam"
samtools index "$OUTDIR/hg19_sorted.bam"

# ===== 步骤 2: 比对到 hg38 =====
echo "[2/5] 比对到 hg38 ..."
bwa mem -t "$THREADS" "$REF_HG38" "$FASTQ_R1" "$FASTQ_R2" \
    | samtools view -F 4 -bS - \
    | samtools sort -@ "$THREADS" -o "$OUTDIR/hg38_sorted.bam"
samtools index "$OUTDIR/hg38_sorted.bam"

# ===== 步骤 3: 将 hg19 BAM liftOver 到 hg38 =====
echo "[3/5] 将 hg19 比对结果 liftOver 到 hg38 ..."

if [ "$CONVERT_TOOL" == "crossmap" ]; then
    CrossMap.py bam "$CHAIN_FILE" "$OUTDIR/hg19_sorted.bam" \
        "$REF_HG38" "$OUTDIR/hg19_lifted.bam"
    samtools sort -@ "$THREADS" -o "$OUTDIR/hg19_lifted_sorted.bam" "$OUTDIR/hg19_lifted.bam"
    samtools index "$OUTDIR/hg19_lifted_sorted.bam"
    rm "$OUTDIR/hg19_lifted.bam"
    LIFTED_BAM="$OUTDIR/hg19_lifted_sorted.bam"
    USE_SEQ_LEN=true

elif [ "$CONVERT_TOOL" == "liftOver" ]; then
    bedtools bamtobed -i "$OUTDIR/hg19_sorted.bam" > "$OUTDIR/hg19.bed"

    liftOver -bedPlus=4 \
        "$OUTDIR/hg19.bed" \
        "$CHAIN_FILE" \
        "$OUTDIR/hg19_lifted.bed" \
        "$OUTDIR/unmapped.bed"

    echo "  liftOver 未映射 read 数: $(wc -l < "$OUTDIR/unmapped.bed")"

    FAI="${REF_HG38}.fai"
    [ -f "$FAI" ] || { echo "错误: 找不到 ${FAI}，请先运行 samtools faidx"; exit 1; }

    bedtools bedtobam -g "$FAI" -i "$OUTDIR/hg19_lifted.bed" \
        > "$OUTDIR/hg19_lifted_unsorted.bam"
    samtools sort -@ "$THREADS" -o "$OUTDIR/hg19_lifted_sorted.bam" \
        "$OUTDIR/hg19_lifted_unsorted.bam"
    samtools index "$OUTDIR/hg19_lifted_sorted.bam"

    rm "$OUTDIR/hg19_lifted_unsorted.bam" \
       "$OUTDIR/hg19.bed" \
       "$OUTDIR/hg19_lifted.bed"
    LIFTED_BAM="$OUTDIR/hg19_lifted_sorted.bam"
    USE_SEQ_LEN=false
fi

fi

# ===== 步骤 4: 提取每条 read 的比对坐标 =====
echo "[4/5] 提取 read 坐标 ..."

# 不再需要外部 sort，Python 内部处理排序和匹配

if [ "$USE_SEQ_LEN" == "true" ]; then
    samtools view -F 4 "$OUTDIR/hg38_sorted.bam" \
        | awk -F'\t' '{print $3"\t"$4"\t"($4+length($10)-1)"\t"$1}' \
        > "$OUTDIR/hg38_coords.txt"

    samtools view -F 4 "$LIFTED_BAM" \
        | awk -F'\t' '{print $3"\t"$4"\t"($4+length($10)-1)"\t"$1}' \
        > "$OUTDIR/lifted_coords.txt"
else
    samtools view -F 4 "$OUTDIR/hg38_sorted.bam" \
        | awk -F'\t' '{print $3"\t"$4"\t.\t"$1}' \
        > "$OUTDIR/hg38_coords.txt"

    samtools view -F 4 "$LIFTED_BAM" \
        | awk -F'\t' '{print $3"\t"$4"\t.\t"$1}' \
        > "$OUTDIR/lifted_coords.txt"
fi

# ===== 步骤 5: 用 Python 按 read name 合并并比较坐标 =====
echo "[5/5] 比较坐标差异 ..."

python3 - "$OUTDIR" "$USE_SEQ_LEN" << 'PYEOF'
"""
Compare lifted-over coordinates vs directly-aligned hg38 coordinates.

Key fix: bedtools bamtobed appends /1 or /2 to paired-end read names,
while samtools view from BWA BAM does not. This script strips those
suffixes before matching, then aligns mates positionally by sorting
each read's entries by start coordinate.
"""
import sys
import os
from collections import defaultdict

def read_coords(filepath):
    """Read coordinate file into dict: read_name -> list of (chr, start, end)

    Strips /1 /2 suffixes from read names (added by bedtools bamtobed
    for paired-end reads) so they can be matched against BAM QNAME fields
    which lack these suffixes.
    """
    data = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            chrom, start_str, end, name = parts[0], parts[1], parts[2], parts[3]
            # Strip /1 or /2 suffix added by bedtools bamtobed
            if name.endswith('/1') or name.endswith('/2'):
                name = name[:-2]
            data[name].append((chrom, int(start_str), end))
    return data

def main():
    outdir = sys.argv[1]
    use_seq_len = (sys.argv[2].lower() == 'true')

    hg38_file = os.path.join(outdir, 'hg38_coords.txt')
    lifted_file = os.path.join(outdir, 'lifted_coords.txt')
    joined_file = os.path.join(outdir, 'joined_coords.txt')
    report_file = os.path.join(outdir, 'report.txt')

    print("  读取 hg38 坐标 ...")
    hg38_data = read_coords(hg38_file)
    hg38_total = sum(len(v) for v in hg38_data.values())
    print(f"    hg38: {len(hg38_data)} 唯一 read 名, {hg38_total} 条记录")

    print("  读取 lifted 坐标 ...")
    lifted_data = read_coords(lifted_file)
    lifted_total = sum(len(v) for v in lifted_data.values())
    print(f"    lifted: {len(lifted_data)} 唯一 read 名, {lifted_total} 条记录")

    common_names = set(hg38_data.keys()) & set(lifted_data.keys())
    hg38_only = set(hg38_data.keys()) - set(lifted_data.keys())
    lifted_only = set(lifted_data.keys()) - set(hg38_data.keys())
    print(f"    共有 read 名: {len(common_names)}")
    print(f"    仅 hg38: {len(hg38_only)}, 仅 lifted: {len(lifted_only)}")

    # Join by read name, match mates positionally
    total_joined = 0
    chr_mismatch = 0
    exact_start = 0
    exact_both = 0
    diff_sum = 0
    max_diff = 0
    hist = defaultdict(int)

    with open(joined_file, 'w') as jf:
        for read_name in sorted(common_names):
            # Sort entries by start position for positional mate matching
            hg38_entries = sorted(hg38_data[read_name], key=lambda x: x[1])
            lifted_entries = sorted(lifted_data[read_name], key=lambda x: x[1])

            pair_count = min(len(hg38_entries), len(lifted_entries))

            for i in range(pair_count):
                l_chr, l_start, l_end = lifted_entries[i]
                h_chr, h_start, h_end = hg38_entries[i]

                jf.write(f"{l_chr}\t{l_start}\t{l_end}\t{h_chr}\t{h_start}\t{h_end}\n")
                total_joined += 1

                if l_chr != h_chr:
                    chr_mismatch += 1
                    continue

                diff = abs(l_start - h_start)
                diff_sum += diff
                if diff > max_diff:
                    max_diff = diff

                if diff == 0:
                    exact_start += 1
                    if use_seq_len and l_end != '.' and h_end != '.' and str(l_end) == str(h_end):
                        exact_both += 1

                bin_val = (diff // 100) * 100
                if bin_val > 10000:
                    bin_val = 10000
                hist[bin_val] += 1

    print(f"  成功合并 read 对数: {total_joined}")

    # Generate report
    same_chr = total_joined - chr_mismatch
    lines = []
    lines.append("=== 坐标一致性统计报告 ===")
    lines.append(f"合并 read 总数:              {total_joined}")
    if total_joined > 0:
        lines.append(f"染色体不一致 (转换失败等):   {chr_mismatch} "
                     f"({chr_mismatch/total_joined*100:.2f}%)")
    else:
        lines.append(f"染色体不一致 (转换失败等):   {chr_mismatch} (0.00%)")
    lines.append(f"同染色体 read 对数:          {same_chr}")
    lines.append("")

    if same_chr > 0:
        lines.append(f"  起始坐标完全一致:          {exact_start} "
                     f"({exact_start/same_chr*100:.2f}%)")
        if use_seq_len:
            lines.append(f"  起始+终止均一致:           {exact_both} "
                         f"({exact_both/same_chr*100:.2f}%)")
        lines.append(f"  平均起始偏移 (bp):         {diff_sum/same_chr:.2f}")
        lines.append(f"  最大起始偏移 (bp):         {max_diff}")
        lines.append("")
        lines.append("  起始偏移分布 (bin = 100 bp):")
        for b in sorted(hist.keys()):
            label = ">=10000" if b == 10000 else f"{b}-{b+99}"
            lines.append(f"    {label:<15s} {hist[b]}")

    report_text = '\n'.join(lines) + '\n'

    with open(report_file, 'w') as rf:
        rf.write(report_text)

    print()
    print(report_text)
    print(f"详细报告已保存至: {report_file}")
    print(f"合并坐标文件:     {joined_file}")

if __name__ == '__main__':
    main()
PYEOF
