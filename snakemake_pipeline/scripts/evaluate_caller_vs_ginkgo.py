#!/usr/bin/env python3

'''
<REVISION_HISTORY>
- https://sorryios.ai/chat/882715b7-e2f7-4f88-ad1a-a7a2e7f25cbc
- https://sorryios.ai/chat/15fbca34-21b0-467d-805c-b207952bc631
  Initial
- https://sorryios.ai/chat/dd8f4c23-9f1e-450e-af0b-a8f7128a02db
  Added CONICSmat, Numbat, and CaSpER into the benchmark
- https://agent.minimaxi.com/share/389393240474113?chat_type=2
  Added infercna support
</REVISION_HISTORY>
'''

"""
Evaluate a CNV caller's output against Ginkgo ground truth (cell-by-cell).

Metrics:
  - CopyNumber gain precision, recall, F-score
  - CopyNumber loss precision, recall, F-score
  - Multiclass classification accuracy (neutral / gain / loss)
  - Pearson correlation coefficient
  - Spearman correlation coefficient
  - Fraction of human exome covered by CNV calling results

The ground truth file is expected to be a wide TSV like:
    CHR START END <cell1> <cell2> ...

The caller result file is auto-detected by format. This script is designed
primarily for RNA-seq CNV callers that produce a feature-by-cell matrix
(copykat / infercnv / generic matrix), but it also keeps partial support for
segment-like formats.

Usage:
    python evaluate_caller_vs_ginkgo.py \
        --ground-truth SegCopy_grch38.tsv \
        --caller-result scWGS_scRNA_copykat_CNA_raw_results_gene_by_cell.txt \
        --caller-name copykat_predict \
        --gene-pos hg38_gencode_v27.txt \
        --sample-annotation data/input_{dataset}/sample_annotation.txt \
        --output evaluation.tsv

Notes:
- Cell matching is by column order, as requested by the user.
- For copykat-like centered signals, predicted classes are assigned using
  thresholds around 0 by default:
      loss < -0.1, neutral [-0.1, 0.1], gain > 0.1
- For absolute copy-number-like predictions, the script falls back to
  median-relative thresholds.

Modified to support SCEVAN:
    - Reads SCEVAN's '*_CNAmtx.RData' file (gene-by-cell matrix).
    - Requires 'pyreadr' and 'pandas'.

Added features:
    - --sample-annotation: 2-column TSV (no header): original_sample_name\tumor|reference
    - Output TSV includes celltype (tumor/reference)
    - Split boxplot into Tumor vs Normal/Reference cells
"""

import argparse
import csv
import logging
import os
import sys
import math
import statistics
from collections import defaultdict

import gzip

import numpy as np
import sklearn

import weightedstats
import wcorr

def _open_text(path):
    """Open plain text or .gz transparently."""
    if path.endswith(".gz"):
        logging.info(f"Treating {path} as a gzip file.")
        return gzip.open(path, mode="rt", newline="")
    return open(path, "r", newline="")

# --- New imports for SCEVAN support ---
try:
    import pyreadr
except ImportError:
    pyreadr = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    plt = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", required=True,
                   help="Path to Ginkgo-like wide TSV ground truth")
    p.add_argument("--caller-result", required=True,
                   help="Path to caller output file")
    p.add_argument("--caller-name", required=True,
                   help="Name of the caller (for labelling)")
    p.add_argument("--gene-pos", required=True,
                   help="Gene position file: gene<TAB>chr<TAB>start<TAB>end")
    p.add_argument("--chrom-arm-pos", default=None,
                   help="Optional chromosome arm positions file (used for CaSpER)")
    p.add_argument("--output", required=True,
                   help="Output TSV path")
    p.add_argument("--plot", default=None,
                   help="Optional output plot path (default: output prefix + .boxplot.png)")
    p.add_argument("--scevan_count_mtx_annot_rdata_path", default=None,
                   help="The *_count_mtx_annot.RData file.")
    # NEW: Sample annotation argument
    p.add_argument("--sample-annotation", required=True,
                   help="Path to original author-provided sample annotation TSV (NO HEADER): original_sample_name\\ttumor|reference")
    p.add_argument("--dna-annotation", required=True,
                   help="Path to DNA sample annotation TSV (NO HEADER): DNA-inferred_sample_name\\ttumor|reference")
    p.add_argument("--rna-annotation", required=True,
                   help="Path to RNA sample annotation TSV (NO HEADER): RNA-inferred_sample_name\\ttumor|reference")
    return p.parse_args()

# NEW: Load sample annotation (2-column TSV: cell name -> tumor/reference)
def load_sample_annotation(path):
    annot = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                cell_name = parts[0].strip()
                cell_type = parts[1].strip()
                annot.append((cell_name, cell_type))
    return annot

def build_annot_dict(annot_list):
    """Return {cellpath2id(name): celltype} from load_sample_annotation output."""
    return {cellpath2id(name): ct for name, ct in annot_list}

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def normalise_chrom(chrom):
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        return chrom
    return "chr" + chrom


def interval_overlap(a_start, a_end, b_start, b_end):
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0, e - s)


def load_ground_truth_wide(path):
    """
    Load wide Ginkgo-style ground truth TSV.

    Returns:
        gt_by_cell: dict[cell_name] -> list of (chrom, start, end, cn)
        cell_names: list[str]
    """
    gt_by_cell = defaultdict(list)

    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        if len(header) < 4:
            raise ValueError("Ground truth header has too few columns.")

        cell_names = header[3:]

        for row in reader:
            if len(row) < 4:
                continue

            chrom = normalise_chrom(row[0])
            try:
                start = int(row[1])
                end = int(row[2])
            except ValueError:
                continue

            values = row[3:]
            for i, cell in enumerate(cell_names):
                if i >= len(values):
                    continue
                cn = safe_float(values[i])
                if cn is None:
                    continue
                gt_by_cell[cell].append((chrom, start, end, cn))

    return dict(gt_by_cell), cell_names


def load_gene_positions(path):
    """
    Load gene position file: gene chr start end
    Returns:
        genes: dict[gene] -> (chrom, start, end)
        total_exome_bp: int, union length across all listed gene intervals
    """
    genes = {}
    intervals_by_chrom = defaultdict(list)

    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            gene = parts[0]
            chrom = normalise_chrom(parts[1])

            try:
                start = int(parts[2])
                end = int(parts[3])
            except ValueError:
                continue

            genes[gene] = (chrom, start, end)
            intervals_by_chrom[chrom].append((start, end))

    total_exome_bp = 0
    for chrom, intervals in intervals_by_chrom.items():
        if not intervals:
            continue
        intervals.sort()
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total_exome_bp += cur_e - cur_s
                cur_s, cur_e = s, e
        total_exome_bp += cur_e - cur_s

    return genes, total_exome_bp

def load_chrom_arm_positions(path):
    """
    Load chromosome arm positions file.
    Expected columns (TAB-separated, with header):
        chromosome_name  start  end  arm   (arm like '1p', '1q', ...)
    Or the colomemaria format: Chrom Arm Start End
    Returns dict[arm_name] -> (chrom, start, end)
    """
    arm2pos = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        # auto-detect column layout
        hdr = [h.lower() for h in header]
        if "arm" in hdr:
            i_arm  = hdr.index("arm")
            i_chr  = hdr.index("chrom") if "chrom" in hdr else hdr.index("chromosome_name")
            i_s    = hdr.index("start")
            i_e    = hdr.index("end")
        else:
            # assume:  chrom arm start end # Manual override
            i_chr, i_arm, i_s, i_e = 1, 0, 2, 3
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_chr, i_arm, i_s, i_e):
                continue
            try:
                chrom = normalise_chrom(parts[i_chr])
                arm_suffix = parts[i_arm].strip().lower()  # 'p' or 'q' or '1p'
                start = int(parts[i_s]); end = int(parts[i_e])
            except ValueError:
                continue
            # Normalise arm name: want '1p', '2q', ...
            if arm_suffix in ("p", "q"):
                chrom_num = chrom.replace("chr", "")
                arm_name = chrom_num + arm_suffix
            else:
                arm_name = arm_suffix.replace("chr", "")
            arm2pos[arm_name] = (chrom, start, end)
    return arm2pos

def detect_caller_format(path, caller_name):
    cn = caller_name.lower()
    if "copykat" in cn:
        return "copykat"
    
    # --- Modified logic for SCEVAN ---
    if "scevan" in cn:
        if path.endswith(".RData") or path.endswith(".rda"):
            return "scevan_rdata"
        else:
            return "scevan_seg"
            
    if "infercnv" in cn:
        return "infercnv"
    # --- infercna detection (similar to infercnv, gene x cell matrix) ---
    if "infercna" in cn:
        return "infercna"
    if "conicsmat" in cn or "conics" in cn:
        return "conicsmat"
    if "numbat" in cn:                # NEW
        return "numbat"
    if "casper" in cn:                # NEW
        return "casper"

    with _open_text(path) as f:
        header = f.readline().strip().lower()
        if "abspos" in header and "chromosome_name" in header:
            return "copykat"
        if header.startswith("chr\t") and "cn" in header:
            return "scevan_seg"
    return "generic_matrix"


def load_matrix_like_caller(path, caller_format, gene_positions, gt_cell_names, scevan_count_mtx_annot_rdata_path=None):
    """
    Load matrix-like caller outputs into per-cell genomic features.

    Returns:
        caller_by_cell: dict[cell_label] -> list of (chrom, start, end, value)
        exome_fraction: float
        caller_cell_names: list[str]
    """
    caller_by_cell = defaultdict(list)

    # --- New Branch for SCEVAN RData (does not use text file reader) ---
    if caller_format == "scevan_rdata":
        if pyreadr is None:
            raise ImportError("Please install 'pyreadr' to read SCEVAN files: pip install pyreadr")
        if pd is None:
            raise ImportError("Please install 'pandas' to process SCEVAN files: pip install pandas")

        # Read RData file
        r_data = pyreadr.read_r(path)
        r_anno = pyreadr.read_r(scevan_count_mtx_annot_rdata_path)
        
        # Attempt to find the matrix (SCEVAN usually stores it as 'CNAmtx')
        cna_mtx = None
        for key, val in r_data.items():
            if isinstance(val, pd.DataFrame):
                cna_mtx = val
                break
        if cna_mtx is None:
            raise ValueError(f"Could not find a DataFrame matrix in {path}. Ensure this is a SCEVAN '*_CNAmtx.RData' file.")

        assert 'count_mtx_annot' in r_anno, f'The annotation file {scevan_count_mtx_annot_rdata_path} does not contain the required key count_mtx_annot!'
        annot_mtx = r_anno['count_mtx_annot']
        assert isinstance(annot_mtx, pd.DataFrame), f'The annotation {annot_mtx} from {scevan_count_mtx_annot_rdata_path} is not a pandas.DataFrame!'

        assert len(annot_mtx) == len(cna_mtx), f'The matrices from {path} and {scevan_count_mtx_annot_rdata_path} do not have the same number of rows!'

        caller_cell_names = cna_mtx.columns.tolist()
        covered_intervals = defaultdict(list)

        # Iterate through genes (rows)
        for row_idx, (gene_name, row_data) in enumerate(cna_mtx.iterrows()):
            gene_name = annot_mtx['gene_name'].iloc[row_idx]
            # Match gene name to coordinates
            if gene_name not in gene_positions:
                continue
            
            chrom, start, end = gene_positions[gene_name]
            covered_intervals[chrom].append((start, end))

            # Iterate through cells (columns)
            for cell_name in caller_cell_names:
                val = safe_float(row_data[cell_name])
                if val is None:
                    continue
                caller_by_cell[cell_name].append((chrom, start, end, val))

        return dict(caller_by_cell), covered_intervals, caller_cell_names

    # --- Original logic for text-based formats ---
    
    with _open_text(path) as f:
        sample = f.read(2**22) # Read a small chunk to analyze
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t ')
        reader = csv.reader(f, dialect)
        header = next(reader)

        if caller_format == "copykat":
            # Expected example:
            # abspos chromosome_name start_position end_position ensembl_gene_id
            # hgnc_symbol band V2 V3 ...
            meta_cols = 7
            if len(header) <= meta_cols:
                raise ValueError("copykat-like header does not contain cell columns.")

            caller_cell_names = header[meta_cols:]

            covered_intervals = defaultdict(list)

            for row in reader:
                if len(row) <= meta_cols:
                    continue

                chrom = normalise_chrom(row[1])
                try:
                    start = int(row[2])
                    end = int(row[3])
                except ValueError:
                    continue

                covered_intervals[chrom].append((start, end))

                vals = row[meta_cols:]
                for i, cell in enumerate(caller_cell_names):
                    if i >= len(vals):
                        continue
                    v = safe_float(vals[i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))

            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format == "infercna":
            # infercna output: gene x cell matrix with log2 ratios
            # Header: first column is empty/header, followed by cell names
            if len(header) < 2:
                raise ValueError("infercna matrix has too few columns.")
            caller_cell_names = header[1:]
            covered_intervals = defaultdict(list)
            for row in reader:
                if len(row) < 2:
                    continue
                feature = row[0]
                if feature not in gene_positions:
                    continue
                chrom, start, end = gene_positions[feature]
                covered_intervals[chrom].append((start, end))
                vals = row[1:]
                for i, cell in enumerate(caller_cell_names):
                    if i >= len(vals):
                        continue
                    v = safe_float(vals[i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))
            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format in ("infercnv", "generic_matrix"):
            # infercnv-like:
            # first column = gene/feature, remaining columns = cells
            if len(header) < 2:
                raise ValueError("Matrix-like caller result has too few columns.")

            caller_cell_names = header[1:]
            covered_intervals = defaultdict(list)

            for row in reader:
                if len(row) < 2:
                    continue

                feature = row[0]

                if feature not in gene_positions:
                    continue

                chrom, start, end = gene_positions[feature]
                covered_intervals[chrom].append((start, end))

                vals = row[1:]
                for i, cell in enumerate(caller_cell_names):
                    if i >= len(vals):
                        continue
                    v = safe_float(vals[i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))

            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format == "scevan_seg":
            # Segment-style file. We still load it, but it is not naturally cell-by-cell.
            # We store one pseudo-cell so the rest of the code can run.
            pseudo = "SEGMENT_RESULT"
            caller_cell_names = [pseudo]
            covered_intervals = defaultdict(list)

            for row in reader:
                if len(row) < 4:
                    continue
                chrom = normalise_chrom(row[0])
                try:
                    start = int(row[1])
                    end = int(row[2])
                    val = float(row[3])
                except ValueError:
                    continue
                covered_intervals[chrom].append((start, end))
                caller_by_cell[pseudo].append((chrom, start, end, val))

            return dict(caller_by_cell), covered_intervals, caller_cell_names
        
        # newly added from https://sorryios.ai/chat/dd8f4c23-9f1e-450e-af0b-a8f7128a02db
        elif caller_format == "casper":
            # CaSpER cell_matrix.tsv: gene-rows x cell-cols, discrete -1/0/1.
            # The header line lists ONLY cell names (R write.table default,
            # row.names but no name for the rowname column), so each data
            # row has len(header)+1 fields: [gene, val_1, ..., val_N].
            if len(header) < 1:
                raise ValueError("CaSpER cell_matrix has empty header.")
            caller_cell_names = header                      # all of header = cells
            n_cells = len(caller_cell_names)
            covered_intervals = defaultdict(list)
            for row in reader:
                if len(row) < n_cells + 1:
                    continue
                feature = row[0]
                if feature not in gene_positions:
                    continue
                chrom, start, end = gene_positions[feature]
                covered_intervals[chrom].append((start, end))
                vals = row[1:n_cells + 1]
                for i, cell in enumerate(caller_cell_names):
                    v = safe_float(vals[i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))
            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format == "numbat":
            # Numbat gexp_roll_wide.tsv(.gz): cell-rows x gene-cols.
            # Header: 'cell\tgene1\tgene2\t...'   -> header[0] == 'cell'
            # Data:   '<cell>\tval1\tval2\t...'
            # Transpose on the fly into per-cell intervals.
            if len(header) < 2:
                raise ValueError("Numbat gexp_roll_wide has too few columns.")
            gene_names = header[1:]
            # Pre-resolve gene -> coords once
            gene_coords = [gene_positions.get(g) for g in gene_names]
            caller_cell_names = []
            covered_intervals = defaultdict(list)
            for row in reader:
                if len(row) < 2:
                    continue
                cell = row[0]
                caller_cell_names.append(cell)
                vals = row[1:]
                for g_idx, coords in enumerate(gene_coords):
                    if coords is None or g_idx >= len(vals):
                        continue
                    v = safe_float(vals[g_idx])
                    if v is None:
                        continue
                    chrom, start, end = coords
                    caller_by_cell[cell].append((chrom, start, end, v))
                    covered_intervals[chrom].append((start, end))
            # covered_intervals will accumulate duplicates; that's fine,
            # compute_exome_fraction merges them anyway.
            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format == "conicsmat":
            # CONICSmat cnv_types.tsv: cells x chromosome-arms, discrete -1/0/1
            # (or 'gain'/'loss'/'neutral' depending on version).
            # Requires --chrom-arm-pos.
            arm_pos_path = getattr(load_matrix_like_caller, "_arm_pos_path", None)
            if arm_pos_path is None:
                raise ValueError("CONICSmat needs --chrom-arm-pos on the command line.")
            arm2pos = load_chrom_arm_positions(arm_pos_path)

            def _norm_arm(a):
                return a.strip().lower().replace("chr", "").replace("_", "").replace('"', '')
            def _norm_cell(a):
                return a.strip().replace('"', '')

            arm_keys = [_norm_arm(a) for a in header[1:]]
            _str2num = {"gain": 1.0, "amp": 1.0, "loss": -1.0, "del": -1.0,
                        "neutral": 0.0, "none": 0.0, "": 0.0}
            caller_cell_names = []
            covered_intervals = defaultdict(list)
            for row in reader:
                if len(row) < 2:
                    continue
                cell = _norm_cell(row[0])
                caller_cell_names.append(cell)
                vals = row[1:]
                for i, arm in enumerate(arm_keys):
                    if i >= len(vals) or arm not in arm2pos:
                        logging.warning(f"Something wrong: {i} >= {len(vals)} or arm {arm} not in arm2pos {arm2pos}!")
                        continue
                    raw = vals[i].strip()
                    v = safe_float(raw)
                    if v is None:
                        v = _str2num.get(raw.lower())
                    if v is None:
                        logging.warning(f"The line {row} in the file {path} is invalid since {raw} is not a valid copy-number signal! Skipping this signal!")
                        continue
                    chrom, start, end = arm2pos[arm]
                    caller_by_cell[cell].append((chrom, start, end, v))
                    covered_intervals[chrom].append((start, end))
            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format in ("casper?", "conicsmat?"):
            # cells x chromosome-arms, discrete -1/0/1 (or strings for CONICSmat).
            # Requires --chrom-arm-pos.
            if not hasattr(load_matrix_like_caller, "_arm_pos_path") \
                    or load_matrix_like_caller._arm_pos_path is None:
                raise ValueError(
                    f"{caller_format} format needs --chrom-arm-pos on the command line."
                )
            arm2pos = load_chrom_arm_positions(load_matrix_like_caller._arm_pos_path)
            arm_names_raw = header[1:]            # first col = cell name / rowname
            # Normalise arm-name strings ('chr1p' -> '1p', '1_p' -> '1p', etc.)
            def _norm_arm(a):
                a = a.strip().lower().replace("chr", "").replace("_", "")
                return a
            arm_keys = [_norm_arm(a) for a in arm_names_raw]

            caller_cell_names = []
            covered_intervals = defaultdict(list)
            # Map CONICSmat string codes -> numeric
            _str2num = {"gain": 1.0, "amp": 1.0, "loss": -1.0, "del": -1.0,
                        "neutral": 0.0, "none": 0.0, "": 0.0}
            for row in reader:
                if len(row) < 2:
                    continue
                cell = row[0]
                caller_cell_names.append(cell)
                vals = row[1:]
                for i, arm in enumerate(arm_keys):
                    if i >= len(vals) or arm not in arm2pos:
                        continue
                    raw = vals[i].strip()
                    v = safe_float(raw)
                    if v is None:
                        v = _str2num.get(raw.lower())
                    if v is None:
                        continue
                    chrom, start, end = arm2pos[arm]
                    caller_by_cell[cell].append((chrom, start, end, v))
                    covered_intervals[chrom].append((start, end))
            return dict(caller_by_cell), covered_intervals, caller_cell_names
        
        elif caller_format == "conicsmat??":
            raise ValueError("conicsmat arm-level format is not supported in this cell-by-cell exome evaluation script.")

        else:
            raise ValueError(f"Unsupported caller format: {caller_format}")


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def compute_exome_fraction(covered_intervals_by_chrom, gene_positions_total_bp):
    if gene_positions_total_bp <= 0:
        return float("nan")

    covered_bp = 0
    for chrom, intervals in covered_intervals_by_chrom.items():
        for s, e in merge_intervals(intervals):
            covered_bp += (e - s)

    frac = covered_bp / gene_positions_total_bp
    return min(1.0, max(0.0, frac))


def classify_gt_cn(cn):
    if cn > 2:
        return "gain"
    if cn < 2:
        return "loss"
    return "neutral"


def classify_pred_value(med, value, caller_format):
    """
    Classify a prediction into gain/loss/neutral.

    For copykat-like centered signals, predicted classes are assigned using
    thresholds around 0 by default:
        loss < -0.1, neutral [-0.1, 0.1], gain > 0.1
    For absolute copy-number-like predictions, the script falls back to
    median-relative thresholds.
    """
    if caller_format in ("copykat", "infercnv", "generic_matrix",
                         "infercna", # infercna uses similar log2 ratio format
                         "scevan_rdata", "numbat",            # NEW
                         "casper", "conicsmat"):              # NEW
        # Check if data look centered around zero.
        # med = statistics.median(values) if values else 0.0
        if abs(med) < 0.5:
            if value > 0.1:
                return "gain"
            if value < -0.1:
                return "loss"
            return "neutral"

    # Absolute-CN style or fallback.
    # med = statistics.median(values) if values else 2.0
    if med > 1.0:
        if value > 2.5:
            return "gain"
        if value < 1.5:
            return "loss"
        return "neutral"

    if value > med + 0.1:
        return "gain"
    if value < med - 0.1:
        return "loss"
    return "neutral"


def rankdata(values):
    """
    Average-rank implementation for ties.
    """
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_correlation(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / (sx * sy)


def spearman_correlation(x, y):
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_correlation(rx, ry)


def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, fscore


def compute_cell_classification_benchmark(pred_types, gt_types):
    """
    Benchmark tumor-vs-normal cell labelling.
    pred_types, gt_types: parallel lists of 'tumor' | 'reference' | None.
    Returns a dict of classification metrics.
    """
    valid = [
        (p, g) for p, g in zip(pred_types, gt_types)
        if p in ("tumor", "reference") and g in ("tumor", "reference")
    ]
    if len(valid) < 2:
        return {}

    pred_labels = [1 if p == "tumor" else 0 for p, g in valid]
    gt_labels   = [1 if g == "tumor" else 0 for p, g in valid]

    tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
        else float("nan")
    )
    accuracy = (tp + tn) / len(valid)

    auc = (
        sklearn.metrics.roc_auc_score(gt_labels, pred_labels)
        if len(set(gt_labels)) == 2
        else float("nan")
    )

    return {
        "n_cells":         len(valid),
        "tumor_precision": precision,
        "tumor_recall":    recall,
        "tumor_f1":        f1,
        "accuracy":        accuracy,
        "roc_auc":         auc,
    }

def build_overlap_pairs(gt_intervals, pred_intervals):
    """
    Pair GT and prediction values through interval overlap.

    Returns:
        list of (gt_cn, pred_value)
    """
    gt_by_chrom = defaultdict(list)
    pred_by_chrom = defaultdict(list)

    for chrom, start, end, cn in gt_intervals:
        gt_by_chrom[chrom].append((start, end, cn))
    for chrom, start, end, val in pred_intervals:
        pred_by_chrom[chrom].append((start, end, val))

    for chrom in gt_by_chrom:
        gt_by_chrom[chrom].sort()
    for chrom in pred_by_chrom:
        pred_by_chrom[chrom].sort()

    pairs = []

    for chrom, gt_list in gt_by_chrom.items():
        preds = pred_by_chrom.get(chrom, [])
        if not preds:
            continue

        j = 0
        for gstart, gend, gcn in gt_list:
            overlapping = []

            while j < len(preds) and preds[j][1] <= gstart:
                j += 1

            k = j
            while k < len(preds) and preds[k][0] < gend:
                pstart, pend, pval = preds[k]
                ov = interval_overlap(gstart, gend, pstart, pend)
                if ov > 0:
                    overlapping.append((ov, pval))
                k += 1

            if overlapping:
                total_ov = sum(ov for ov, _ in overlapping)
                weighted_pred = sum(ov * pval for ov, pval in overlapping) / total_ov
                pairs.append((gcn, weighted_pred))

    return pairs

# Debugged with prompts at https://sorryios.ai/chat/9af3a6ff-4e2e-4700-9b64-6c5058f6fe53
def build_overlap_intervals(gt_intervals, pred_intervals):
    """
    Pair GT and prediction values through interval overlap.
    Returns:
        list of (chrom, start, end, gt_cn, pred_cn) where each entry
        is a sub-interval from the intersection of a GT interval and
        a single prediction interval.
    """
    gt_by_chrom = defaultdict(list)
    pred_by_chrom = defaultdict(list)
    for chrom, start, end, cn in gt_intervals:
        gt_by_chrom[chrom].append((start, end, cn))
    for chrom, start, end, val in pred_intervals:
        pred_by_chrom[chrom].append((start, end, val))
    for chrom in gt_by_chrom:
        gt_by_chrom[chrom].sort()
    for chrom in pred_by_chrom:
        pred_by_chrom[chrom].sort()
    ret = []
    for chrom, gt_list in gt_by_chrom.items():
        preds = pred_by_chrom.get(chrom, [])
        if not preds:
            continue
        j = 0
        for gstart, gend, gcn in gt_list:
            while j < len(preds) and preds[j][1] <= gstart:
                j += 1
            k = j
            while k < len(preds) and preds[k][0] < gend:
                pstart, pend, pval = preds[k]
                ov_start = max(gstart, pstart)
                ov_end = min(gend, pend)
                if ov_start < ov_end:
                    ret.append((chrom, ov_start, ov_end, gcn, pval))
                k += 1
    return ret

def compute_metrics_for_cell(gt_intervals, pred_intervals, caller_format, exome_fraction):
    # pairs = build_overlap_pairs(gt_intervals, pred_intervals)
    # list of (chrom, ov_start, ov_end, gcn, pval)
    intersect_intervals = build_overlap_intervals(gt_intervals, pred_intervals)
    
    if not intersect_intervals:
        return {
            "CopyNumber gain precision": float("nan"),
            "CopyNumber gain recall": float("nan"),
            "CopyNumber gain F-score": float("nan"),
            "CopyNumber loss precision": float("nan"),
            "CopyNumber loss recall": float("nan"),
            "CopyNumber loss F-score": float("nan"),
            "Multiclass classification accuracy": float("nan"),
            "Pearson Correlation Coefficient": float("nan"),
            "Spearman Correlation Coefficient": float("nan"),
            "Fraction of the exome with inferred copy numbers": exome_fraction,
            "n_overlap_pairs": 0,
        }

    weights = [(x[2]-x[1]) for x in intersect_intervals]
    gt_vals = [x[3] for x in intersect_intervals]
    pred_vals = [x[4] for x in intersect_intervals]

    # https://www.doubao.com/chat/38420994951737090
    # CRITICAL FIX: Compute median ONCE per cell to avoid O(n^2) runtime 
    # pred_median = statistics.median(pred_vals) if pred_vals else 0.0
    pred_median = weightedstats.weighted_median(pred_vals, weights=weights)

    gt_cls = [classify_gt_cn(v) for v in gt_vals]
    pred_cls = [classify_pred_value(pred_median, v, caller_format) for v in pred_vals]

    # TODO: RNA-seq-based callers do not generate integer copy numbers,
    # so how to compute classification metrics (such as true-positive, accuracy, and F-score) here?
    # https://academic.oup.com/bib/article/26/2/bbaf076/8051529
    #  - the word accuracy does not denote the one for classification
    #  - F1 score was for LOH detection and tumor-vs-normal classification of cells
    #  - no integer copy numbers were evaluated
    # https://www.nature.com/articles/s41467-025-62359-9
    #  - accuracy was only used for tumor-vs-normal classification of cells
    #  - RNA-seq threshold was variable, thereby giving ROC and maximized F1 score
    #  - did not evaluate breakpoint detection in RNA-seq
    #  - no integer copy numbers were evaluated otherwise

    tp_gain = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g == "gain" and p == "gain")
    fp_gain = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g != "gain" and p == "gain")
    fn_gain = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g == "gain" and p != "gain")

    tp_loss = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g == "loss" and p == "loss")
    fp_loss = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g != "loss" and p == "loss")
    fn_loss = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g == "loss" and p != "loss")

    gain_precision, gain_recall, gain_fscore = compute_prf(tp_gain, fp_gain, fn_gain)
    loss_precision, loss_recall, loss_fscore = compute_prf(tp_loss, fp_loss, fn_loss)

    multiclass_acc = sum(w for w, g, p in zip(weights, gt_cls, pred_cls) if g == p) / sum(weights) if gt_cls else float("nan")

    pearson_r = wcorr.WeightedCorr(x=gt_vals, y=pred_vals, w=weights)(method='pearson')
    spearman_r = wcorr.WeightedCorr(x=gt_vals, y=pred_vals, w=weights)(method='spearman')

    gt_gain_bools = [(1 if v > 2 else 0) for v in gt_vals]
    gt_loss_bools = [(1 if v < 2 else 0) for v in gt_vals]
    gain_weighted_auc = sklearn.metrics.roc_auc_score(gt_gain_bools, 0 + np.array(pred_vals), sample_weight=weights)
    loss_weighted_auc = sklearn.metrics.roc_auc_score(gt_loss_bools, 0 - np.array(pred_vals), sample_weight=weights)
    # gain_weighted_f1s = sklearn.metrics.f1_score(gt_gain_bools, 0 + np.array(pred_vals), sample_weight=weights)
    # loss_weighted_f1s = sklearn.metrics.f1_score(gt_loss_bools, 0 - np.array(pred_vals), sample_weight=weights)

    return {
        "CopyNumber gain precision": gain_precision,
        "CopyNumber gain recall": gain_recall,
        "CopyNumber gain F-score": gain_fscore,
        "CopyNumber loss precision": loss_precision,
        "CopyNumber loss recall": loss_recall,
        "CopyNumber loss F-score": loss_fscore,
        "Multiclass classification accuracy": multiclass_acc,
        "Pearson Correlation Coefficient": pearson_r,
        "Spearman Correlation Coefficient": spearman_r,
        #"CopyNumber gain F1-score": gain_weighted_f1s,
        #"CopyNumber loss F1-score": loss_weighted_f1s,
        "CopyNumber gain ROC-AUC": gain_weighted_auc,
        "CopyNumber loss ROC-AUC": loss_weighted_auc,
        "Fraction of the exome with inferred copy numbers": exome_fraction,
        "n_overlap_pairs": sum(weights), # len(pairs), # number of base-pairs in the evaluated genome intervals
    }

# MODIFIED: Split boxplot into Tumor vs Normal/Reference
def make_boxplot(long_rows, caller_name, out_png):
    if plt is None:
        sys.stderr.write("WARNING: matplotlib not installed. Skipping boxplot.\n")
        return

    metric_order = [
        "Pearson Correlation Coefficient",
        "Spearman Correlation Coefficient",
        "CopyNumber gain ROC-AUC",
        "CopyNumber loss ROC-AUC",
        "Fraction of the exome with inferred copy numbers",
        "Fraction of the cells with inferred copy numbers",
        
        #"CopyNumber gain precision",
        #"CopyNumber gain recall",
        #"CopyNumber gain F-score",
        #"CopyNumber loss precision",
        #"CopyNumber loss recall",
        #"CopyNumber loss F-score",
        #"Multiclass classification accuracy",
    ]

    # Three stratification schemes: (field_in_row, label, tumor_color, normal_color)
    stratifications = [
        ("celltype",     "Sample Annotation",  "#FF6B6B", "#4ECDC4"),
        ("celltype_dna", "DNA Annotation",     "#E07B54", "#54A0C8"),
        ("celltype_real_rna", "Real RNA Annotation",  "#C966CC", "#66CC88"),
        ("celltype_refset_rna", "AtLeastOneRefCluster RNA Annotation", "#C966CC", "#66CC88"),
    ]

    n_rows = len(stratifications)
    fig, axes = plt.subplots(n_rows, 2, figsize=(24, 8 * n_rows), sharey="row")

    def group_metrics(rows):
        grouped = defaultdict(list)
        for r in rows:
            m, v = r["metric"], r["value"]
            if m in metric_order and not math.isnan(v):
                grouped[m].append(v)
        return [grouped[m] for m in metric_order]

    for row_idx, (ct_field, strat_label, t_color, n_color) in enumerate(stratifications):
        tumor_rows  = [r for r in long_rows if r.get(ct_field) == "tumor"]
        normal_rows = [r for r in long_rows if r.get(ct_field) == "reference"]

        ax_t = axes[row_idx][0]
        ax_n = axes[row_idx][1]

        for ax, rows, color, side in (
            (ax_t, tumor_rows,  t_color, "Tumor"),
            (ax_n, normal_rows, n_color, "Normal/Reference"),
        ):
            n_cells = len({r["ground_truth_cell"] for r in rows})
            bp = ax.boxplot(group_metrics(rows), patch_artist=True,
                            labels=metric_order, showfliers=True)
            ax.grid(which="both", axis="y")
            ax.set_title(f"[{strat_label}] {side} (n={n_cells})", fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=30, fontsize=7, ha="right",
                               va="top", rotation_mode="anchor")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

        ax_t.set_ylabel("Performance Score")

    for ax in axes.flatten():
        ax.set_xlabel("Evaluation Metric")

    legend_handles = [
        Patch(facecolor="#FF6B6B", edgecolor="black", label="Tumor"),
        Patch(facecolor="#4ECDC4", edgecolor="black", label="Normal/Reference"),
    ]
    fig.suptitle(
        f"CNV Caller Performance: {caller_name}\n"
        f"Rows: sample-annotation / DNA-annotation / RNA-annotation",
        fontsize=13,
    )
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _old_make_boxplot(long_rows, caller_name, out_png):
    if plt is None:
        sys.stderr.write("WARNING: matplotlib is not installed. Skipping boxplot.\n")
        return

    metric_order = [
        "Pearson Correlation Coefficient",
        "Spearman Correlation Coefficient",
        #"CopyNumber gain F1-score",
        #"CopyNumber loss F1-score",
        "CopyNumber gain ROC-AUC",
        "CopyNumber loss ROC-AUC",
        "Fraction of the exome with inferred copy numbers",
        'Fraction of the cells with inferred copy numbers',
        # The following are ill-defined
        # because integer copy numbers cannot be inferred from RNA-seq data
        "CopyNumber gain precision",
        "CopyNumber gain recall",
        "CopyNumber gain F-score",
        "CopyNumber loss precision",
        "CopyNumber loss recall",
        "CopyNumber loss F-score",
        "Multiclass classification accuracy",
    ]

    # Split data by celltype
    tumor_rows = [r for r in long_rows if r["celltype"] == "tumor"]
    normal_rows = [r for r in long_rows if r["celltype"] == "reference"]

    def group_metrics(rows):
        grouped = defaultdict(list)
        for row in rows:
            metric = row["metric"]
            val = row["value"]
            if metric in metric_order and not math.isnan(val):
                grouped[metric].append(val)
        return [grouped[m] for m in metric_order]

    tumor_data = group_metrics(tumor_rows)
    normal_data = group_metrics(normal_rows)

    # Create 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    
    # Plot Tumor cells
    bp1 = ax1.boxplot(tumor_data, patch_artist=True, labels=metric_order, showfliers=True)
    ax1.grid(which='both', axis='y')
    ax1.set_title(f"Tumor Cells (n={len(tumor_rows)//len(metric_order)})")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, fontsize=8, ha='right', va='top', rotation_mode='anchor')

    for patch in bp1["boxes"]:
        patch.set_facecolor("#FF6B6B")
        patch.set_alpha(0.8)

    # Plot Normal/Reference cells
    bp2 = ax2.boxplot(normal_data, patch_artist=True, labels=metric_order, showfliers=True)
    ax2.grid(which='both', axis='y', linestyle='-')
    ax2.set_title(f"Normal/Reference Cells (n={len(normal_rows)//len(metric_order)})")

    # Tested at: https://colab.research.google.com/drive/1fqksWh1jupl5w2mxlR5F6zKJpdVUMfAe#scrollTo=udfq5Rjm0rN1
    # ax2.tick_params(axis='x', rotation=30, labelsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, fontsize=8, ha='right', va='top', rotation_mode='anchor')

    for patch in bp2["boxes"]:
        patch.set_facecolor("#4ECDC4")
        patch.set_alpha(0.8)

    # Global labels
    fig.suptitle(f"CNV Caller Performance: {caller_name} (By Cell Type)", fontsize=14)
    ax1.set_ylabel("Performance Score")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Evaluation Metric")

    # Legend
    legend_handles = [
        Patch(facecolor="#FF6B6B", edgecolor="black", label="Tumor"),
        Patch(facecolor="#4ECDC4", edgecolor="black", label="Normal/Reference")
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def cellpath2id(c):
    multidots = len(c.split('.')) > 2
    if '..' in c: sep = '..'
    elif multidots: sep = '.'
    else: sep = '..'
    return c.split(sep)[-1].replace('/', '.').replace('_', '.').replace('-', '.')

def cellnames_to_id2name(cell_names):
    keys = [cellpath2id(c) for c in cell_names]
    seen = set()
    duplicates = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate keys from cellpath2id: {sorted(duplicates)}")
    return {cellpath2id(c) : c for c in cell_names}

def main():
    args = parse_args()

    logging.info('Started loading ground truth')
    gt_by_cell, gt_cell_names = load_ground_truth_wide(args.ground_truth)
    # NEW: Load sample annotation
    logging.info('Started loading sample annotation')
    sample_annot = load_sample_annotation(args.sample_annotation)
    dna_annot    = load_sample_annotation(args.dna_annotation)
    rna_annot    = load_sample_annotation(args.rna_annotation)
    rna_real_annot = load_sample_annotation(args.rna_annotation + '.maybe_zero_refs')
    
    # Keyed by normalised cell ID
    sample_annot_dict = build_annot_dict(sample_annot)
    dna_annot_dict    = build_annot_dict(dna_annot)    # matched against gt_cell (DNA space)
    rna_refset_annot_dict    = build_annot_dict(rna_annot)    # matched against pred_cell (RNA space)
    rna_real_annot_dict = build_annot_dict(rna_real_annot)

    logging.info('Started loading gene positions')
    gene_positions, total_exome_bp = load_gene_positions(args.gene_pos)
    caller_format = detect_caller_format(args.caller_result, args.caller_name)

    load_matrix_like_caller._arm_pos_path = args.chrom_arm_pos   # NEW
    caller_by_cell, covered_intervals_by_chrom, caller_cell_names = load_matrix_like_caller(
        args.caller_result, caller_format, gene_positions, gt_cell_names,
        scevan_count_mtx_annot_rdata_path=args.scevan_count_mtx_annot_rdata_path,
    )

    if not gt_by_cell:
        print("ERROR: Ground truth is empty.", file=sys.stderr)
        sys.exit(1)

    if not caller_by_cell:
        print(f"ERROR: No caller signal loaded from {args.caller_result}.", file=sys.stderr)
        sys.exit(1)

    cell_to_gt_name = cellnames_to_id2name(gt_cell_names)
    cell_to_caller_name = cellnames_to_id2name(caller_cell_names)
    common_cells = set(cell_to_gt_name.keys()) & set(cell_to_caller_name.keys())

    logging.info('Started computing exome fraction')
    exome_fraction = compute_exome_fraction(covered_intervals_by_chrom, total_exome_bp)

    def multiline_zip(list1, list2):
        ret = []
        for e1, e2 in zip(list1, list2):
            ret.append('  ' + str(e1) + '\n  ' + str(e2))
        return '\n,\n'.join(ret)

    gt_cell_names_2 = sorted(gt_cell_names)
    if gt_cell_names != gt_cell_names_2:
        logging.warning(
            f'The cmd-line params {args} results in something expected: cells in the truth set was not sorted. '
            f'{multiline_zip(gt_cell_names, gt_cell_names_2)} is not element-wise matching)')
        gt_cell_names = gt_cell_names_2
    caller_cell_names_2 = sorted(caller_cell_names)
    if caller_cell_names != caller_cell_names_2:
        logging.warning(
            f'The cmd-line params {args} results in something expected: '
            'cells in the call set was not sorted, so perform sorting. ')
        caller_cell_names = caller_cell_names_2

    # Match cells by order, as requested by the user.
    if caller_format in ("copykat", "infercnv", "generic_matrix", "scevan_rdata", "infercna",
                         "numbat", "casper", "conicsmat"):    # extended
        n_match = min(len(gt_cell_names), len(caller_cell_names))
        if n_match == 0:
            print("ERROR: No overlapping cell columns by order.", file=sys.stderr)
            sys.exit(1)

        # cell_pairs = [(gt_cell_names[i], caller_cell_names[i]) for i in range(n_match)]
        cell_pairs =  [(cell_to_gt_name[c], cell_to_caller_name[c]) for c in sorted(common_cells)]
    else:
        # Segment-like fallback
        only_caller_cell = caller_cell_names[0]
        cell_pairs = [(gt_cell_names[0], only_caller_cell)]

    long_rows = []
    summary_rows = []

    # ── collect per-cell annotation tuples for the classification benchmarks ──
    cell_ct_sample, cell_ct_dna, cell_ct_rna, cell_ct_real_rna = [], [], [], []

    for gt_cell, pred_cell in cell_pairs:
        gt_id   = cellpath2id(gt_cell)
        pred_id = cellpath2id(pred_cell)

        celltype_sample = sample_annot_dict.get(gt_id, "unknown")
        celltype_dna    = dna_annot_dict.get(gt_id, "unknown")
        celltype_refset_rna    = rna_refset_annot_dict.get(pred_id, "unknown")
        celltype_real_rna = rna_real_annot_dict.get(pred_id, "unknown")

        cell_ct_sample.append(celltype_sample)
        cell_ct_dna.append(celltype_dna)
        cell_ct_rna.append(celltype_refset_rna)
        cell_ct_real_rna.append(celltype_real_rna)

        logging.info(f"Processing {gt_cell} <-> {pred_cell} | "
                     f"sample={celltype_sample}  dna={celltype_dna}  rna={celltype_refset_rna}  real_rna={celltype_real_rna}")

        gt_intervals   = gt_by_cell.get(gt_cell, [])
        pred_intervals = caller_by_cell.get(pred_cell, [])
        metrics = compute_metrics_for_cell(
            gt_intervals=gt_intervals,
            pred_intervals=pred_intervals,
            caller_format=caller_format,
            exome_fraction=exome_fraction,
        )
        metrics["Fraction of the cells with inferred copy numbers"] = (
            len(common_cells) / float(len(cell_to_gt_name))
        )
        summary_rows.append((gt_cell, pred_cell, celltype_sample, metrics.get("n_overlap_pairs", 0)))

        for metric_name, metric_value in metrics.items():
            if metric_name == "n_overlap_pairs":
                continue
            long_rows.append({
                "caller":            args.caller_name,
                "ground_truth_cell": gt_cell,
                "caller_cell":       pred_cell,
                "celltype":          celltype_sample,   # ← backward-compatible
                "celltype_dna":      celltype_dna,      # ← new
                "celltype_refset_rna":      celltype_refset_rna,      # ← new
                "celltype_real_rna": celltype_real_rna, # ← new
                "metric":            metric_name,
                "value":             metric_value,
            })

    '''
    for (gt_cell, pred_cell), (orig_cell, celltype) in zip(cell_pairs, sample_annot):
        logging.info(f'Started iterating over the cell {gt_cell} {pred_cell} {orig_cell} {celltype}')
        gt_intervals = gt_by_cell.get(gt_cell, [])
        pred_intervals = caller_by_cell.get(pred_cell, [])
        metrics = compute_metrics_for_cell(
            gt_intervals=gt_intervals,
            pred_intervals=pred_intervals,
            caller_format=caller_format,
            exome_fraction=exome_fraction,
        )
        metrics['Fraction of the cells with inferred copy numbers'] = len(common_cells) / float(len(cell_to_gt_name))
        summary_rows.append((gt_cell, pred_cell, celltype, metrics.get("n_overlap_pairs", 0)))

        for metric_name, metric_value in metrics.items():
            if metric_name == "n_overlap_pairs":
                continue
            # NEW: Add celltype to output rows
            long_rows.append({
                "caller": args.caller_name,
                "ground_truth_cell": gt_cell,
                "caller_cell": pred_cell,
                "celltype": celltype,
                "metric": metric_name,
                "value": metric_value,
            })
    '''
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clf_benchmarks = {
        "rna_vs_dna":    compute_cell_classification_benchmark(cell_ct_real_rna, cell_ct_dna),
        "rna_vs_sample": compute_cell_classification_benchmark(cell_ct_real_rna, cell_ct_sample),
        "dna_vs_sample": compute_cell_classification_benchmark(cell_ct_dna, cell_ct_sample),
    }

    clf_out = os.path.splitext(args.output)[0] + ".cell_classification.tsv"
    with open(clf_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["caller", "comparison", "metric", "value"])
        for comparison, m in clf_benchmarks.items():
            for metric_name, val in m.items():
                sval = f"{val:.6f}" if isinstance(val, float) else str(val)
                writer.writerow([args.caller_name, comparison, metric_name, sval])

    print(f"Cell-classification benchmarks written to: {clf_out}")
    for comp, m in clf_benchmarks.items():
        print(f"  {comp}: {m}")

    # NEW: Updated TSV header with celltype
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["caller", "ground_truth_cell", "caller_cell",
                 "celltype", "celltype_dna", "celltype_refset_rna", "celltype_real_rna", "metric", "value"])
        for row in long_rows:
            val = row["value"]
            if isinstance(val, float) and math.isnan(val):
                sval = "nan"
            else:
                sval = f"{val:.6f}" if isinstance(val, float) else str(val)
            writer.writerow([
                row["caller"],
                row["ground_truth_cell"],
                row["caller_cell"],
                row["celltype"],
                row["celltype_dna"],
                row["celltype_refset_rna"],
                row["celltype_real_rna"],
                row["metric"],
                sval,
            ])

    plot_path = args.plot
    if plot_path is None:
        base, _ = os.path.splitext(args.output)
        plot_path = base + ".boxplot.png"

    make_boxplot(long_rows, args.caller_name, plot_path)

    print(f"Evaluation complete for {args.caller_name}")
    print(f"Caller format: {caller_format}")
    print(f"Matched cells by order: {len(cell_pairs)}")
    print(f"Exome coverage fraction: {exome_fraction:.6f}" if not math.isnan(exome_fraction) else "Exome coverage fraction: nan")
    print(f"Metrics written to: {args.output}")
    print(f"Boxplot written to: {plot_path}")
    for gt_cell, pred_cell, celltype, n_pairs in summary_rows[:5]:
        print(f"  {gt_cell} <-> {pred_cell} ({celltype}): n_overlap_pairs={n_pairs}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s')
    main()
