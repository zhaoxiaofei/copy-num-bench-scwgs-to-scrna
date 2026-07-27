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
- Replaced the per-cell median bin normalization with a fixed, method-appropriate
  diploid baseline estimated once from the reference (diploid) cells when
  available (euploid-reference / two-pass normalization, after
  Song et al. 2025, Brief. Bioinform. 26(2):bbaf076, and
  Schmid et al. 2025, Nat. Commun. 16:8777).
- Added optional per-cell read-count labels to both CNV clustermaps. Read counts
  are loaded from a headered or headerless TSV/CSV/whitespace table and matched
  to cells using the same cellpath2id normalization as the benchmark.
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
- Each genomic bin is classified gain/loss/neutral relative to a FIXED,
  method-appropriate diploid baseline (0 for log-ratio / discrete callers, 2 for
  absolute copy number), NOT relative to the per-cell median. The per-cell median
  is a biased estimate of the diploid level in aneuploid cells, where the median
  sits on the dominant ALTERED state; see classify_pred_value for the references.
- When cells annotated as 'reference' (diploid) are available, the baseline is
  estimated once from those cells (euploid-reference / two-pass normalization,
  after Song et al. 2025, Brief. Bioinform. bbaf076, and Schmid et al. 2025,
  Nat. Commun. 16:8777) and reused for every cell; otherwise the fixed per-format
  baseline is used. See compute_reference_baseline / compute_metrics_for_cell.

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
import sklearn.metrics  # explicit: modern sklearn no longer auto-loads .metrics on `import sklearn`

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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
except ImportError:
    plt = None
    ListedColormap = None
    BoundaryNorm = None
    TwoSlopeNorm = None

try:
    import seaborn as sns
except ImportError:
    sns = None


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

    # ── Performance / subsampling options ──────────────────────────────────
    p.add_argument("--max-cells", type=int, default=100,
                   help="Maximum number of matched cells to load and evaluate. "
                        "Default: 100. Set to 0 to evaluate all cells.")
    p.add_argument("--subsample-seed", type=int, default=1,
                   help="Random seed for deterministic cell subsampling. Default: 1.")
    p.add_argument("--no-stratified-subsample", action="store_true",
                   help="Sample cells without preserving tumor/reference proportions.")
    p.add_argument("--skip-boxplot", action="store_true",
                   help="Skip the summary boxplot to save time.")
    p.add_argument("--skip-clustermap", action="store_true",
                   help="Skip both CNV clustermaps to save time.")

    # ── CNV clustermap options (NEW) ──────────────────────────────────────
    # Produces two dendrogram-like heatmaps side by side: one for the scWGS
    # Ginkgo-derived truth set, one for the scRNA-seq caller output. Plotting
    # style mirrors cnv_clustermap.py (discrete integer-CN colormap for absolute
    # copy number, continuous diverging colormap for log-ratio / discrete
    # callers). Skipped silently when seaborn or pandas is not installed.
    p.add_argument("--clustermap-prefix", default=None,
                   help="Prefix for the two CNV clustermap outputs. Defaults to "
                        "<output stem>.clustermap; truth and caller plots are "
                        "written as <prefix>.truth.{pdf,png} and "
                        "<prefix>.caller.{pdf,png}.")
    p.add_argument("--fai", default=None,
                   help="Chromosome sizes file (.fai / chrom.sizes). OPTIONAL: "
                        "when not given, chromosome sizes are inferred from "
                        "the furthest observed genomic position per chromosome "
                        "across the truth + caller interval lists, which is "
                        "enough to lay out fixed-size bins covering the data.")
    p.add_argument("--clustermap-bin-size", type=int, default=5_000_000,
                   help="Bin size in bp for the clustermaps. Default 5 Mb "
                        "(5000000) -> sub-chromosomal resolution. Set to 0 to "
                        "fall back to per-chromosome means (one column per "
                        "chromosome, no sub-chrom info). When >0 and --fai is "
                        "not given, chromosome sizes are inferred from the data.")
    p.add_argument("--clustermap-cmap", default='RdBu_r',
                   help="Base colormap for the clustermaps.")
    p.add_argument("--clustermap-truth-vmin", type=int, default=0,
                   help="Min CN for the truth clustermap colorbar.")
    p.add_argument("--clustermap-truth-vmax", type=int, default=6,
                   help="Max CN for the truth clustermap colorbar.")
    p.add_argument("--clustermap-truth-center", type=int, default=2,
                   help="Diploid CN for the truth clustermap (used to fill NaN).")
    p.add_argument("--clustermap-caller-vmin", type=float, default=None,
                   help="Min value for the caller clustermap colorbar (continuous "
                        "mode only). Defaults to the 1st percentile of the data.")
    p.add_argument("--clustermap-caller-vmax", type=float, default=None,
                   help="Max value for the caller clustermap colorbar (continuous "
                        "mode only). Defaults to the 99th percentile of the data.")
    p.add_argument("--clustermap-show-sample-labels", type=int, default=1,
                   help="Print cell names on the y-axis of the clustermaps.")
    p.add_argument(
        "--cell-read-counts", "--clustermap-read-counts",
        dest="cell_read_counts", default=None,
        help="Optional TSV/CSV/whitespace table containing one cell name and "
             "one read count per row. When supplied, clustermap y-axis labels "
             "are formatted as '<cell> | reads=<count>'. Headered and "
             "headerless files, including .gz files, are supported.")
    p.add_argument(
        "--read-count-cell-column", default=None,
        help="Cell-name column in --cell-read-counts, specified as a header "
             "name or zero-based column index. Default: auto-detect common "
             "cell/barcode column names, otherwise column 0.")
    p.add_argument(
        "--read-count-value-column", default=None,
        help="Read-count column in --cell-read-counts, specified as a header "
             "name or zero-based column index. Default: auto-detect common "
             "read-count column names, otherwise column 1.")
    p.add_argument(
        "--read-count-dna-value-column", default=None,
        help="DNA/scWGS read-count column in --cell-read-counts, specified as a header "
             "name or zero-based column index. When supplied alongside --cell-read-counts, "
             "DNA read counts are used for the truth (scWGS) clustermap labels while "
             "the RNA read counts (from --read-count-value-column) are used for the "
             "caller (scRNA) clustermap labels. Default: auto-detect common "
             "DNA/scWGS read-count column names (dna_reads, scwgs_reads, wgs_reads), "
             "otherwise None (same read counts used for both clustermaps).")
    p.add_argument(
        "--clustermap-read-label-format", default="{cell} | reads={reads:,}",
        help="Python format string for clustermap cell labels. Available fields: "
             "{cell} and integer {reads}. Default: '{cell} | reads={reads:,}'.")
    return p.parse_args()

# NEW: Load sample annotation (2-column TSV: cell name -> tumor/reference)
def load_sample_annotation(path):
    normal_celltypes = ['normal', 'reference', 'Normal', 'Reference']
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
                if cell_type in normal_celltypes: cell_type = 'reference'
                annot.append((cell_name, cell_type))
    return annot

def build_annot_dict(annot_list):
    """Return {cellpath2id(name): celltype} from load_sample_annotation output."""
    return {cellpath2id(name): ct for name, ct in annot_list}

def annotation_is_all_unknown(annot_list):
    """True if a sample annotation carries no real tumor/normal label.
    ORI_CELL_ANNOTATION is written by create_sample_annotation_1 as all-'unknown'
    exactly when the config declined to label cells (every celltype 'Unknown').
    In that case the author/config labeling is a placeholder and the DNA (Ginkgo)
    labeling should be used as the effective 'sample' labeling instead.
    load_sample_annotation only canonicalises the normal synonyms, so anything
    that is neither 'reference' nor 'tumor' (i.e. 'unknown') signals the deferral.
    """
    if not annot_list:
        return True
    return all(ct not in ("reference", "tumor") for _name, ct in annot_list)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


_CELL_COLUMN_ALIASES = {
    "cell", "cellid", "cell_id", "cellname", "cell_name", "sample",
    "sampleid", "sample_id", "barcode", "cellbarcode", "cell_barcode", "cb",
}
_READ_COUNT_COLUMN_ALIASES = {
    "reads", "read", "nreads", "n_reads", "readcount", "read_count",
    "numreads", "num_reads", "numberofreads", "number_of_reads",
    "totalreads", "total_reads", "rawreads", "raw_reads",
    "rna_reads", "rna_read",
}
_DNA_READ_COUNT_COLUMN_ALIASES = {
    "dna_reads", "dna_read", "scwgs_reads", "scwgs_read",
    "wgs_reads", "wgs_read",
}


def _normalise_header_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_column_spec(spec, header, fallback_index, aliases, column_kind):
    """Resolve a column name or zero-based integer string to an index."""
    if spec is not None:
        spec_text = str(spec).strip()
        try:
            index = int(spec_text)
        except ValueError:
            if header is None:
                raise ValueError(
                    f"--{column_kind}={spec!r} is a column name, but the read-count "
                    "table was detected as headerless. Use a zero-based index instead.")
            normalised = [_normalise_header_name(x) for x in header]
            target = _normalise_header_name(spec_text)
            if target not in normalised:
                raise ValueError(
                    f"Column {spec!r} was not found in the read-count header: {header}")
            return normalised.index(target)
        if index < 0:
            raise ValueError(f"Column index must be non-negative: {spec!r}")
        return index

    if header is not None:
        normalised = [_normalise_header_name(x) for x in header]
        for index, name in enumerate(normalised):
            if name in aliases:
                return index
    return fallback_index


def _parse_read_count(value):
    """Parse integer-like read counts, accepting commas and scientific notation."""
    text = str(value).strip().strip('"').strip("'").replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(round(number))


def _split_read_count_line(line, delimiter):
    if delimiter is None:
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def load_cell_read_counts(path, cell_column=None, value_column=None):
    """Load ``{normalised_cell_id: integer_read_count}`` from a small table.

    The table may be tab-, comma-, semicolon-, or whitespace-delimited and may
    have a header. Column names or zero-based indices can be provided explicitly.
    In auto mode, common names such as ``cell``, ``barcode``, ``reads``,
    ``n_reads``, and ``read_count`` are recognised. Headerless input defaults to
    the first two columns. Duplicate normalised cell IDs are warned about and the
    last row is retained.
    """
    with _open_text(path) as handle:
        lines = [line.strip() for line in handle
                 if line.strip() and not line.lstrip().startswith("#")]

    if not lines:
        raise ValueError(f"Read-count table is empty: {path}")

    first = lines[0]
    if "\t" in first:
        delimiter = "\t"
    elif "," in first:
        delimiter = ","
    elif ";" in first:
        delimiter = ";"
    else:
        delimiter = None

    rows = [_split_read_count_line(line, delimiter) for line in lines]
    if not rows or len(rows[0]) < 2:
        raise ValueError(
            f"Read-count table must contain at least two columns: {path}")

    first_normalised = [_normalise_header_name(x) for x in rows[0]]
    explicit_named_column = any(
        spec is not None and not str(spec).strip().lstrip("+").isdigit()
        for spec in (cell_column, value_column)
    )
    explicit_index_column = any(
        spec is not None and str(spec).strip().lstrip("+").isdigit()
        for spec in (cell_column, value_column)
    )
    recognised_header = (
        any(name in _CELL_COLUMN_ALIASES for name in first_normalised)
        or any(name in _READ_COUNT_COLUMN_ALIASES for name in first_normalised)
    )

    # A nonnumeric second field is also a useful header signal for conventional
    # two-column files. Explicit named columns necessarily require a header.
    second_field_is_numeric = _parse_read_count(rows[0][1]) is not None
    has_header = (
        explicit_named_column
        or recognised_header
        or (not explicit_index_column and not second_field_is_numeric)
    )
    header = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows

    cell_index = _parse_column_spec(
        cell_column, header, 0, _CELL_COLUMN_ALIASES, "read-count-cell-column")
    value_index = _parse_column_spec(
        value_column, header, 1, _READ_COUNT_COLUMN_ALIASES,
        "read-count-value-column")
    max_index = max(cell_index, value_index)

    counts = {}
    malformed = 0
    duplicates = 0
    for row_number, row in enumerate(data_rows, start=(2 if has_header else 1)):
        if len(row) <= max_index:
            malformed += 1
            logging.warning(
                "Skipping read-count row %d with %d fields; column %d is required.",
                row_number, len(row), max_index)
            continue
        raw_cell = str(row[cell_index]).strip().strip('"').strip("'")
        count = _parse_read_count(row[value_index])
        if not raw_cell or count is None:
            malformed += 1
            logging.warning(
                "Skipping invalid read-count row %d: cell=%r count=%r",
                row_number, raw_cell, row[value_index])
            continue
        cell_id = cellpath2id(raw_cell)
        if cell_id in counts:
            duplicates += 1
            logging.warning(
                "Duplicate read-count cell ID %r at row %d; replacing %d with %d.",
                cell_id, row_number, counts[cell_id], count)
        counts[cell_id] = count

    if not counts:
        raise ValueError(f"No valid per-cell read counts were loaded from {path}")

    logging.info(
        "Loaded read counts for %d cells from %s (header=%s, cell column=%d, "
        "read-count column=%d, malformed=%d, duplicates=%d).",
        len(counts), path, has_header, cell_index, value_index,
        malformed, duplicates)
    return counts


def format_clustermap_cell_label(cell_name, read_counts, label_format):
    """Return a display-only cell label with a formatted read count."""
    cell_text = str(cell_name)
    if not read_counts:
        return cell_text
    count = read_counts.get(cellpath2id(cell_text))
    if count is None:
        return f"{cell_text} | reads=NA"
    try:
        return label_format.format(cell=cell_text, reads=count)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(
            "Invalid --clustermap-read-label-format. Only {cell} and {reads} "
            f"are available: {label_format!r}") from exc


def normalise_chrom(chrom):
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        return chrom
    return "chr" + chrom


def interval_overlap(a_start, a_end, b_start, b_end):
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0, e - s)


def select_named_cells(names, max_cells=0, seed=1, annotation_dict=None,
                       allowed_ids=None):
    """Select a deterministic, optionally stratified subset of named cells.

    Returns ``(source_indices, selected_names)``. Cell IDs are normalized with
    ``cellpath2id``. When ``allowed_ids`` is provided, unmatched cells are
    discarded before sampling. Stratification uses annotation labels such as
    tumor/reference and approximately preserves their proportions.
    """
    candidates = []
    for idx, name in enumerate(names):
        cell_id = cellpath2id(name)
        if allowed_ids is not None and cell_id not in allowed_ids:
            continue
        label = (annotation_dict or {}).get(cell_id, "all")
        candidates.append((idx, name, label))

    if max_cells is None or max_cells <= 0 or len(candidates) <= max_cells:
        return [x[0] for x in candidates], [x[1] for x in candidates]

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for item in candidates:
        groups[item[2]].append(item)
    for group in groups.values():
        rng.shuffle(group)

    # Proportional fair selection. Each nonempty stratum is visited early, while
    # larger strata receive proportionally more of the final sample.
    taken = {label: 0 for label in groups}
    selected = []
    while len(selected) < max_cells:
        available = [label for label, group in groups.items()
                     if taken[label] < len(group)]
        if not available:
            break
        label = min(available, key=lambda x: taken[x] / len(groups[x]))
        selected.append(groups[label][taken[label]])
        taken[label] += 1

    # Preserve the original column order after random selection.
    selected.sort(key=lambda x: x[0])
    return [x[0] for x in selected], [x[1] for x in selected]


def load_ground_truth_wide(path, max_cells=0, seed=1, annotation_dict=None):
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

        all_cell_names = header[3:]
        selected_indices, cell_names = select_named_cells(
            all_cell_names, max_cells=max_cells, seed=seed,
            annotation_dict=annotation_dict)

        logging.info(
            "Ground-truth subsampling selected %d of %d cells (seed=%d).",
            len(cell_names), len(all_cell_names), seed)

        for row in reader:
            if len(row) < 4:
                continue

            chrom = normalise_chrom(row[0])
            # Sex chromosomes have a sex-dependent EUPLOID copy number (chrX/chrY
            # are CN 1 in males, chrX is CN 2 in females), so CN != 2 there is not
            # a somatic CNV and must not be scored as a gain/loss or counted as
            # non-diploid. Drop them here so every downstream ground-truth metric
            # (gain/loss/neutral accuracy AND "fraction of the scWGS genome
            # diploid") is restricted to autosomes -- matching the tumor/normal
            # classifier separate_tumor_normal_from_ginkgo_segcopy.py, which also
            # excludes them.
            if chrom in ("chrX", "chrY", "chr23", "chr24"):
                continue
            try:
                start = int(row[1])
                end = int(row[2])
            except ValueError:
                continue

            values = row[3:]
            for source_i, cell in zip(selected_indices, cell_names):
                if source_i >= len(values):
                    continue
                cn = safe_float(values[source_i])
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


def load_matrix_like_caller(path, caller_format, gene_positions, gt_cell_names,
                            scevan_count_mtx_annot_rdata_path=None):
    """
    Load matrix-like caller outputs into per-cell genomic features.

    Returns:
        caller_by_cell: dict[cell_label] -> list of (chrom, start, end, value)
        exome_fraction: float
        caller_cell_names: list[str]
    """
    caller_by_cell = defaultdict(list)
    selected_gt_ids = {cellpath2id(name) for name in gt_cell_names}

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

        all_caller_cell_names = cna_mtx.columns.tolist()
        _selected_indices, caller_cell_names = select_named_cells(
            all_caller_cell_names, allowed_ids=selected_gt_ids)
        cna_mtx = cna_mtx.loc[:, caller_cell_names]
        logging.info("Loading %d of %d SCEVAN cell columns.",
                     len(caller_cell_names), len(all_caller_cell_names))
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

            all_caller_cell_names = header[meta_cols:]
            selected_indices, caller_cell_names = select_named_cells(
                all_caller_cell_names, allowed_ids=selected_gt_ids)
            logging.info("Loading %d of %d copyKAT cell columns.",
                         len(caller_cell_names), len(all_caller_cell_names))

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
                for source_i, cell in zip(selected_indices, caller_cell_names):
                    if source_i >= len(vals):
                        continue
                    v = safe_float(vals[source_i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))

            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format == "infercna":
            # infercna output: gene x cell matrix with log2 ratios
            # Header: first column is empty/header, followed by cell names
            if len(header) < 2:
                raise ValueError("infercna matrix has too few columns.")
            all_caller_cell_names = header[1:]
            selected_indices, caller_cell_names = select_named_cells(
                all_caller_cell_names, allowed_ids=selected_gt_ids)
            logging.info("Loading %d of %d inferCNA cell columns.",
                         len(caller_cell_names), len(all_caller_cell_names))
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
                for source_i, cell in zip(selected_indices, caller_cell_names):
                    if source_i >= len(vals):
                        continue
                    v = safe_float(vals[source_i])
                    if v is None:
                        continue
                    caller_by_cell[cell].append((chrom, start, end, v))
            return dict(caller_by_cell), covered_intervals, caller_cell_names

        elif caller_format in ("infercnv", "generic_matrix"):
            # infercnv-like:
            # first column = gene/feature, remaining columns = cells
            if len(header) < 2:
                raise ValueError("Matrix-like caller result has too few columns.")

            all_caller_cell_names = header[1:]
            selected_indices, caller_cell_names = select_named_cells(
                all_caller_cell_names, allowed_ids=selected_gt_ids)
            logging.info("Loading %d of %d matrix cell columns.",
                         len(caller_cell_names), len(all_caller_cell_names))
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
                for source_i, cell in zip(selected_indices, caller_cell_names):
                    if source_i >= len(vals):
                        continue
                    v = safe_float(vals[source_i])
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
            all_caller_cell_names = header                  # all of header = cells
            selected_indices, caller_cell_names = select_named_cells(
                all_caller_cell_names, allowed_ids=selected_gt_ids)
            n_cells_all = len(all_caller_cell_names)
            logging.info("Loading %d of %d CaSpER cell columns.",
                         len(caller_cell_names), n_cells_all)
            covered_intervals = defaultdict(list)
            for row in reader:
                if len(row) < n_cells_all + 1:
                    continue
                feature = row[0]
                if feature not in gene_positions:
                    continue
                chrom, start, end = gene_positions[feature]
                covered_intervals[chrom].append((start, end))
                vals = row[1:n_cells_all + 1]
                for source_i, cell in zip(selected_indices, caller_cell_names):
                    v = safe_float(vals[source_i])
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
            # Pre-resolve gene -> coords once. Coverage is cell-independent, so
            # record every gene interval once rather than once per cell.
            gene_coords = [gene_positions.get(g) for g in gene_names]
            caller_cell_names = []
            covered_intervals = defaultdict(list)
            for coords in gene_coords:
                if coords is not None:
                    chrom, start, end = coords
                    covered_intervals[chrom].append((start, end))

            for row in reader:
                if len(row) < 2:
                    continue
                cell = row[0]
                if cellpath2id(cell) not in selected_gt_ids:
                    continue
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
            logging.info("Loaded %d selected Numbat cell rows.",
                         len(caller_cell_names))
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
            for arm in arm_keys:
                if arm in arm2pos:
                    chrom, start, end = arm2pos[arm]
                    covered_intervals[chrom].append((start, end))
            for row in reader:
                if len(row) < 2:
                    continue
                cell = _norm_cell(row[0])
                if cellpath2id(cell) not in selected_gt_ids:
                    continue
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


# ============================================================================
# Fixed diploid baseline for gain/loss classification
#
# REVISED: the neutral ("diploid") baseline against which each genomic bin is
# called gain / loss is now a FIXED, method-appropriate value instead of the
# per-cell median of the predicted bin values.
#
# Why the old per-cell-median normalization was wrong
# ---------------------------------------------------
# The old code set, for each cell, baseline = weighted_median(pred_values) and
# called a bin gain/loss relative to that per-cell median. The median estimates
# the diploid level ONLY when < 50% of that cell's genome is altered. For the
# highly aneuploid cells that matter most this is false and the baseline lands on
# the dominant ALTERED state:
#   * Song et al., Brief. Bioinform. 26(2):bbaf076 (2025) show that at high CNA
#     burden / tumor purity the gained/lost state is "incorrectly centered" as
#     the baseline (their CRC02 inferCNV/CaSpER case); their remedy is a two-pass
#     run that anchors the baseline on known-normal cells (in SCEVAN, the MEDIAN
#     OF THE REFERENCE CELLS is subtracted).
#   * Schmid et al., Nat. Commun. 16:8777 (2025) state every caller "has a
#     baseline value, usually 0", only losses below / gains above it are
#     biologically meaningful, and a wrongly assigned baseline is exactly what
#     drives partial-AUC below 0.5; they normalize against euploid REFERENCE
#     cells, never against each cell's own median.
#
# Here the caller output is already normalized against its euploid reference, so
# the diploid level is a fixed constant per value convention:
#   * log-ratio / centered callers (copyKat, inferCNV(expr), infercna, SCEVAN,
#     Numbat(expr)) and discrete -1/0/1 callers (CaSpER, CONICSmat, Numbat(CNV))
#       -> baseline 0
#   * absolute integer copy number (segment files, Ginkgo-style ground truth)
#       -> baseline 2
# These constants are also the FALLBACK; when reference (diploid) cells are
# available, compute_reference_baseline() estimates the baseline from them and it
# overrides the constant (see main()).
# ============================================================================

# Fixed neutral (diploid) baseline per caller format.
NEUTRAL_BASELINE = {
    "copykat":        0.0,
    "infercnv":       0.0,   # set to 1.0 if you feed RAW inferCNV obs (centered on 1)
    "infercna":       0.0,
    "generic_matrix": 0.0,
    "scevan_rdata":   0.0,
    "scevan_seg":     2.0,   # segment-style absolute copy number
    "numbat":         0.0,
    "casper":         0.0,   # discrete -1 / 0 / 1
    "conicsmat":      0.0,   # discrete -1 / 0 / 1
}

# Half-width of the neutral band around the baseline (same magnitudes the script
# used before, now applied around the FIXED/reference baseline instead of the
# per-cell median): log-ratio / discrete callers use +/-0.1 around the baseline;
# absolute CN uses +/-0.5 (i.e. the old 1.5 / 2.5 cut points around 2).
NEUTRAL_DELTA = {
    "scevan_seg": 0.5,
}
_DEFAULT_DELTA = 0.1


def neutral_baseline_for_format(caller_format):
    """Fixed diploid baseline for a caller (fallback for the per-cell median)."""
    return NEUTRAL_BASELINE.get(caller_format, 0.0)


def neutral_delta_for_format(caller_format):
    """Half-width of the neutral band around the baseline."""
    return NEUTRAL_DELTA.get(caller_format, _DEFAULT_DELTA)


def _safe_roc_auc(y_true, y_score, sample_weight=None):
    """roc_auc_score that returns NaN when only one class is present in y_true.

    Threshold-free and baseline-independent; this is the metric both Song et al.
    (2025) and Schmid et al. (2025) rely on for CNA-profile accuracy (alongside
    correlation). The guard prevents cells that contain only gains, or only
    losses, from raising ValueError.
    """
    if len(set(y_true)) < 2:
        return float("nan")
    try:
        return sklearn.metrics.roc_auc_score(
            y_true, y_score, sample_weight=sample_weight
        )
    except ValueError:
        return float("nan")


def classify_gt_cn(cn):
    if cn > 2:
        return "gain"
    if cn < 2:
        return "loss"
    return "neutral"


def classify_pred_value(value, caller_format, baseline=None, delta=None):
    """
    Classify a predicted bin into gain / loss / neutral relative to a FIXED,
    method-appropriate diploid baseline.

    The previous `med` argument (the per-cell weighted median) has been REMOVED:
    the median is a biased estimate of the diploid level in aneuploid cells, so
    genuine gains were scored as neutral and neutral regions as losses (Song
    et al. 2025, Brief. Bioinform. bbaf076 -- the CRC02 "incorrect centering";
    Schmid et al. 2025, Nat. Commun. 16:8777 -- baseline "usually 0", a wrong
    baseline drives partial-AUC < 0.5).

    The caller output is already normalized against its euploid reference, so the
    diploid level is a fixed constant (0 for log-ratio / discrete callers, 2 for
    absolute copy number). `baseline` / `delta` may be supplied explicitly (e.g.
    a baseline estimated once from the reference/diploid cells), otherwise the
    fixed per-format defaults are used.
    """
    if baseline is None:
        baseline = neutral_baseline_for_format(caller_format)
    if delta is None:
        delta = neutral_delta_for_format(caller_format)
    if value > baseline + delta:
        return "gain"
    if value < baseline - delta:
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

    # This is balanced accuracy instead of ROC AUC, so skip generating this
    '''
    auc = (
        sklearn.metrics.roc_auc_score(gt_labels, pred_labels)
        if len(set(gt_labels)) == 2
        else float("nan")
    )
    '''
    return {
        "n_cells":         len(valid),
        "tumor_precision": precision,
        "tumor_recall":    recall,
        "tumor_f1":        f1,
        "accuracy":        accuracy,
        # "roc_auc":         auc,
    }


def compute_aneuploidy_score(pred_intervals, baseline):
    """Per-cell aneuploidy score (MADD) from the scRNA-seq CNV profile.

    MADD = base-pair-weighted mean absolute deviation of the caller's CNV values
    from the diploid baseline x0:  sum_i w_i * |x_i - x0| / sum_i w_i, with
    w_i the length (bp) of segment i. Higher = more deviation from diploidy =
    more likely aneuploid. This is the standard continuous 'CNV score' /
    'aneuploidy score' used by CopyKAT / inferCNV / SCEVAN benchmarks, here
    bp-weighted for consistency with the rest of this script (weighted
    correlations, weighted-median reference baseline).

    `baseline` (x0) is the same diploid level used for gain/loss classification:
    the reference-derived weighted median when reference cells exist, else the
    fixed per-format baseline. Returns NaN for an empty profile.
    """
    num = 0.0
    den = 0.0
    for (_chrom, s, e, val) in pred_intervals:
        if val is None:
            continue
        w = max(1, e - s)
        num += w * abs(val - baseline)
        den += w
    return (num / den) if den > 0 else float("nan")


def compute_aneuploidy_auroc(scores, gt_types):
    """Threshold-agnostic AUROC of a CONTINUOUS per-cell aneuploidy score against
    the scWGS gold-standard labels.

    gt 'tumor' (aneuploid) = positive class, 'reference' (near-diploid) =
    negative class; cells whose gt label is neither are dropped, as are cells
    with a NaN score. Measures how well the scRNA-seq score *ranks* aneuploid
    cells above near-diploid ones (0.5 = random, ~0.75-0.95 typical for good
    scRNA-seq CNV callers). Distinct from compute_cell_classification_benchmark's
    'roc_auc', which scores HARD 0/1 predicted labels (= balanced accuracy).
    Returns NaN when fewer than 2 usable cells or only one class is present.
    """
    pairs = []
    for s, g in zip(scores, gt_types):
        if g not in ("tumor", "reference"):
            continue
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        pairs.append((s, g))
    if len(pairs) < 2:
        return float("nan")
    y_score = [s for s, _g in pairs]
    y_true  = [1 if g == "tumor" else 0 for _s, g in pairs]
    return _safe_roc_auc(y_true, y_score)


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

# Fast interval indexing and overlap collection. The original implementation
# rebuilt chromosome dictionaries, re-sorted all intervals, and materialized
# five-element overlap tuples separately for every cell.
def index_intervals_by_chrom(intervals):
    indexed = defaultdict(list)
    for chrom, start, end, value in intervals:
        indexed[chrom].append((start, end, value))
    for values in indexed.values():
        values.sort(key=lambda x: (x[0], x[1]))
    return dict(indexed)


def collect_overlap_arrays(gt_by_chrom, pred_by_chrom):
    """Return overlap lengths, GT values and prediction values as NumPy arrays."""
    weights = []
    gt_values = []
    pred_values = []

    for chrom, gt_list in gt_by_chrom.items():
        preds = pred_by_chrom.get(chrom)
        if not preds:
            continue

        j = 0
        for gstart, gend, gcn in gt_list:
            while j < len(preds) and preds[j][1] <= gstart:
                j += 1

            k = j
            while k < len(preds) and preds[k][0] < gend:
                pstart, pend, pval = preds[k]
                overlap = min(gend, pend) - max(gstart, pstart)
                if overlap > 0:
                    weights.append(overlap)
                    gt_values.append(gcn)
                    pred_values.append(pval)
                k += 1

    return (np.asarray(weights, dtype=np.float64),
            np.asarray(gt_values, dtype=np.float64),
            np.asarray(pred_values, dtype=np.float64))


def weighted_pearson_numpy(x, y, weights):
    """Fast weighted Pearson correlation using NumPy vector operations."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total_weight = weights.sum()
    if x.size < 2 or total_weight <= 0:
        return float("nan")

    mean_x = np.dot(weights, x) / total_weight
    mean_y = np.dot(weights, y) / total_weight
    dx = x - mean_x
    dy = y - mean_y
    covariance = np.dot(weights, dx * dy)
    variance_x = np.dot(weights, dx * dx)
    variance_y = np.dot(weights, dy * dy)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0 else float("nan")


def compute_reference_baseline(cell_pairs, caller_by_cell, ref_annot_dict, caller_format):
    """
    Estimate the diploid baseline from cells annotated 'reference' (diploid),
    pooling their predicted bin values and taking the base-pair-weighted median.

    This is the euploid-reference / two-pass normalization recommended by
    Song et al. 2025 (Brief. Bioinform. bbaf076; in SCEVAN the median of the
    reference cells is the baseline) and Schmid et al. 2025 (Nat. Commun.
    16:8777; normalize against euploid reference cells), and it replaces the old
    per-cell median (biased in aneuploid cells). Because it reads the diploid
    cells' own predicted values, it auto-adapts to each caller's value convention
    (~0 for log-ratio / discrete callers, ~2 for absolute copy number).

    `ref_annot_dict` is keyed by cellpath2id(<ground-truth cell name>). Returns
    None when no reference cells are available, in which case the fixed
    per-format baseline (neutral_baseline_for_format) is used instead.
    """
    ref_vals, ref_w = [], []
    for gt_cell, pred_cell in cell_pairs:
        if ref_annot_dict.get(cellpath2id(gt_cell)) != "reference":
            continue
        for (_chrom, s, e, val) in caller_by_cell.get(pred_cell, []):
            if val is None:
                continue
            ref_vals.append(val)
            ref_w.append(max(1, e - s))
    if not ref_vals:
        return None
    return weightedstats.weighted_median(ref_vals, weights=ref_w)


def compute_metrics_for_cell(gt_intervals, pred_intervals, caller_format, exome_fraction,
                             baseline=None, gt_index=None, pred_index=None):
    # ---- Pure scWGS (Ginkgo ground-truth) diploid fraction for THIS cell ----
    # Base-pair-weighted fraction of the cell's scWGS genome that Ginkgo calls
    # neutral (integer CN == 2, i.e. euploid / non-aneuploid). Depends ONLY on
    # the scWGS ground truth, not on the caller. main() masks it to the cells the
    # scRNA caller labelled normal/reference and emits it as
    # "Fraction of the scWGS genome diploid in scRNA-normal cells" (higher is
    # better: cells the scRNA caller calls normal should be diploid per scWGS).
    gt_total_bp = 0
    gt_diploid_bp = 0
    for (_c, _s, _e, _cn) in gt_intervals:
        w = _e - _s
        if w <= 0:
            continue
        gt_total_bp += w
        if classify_gt_cn(_cn) == "neutral":
            gt_diploid_bp += w
    scwgs_diploid_fraction = (gt_diploid_bp / gt_total_bp) if gt_total_bp > 0 else float("nan")

    if gt_index is None:
        gt_index = index_intervals_by_chrom(gt_intervals)
    if pred_index is None:
        pred_index = index_intervals_by_chrom(pred_intervals)
    weights, gt_vals, pred_vals = collect_overlap_arrays(gt_index, pred_index)

    if weights.size == 0:
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
            "_scwgs_diploid_fraction": scwgs_diploid_fraction,
            "n_overlap_pairs": 0,
        }

    # ---- REVISED NORMALIZATION --------------------------------------------
    # Old (wrong): pred_median = weightedstats.weighted_median(pred_vals, weights=weights)
    #              ... classify each bin relative to that PER-CELL median.
    # New: classify each bin relative to a FIXED, method-appropriate diploid
    #      baseline. The caller has already normalized its output against a
    #      euploid reference, so the diploid level is a constant (0 for
    #      log-ratio/discrete callers, 2 for absolute CN), not the cell's median.
    #      `baseline` is normally supplied by main() as a value estimated ONCE
    #      from the reference (diploid) cells (compute_reference_baseline), per
    #      the euploid-reference / two-pass normalization of Song et al. 2025 and
    #      Schmid et al. 2025; it falls back to the fixed per-format default.
    if baseline is None:
        baseline = neutral_baseline_for_format(caller_format)
    delta = neutral_delta_for_format(caller_format)
    # -----------------------------------------------------------------------

    # Encode loss=-1, neutral=0, gain=1 to avoid repeated Python string work.
    gt_cls = np.where(gt_vals > 2.0, 1, np.where(gt_vals < 2.0, -1, 0))
    pred_cls = np.where(pred_vals > baseline + delta, 1,
                        np.where(pred_vals < baseline - delta, -1, 0))

    # NOTE: RNA-seq-based callers do not generate integer copy numbers, so the
    # discrete gain/loss/neutral metrics below are inherently threshold-dependent
    # and should be read together with the THRESHOLD-FREE metrics computed further
    # down (weighted Pearson/Spearman + gain/loss ROC-AUC), which is what both
    # references foreground:
    # https://academic.oup.com/bib/article/26/2/bbaf076/8051529  (Song et al. 2025)
    #  - the word accuracy does not denote the one for classification
    #  - F1 score was for LOH detection and tumor-vs-normal classification of cells
    #  - no integer copy numbers were evaluated
    # https://www.nature.com/articles/s41467-025-62359-9  (Schmid et al. 2025)
    #  - accuracy was only used for tumor-vs-normal classification of cells
    #  - RNA-seq threshold was variable: gain/loss thresholds were chosen to
    #    MAXIMIZE multi-class F1 over the biologically meaningful range (gains
    #    above baseline, losses below) -- a per-method/global step, not per-bin
    #  - did not evaluate breakpoint detection in RNA-seq
    #  - no integer copy numbers were evaluated otherwise
    # The discrete metrics here use a single fixed neutral band around the
    # (reference-derived or fixed) baseline; for full fidelity to Schmid et al.,
    # optimize the gain/loss thresholds once globally.

    tp_gain = float(weights[(gt_cls == 1) & (pred_cls == 1)].sum())
    fp_gain = float(weights[(gt_cls != 1) & (pred_cls == 1)].sum())
    fn_gain = float(weights[(gt_cls == 1) & (pred_cls != 1)].sum())

    tp_loss = float(weights[(gt_cls == -1) & (pred_cls == -1)].sum())
    fp_loss = float(weights[(gt_cls != -1) & (pred_cls == -1)].sum())
    fn_loss = float(weights[(gt_cls == -1) & (pred_cls != -1)].sum())

    gain_precision, gain_recall, gain_fscore = compute_prf(tp_gain, fp_gain, fn_gain)
    loss_precision, loss_recall, loss_fscore = compute_prf(tp_loss, fp_loss, fn_loss)

    total_overlap = float(weights.sum())
    multiclass_acc = (float(weights[gt_cls == pred_cls].sum()) / total_overlap
                      if total_overlap > 0 else float("nan"))

    pearson_r = weighted_pearson_numpy(gt_vals, pred_vals, weights)
    spearman_r = weighted_pearson_numpy(rankdata(gt_vals), rankdata(pred_vals), weights)

    # Threshold-free, baseline-independent gain/loss AUCs (gain-vs-rest /
    # loss-vs-rest, as in Schmid et al. 2025), now guarded against single-class
    # cells so they no longer raise.
    gt_gain_bools = (gt_vals > 2.0).astype(np.int8)
    gt_loss_bools = (gt_vals < 2.0).astype(np.int8)
    gain_weighted_auc = _safe_roc_auc(gt_gain_bools, pred_vals, sample_weight=weights)
    loss_weighted_auc = _safe_roc_auc(gt_loss_bools, -pred_vals, sample_weight=weights)
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
        "_scwgs_diploid_fraction": scwgs_diploid_fraction,
        "n_overlap_pairs": int(weights.sum()),  # number of evaluated base pairs
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

# ============================================================================
# CNV clustermap helpers (adapted from cnv_clustermap.py)
# ============================================================================
# Plot two dendrogram-like CNV heatmaps side by side: one for the scWGS-based
# Ginkgo-derived truth set, one for the scRNA-seq-based caller output. Rows are
# hierarchically clustered, columns are kept in genomic order; integer copy
# numbers use a discrete ListedColormap + BoundaryNorm, while log-ratio /
# discrete -1/0/1 signals use a continuous diverging colormap (TwoSlopeNorm)
# centred at the diploid baseline. Both plots share the SAME binning scheme so
# the X axes line up.
# ============================================================================

CHROM_ORDER = [f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y']]


def chrom_sort_key(chrom):
    """Sort chr1..chr22, chrX, chrY; unknown chroms go last."""
    c = str(chrom).replace('chr', '')
    if c == 'X':
        return 23
    if c == 'Y':
        return 24
    try:
        return int(c)
    except Exception:
        return 99


def parse_chrom_sizes(fai_path):
    """Parse a .fai / chrom.sizes file into {chrom: length}."""
    sizes = {}
    with open(fai_path) as fh:
        for line in fh:
            toks = line.rstrip('\n').split('\t')
            if len(toks) >= 2:
                try:
                    sizes[toks[0]] = int(toks[1])
                except ValueError:
                    pass
    return sizes


def infer_chrom_sizes_from_intervals(*cell_intervals_dicts):
    """Infer chromosome sizes from per-cell interval lists.

    Takes one or more dicts of {cell_name: [(chrom, start, end, val), ...]} and
    returns {chrom: max_end} across all intervals in all dicts. Used to enable
    fixed-bin clustermaps without a --fai file: the inferred sizes are the
    furthest observed genomic position per chromosome, which is enough to lay
    out fixed-size bins covering the data.
    """
    sizes = {}
    for cell_intervals in cell_intervals_dicts:
        if not cell_intervals:
            continue
        for cell_name, intervals in cell_intervals.items():
            for chrom, start, end, _val in intervals:
                # end is exclusive in BED-like convention; treat it as the size.
                if chrom not in sizes or end > sizes[chrom]:
                    sizes[chrom] = end
    return sizes


def _fixed_bin_layout(bin_size, chrom_sizes):
    labels = []
    chrom_layout = {}
    offset = 0
    for chrom in sorted(chrom_sizes, key=chrom_sort_key):
        chrom_len = int(chrom_sizes[chrom])
        n_bins = max(1, (chrom_len + bin_size - 1) // bin_size)
        chrom_layout[chrom] = (offset, n_bins, chrom_len)
        for bin_idx in range(n_bins):
            start = bin_idx * bin_size
            end = min(start + bin_size, chrom_len)
            labels.append(f'{chrom}:{start}-{end}')
        offset += n_bins
    return labels, chrom_layout


def build_clustermap_matrix(cell_intervals, bin_size=0, chrom_sizes=None):
    """Build a cells x bins DataFrame without per-bin pandas filtering.

    The previous version built one DataFrame per cell and repeatedly filtered it
    for every chromosome and every fixed bin. This version allocates one NumPy
    row per cell and accumulates interval overlaps directly.
    """
    if pd is None:
        raise ImportError("pandas is required for clustermap plotting.")
    if not cell_intervals:
        return pd.DataFrame()

    cell_names = [name for name, intervals in cell_intervals.items() if intervals]
    if not cell_names:
        return pd.DataFrame()

    if bin_size == 0:
        labels = list(CHROM_ORDER)
        col_index = {chrom: i for i, chrom in enumerate(labels)}
        matrix = np.full((len(cell_names), len(labels)), np.nan, dtype=np.float32)

        for row_idx, cell_name in enumerate(cell_names):
            weighted_sum = np.zeros(len(labels), dtype=np.float64)
            total_weight = np.zeros(len(labels), dtype=np.float64)
            for chrom, start, end, value in cell_intervals[cell_name]:
                col = col_index.get(chrom)
                if col is None:
                    continue
                weight = max(1, end - start)
                weighted_sum[col] += weight * value
                total_weight[col] += weight
            valid = total_weight > 0
            matrix[row_idx, valid] = (weighted_sum[valid] / total_weight[valid]).astype(np.float32)

        return pd.DataFrame(matrix, index=cell_names, columns=labels)

    if chrom_sizes is None:
        raise ValueError("Fixed-bin clustermap requires chrom_sizes (from --fai).")

    labels, chrom_layout = _fixed_bin_layout(bin_size, chrom_sizes)
    matrix = np.full((len(cell_names), len(labels)), np.nan, dtype=np.float32)

    for row_idx, cell_name in enumerate(cell_names):
        weighted_sum = np.zeros(len(labels), dtype=np.float64)
        covered_bp = np.zeros(len(labels), dtype=np.float64)

        for chrom, start, end, value in cell_intervals[cell_name]:
            layout = chrom_layout.get(chrom)
            if layout is None or end <= start:
                continue
            offset, n_bins, chrom_len = layout
            clipped_start = max(0, min(int(start), chrom_len))
            clipped_end = max(0, min(int(end), chrom_len))
            if clipped_end <= clipped_start:
                continue

            first_bin = clipped_start // bin_size
            last_bin = min(n_bins - 1, (clipped_end - 1) // bin_size)
            for bin_idx in range(first_bin, last_bin + 1):
                bin_start = bin_idx * bin_size
                bin_end = min(bin_start + bin_size, chrom_len)
                overlap = min(clipped_end, bin_end) - max(clipped_start, bin_start)
                if overlap <= 0:
                    continue
                col = offset + bin_idx
                weighted_sum[col] += overlap * value
                covered_bp[col] += overlap

        valid = covered_bp > 0
        matrix[row_idx, valid] = (weighted_sum[valid] / covered_bp[valid]).astype(np.float32)

    return pd.DataFrame(matrix, index=cell_names, columns=labels)


def _linkage_max_depth(linkage):
    """Compute linkage-tree depth iteratively, avoiding recursive traversal."""
    n_leaves = linkage.shape[0] + 1
    depths = np.zeros(2 * n_leaves - 1, dtype=np.int32)
    for merge_idx, row in enumerate(linkage):
        left = int(row[0])
        right = int(row[1])
        depths[n_leaves + merge_idx] = max(depths[left], depths[right]) + 1
    return int(depths[-1])


def make_cnv_clustermap(mat, output_prefix, title='', discrete=True,
                        vmin=0, vmax=6, center=2, cmap='RdBu_r',
                        show_sample_labels=True, read_counts=None,
                        read_label_format="{cell} | reads={reads:,}"):
    """Plot a clustered CNV heatmap (dendrogram-like) from a cells x bins matrix.

    Adapted from cnv_clustermap.py:
      * rows hierarchically clustered (method='average', metric='euclidean');
      * columns kept in genomic order;
      * integer copy numbers shown with a discrete ListedColormap + BoundaryNorm
        (discrete=True), or continuous log-ratio values shown with a TwoSlopeNorm
        centred at `center` (discrete=False);
      * dashed vertical lines mark chromosome boundaries; chromosome names are
        drawn once, centred under each block;
      * horizontal colorbar above the heatmap;
      * dendrogram_ratio=(0.15, 0.055).

    Args:
        mat: pandas.DataFrame, rows = cells, columns = bins (labels like
             'chr1:0-1000000' for fixed-bin mode, or 'chr1' for per-chrom mode).
        output_prefix: write <prefix>.pdf and <prefix>.png.
        title: optional figure suptitle.
        discrete: True = integer CN colormap; False = continuous diverging.
        vmin, vmax, center: copy-number range and diploid centre.
        cmap: base colormap name.
        show_sample_labels: y-axis cell labels.
        read_counts: optional mapping keyed by normalised cell ID. When supplied,
                     each y-axis label includes the cell's read count.
        read_label_format: Python format string using {cell} and {reads}.
    """
    if plt is None or sns is None or pd is None:
        sys.stderr.write("WARNING: matplotlib/seaborn/pandas not installed; "
                         "skipping clustermap.\n")
        return
    if mat is None or mat.empty:
        sys.stderr.write(f"WARNING: empty matrix for clustermap {output_prefix}; "
                         "skipping.\n")
        return

    # Replace +/-inf with NaN, then round (discrete mode) / clip (both modes).
    mat_plot = mat.replace([np.inf, -np.inf], np.nan)
    if discrete:
        mat_plot = mat_plot.round().clip(lower=vmin, upper=vmax)
    else:
        mat_plot = mat_plot.clip(lower=vmin, upper=vmax)
    mat_plot = mat_plot.dropna(axis=1, how='all')
    if mat_plot.shape[1] == 0:
        sys.stderr.write(f"WARNING: no columns with data for clustermap "
                         f"{output_prefix}; skipping.\n")
        return
    mat_plot = mat_plot.fillna(center)

    # Read counts change display labels only; matrix values, linkage calculation,
    # subsampling, and cell matching are unaffected. Each clustermap uses the
    # appropriate read-count dict: DNA/scWGS counts for the truth clustermap,
    # RNA/scRNA counts for the caller clustermap.
    if read_counts:
        original_index = [str(cell) for cell in mat_plot.index]
        matched_read_counts = sum(
            cellpath2id(cell) in read_counts for cell in original_index)
        mat_plot = mat_plot.copy()
        mat_plot.index = [
            format_clustermap_cell_label(cell, read_counts, read_label_format)
            for cell in original_index
        ]
        sys.stderr.write(
            f"INFO: added read counts to {matched_read_counts}/{len(original_index)} "
            "clustermap cell labels; unmatched cells are labelled reads=NA.\n")

    if mat_plot.shape[0] < 2:
        sys.stderr.write(f"WARNING: fewer than 2 cells for clustermap "
                         f"{output_prefix}; skipping (clustering needs >=2 rows).\n")
        return

    if discrete:
        n_levels = int(vmax - vmin + 1)
        base = plt.get_cmap(cmap, n_levels)
        discrete_cmap = ListedColormap([base(i) for i in range(n_levels)])
        norm = BoundaryNorm(np.arange(vmin - 0.5, vmax + 1.5, 1.0),
                            discrete_cmap.N)
        cbar_kws = {
            'label':  'Copy number',
            'ticks':  np.arange(int(vmin), int(vmax) + 1),
            'spacing': 'proportional',
            'orientation': 'horizontal',
        }
        cmap_kwargs = {'cmap': discrete_cmap, 'norm': norm}
    else:
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        cbar_kws = {
            'label':  'CNV signal',
            'orientation': 'horizontal',
        }
        cmap_kwargs = {'cmap': cmap, 'norm': norm}

    figsize = (max(8, min(0.09 * mat_plot.shape[1] + 6, 12)),
               max(8, min(0.18 * mat_plot.shape[0] + 3, 12)))

    import fastcluster
    from scipy.cluster.hierarchy import leaves_list

    row_linkage = fastcluster.linkage(
        mat_plot.to_numpy(dtype=np.float64, copy=False),
        method="average",
        metric="euclidean",
    )

    tree_depth = _linkage_max_depth(row_linkage)
    recursion_limit = sys.getrecursionlimit()
    draw_row_dendrogram = tree_depth < max(50, recursion_limit - 100)
    plot_linkage = row_linkage

    sys.stderr.write(
        f"INFO: clustermap matrix={mat_plot.shape[0]} cells x "
        f"{mat_plot.shape[1]} bins; linkage depth={tree_depth}.\n")

    if not draw_row_dendrogram:
        # Preserve clustered row order but do not ask SciPy's recursive
        # dendrogram renderer to traverse a pathologically deep tree.
        mat_plot = mat_plot.iloc[leaves_list(row_linkage)]
        plot_linkage = None
        sys.stderr.write(
            "WARNING: linkage tree is too deep for the recursive dendrogram "
            "renderer; plotting clustered rows without the row dendrogram.\n")

    clustermap_kwargs = dict(
        data=mat_plot,
        row_linkage=plot_linkage,
        row_cluster=draw_row_dendrogram,
        col_cluster=False,
        figsize=figsize,
        cbar_kws=cbar_kws,
        xticklabels=False,
        yticklabels=show_sample_labels,
        dendrogram_ratio=(0.15, 0.055),
        **cmap_kwargs,
    )

    try:
        g = sns.clustermap(**clustermap_kwargs)
    except RecursionError:
        # A final defensive fallback for SciPy/Seaborn version-specific behavior.
        plt.close('all')
        sys.stderr.write(
            "WARNING: dendrogram rendering exceeded the recursion limit; "
            "retrying without the row dendrogram.\n")
        mat_plot = mat_plot.iloc[leaves_list(row_linkage)]
        clustermap_kwargs.update(
            data=mat_plot, row_linkage=None, row_cluster=False)
        g = sns.clustermap(**clustermap_kwargs)

    # Horizontal colorbar above the heatmap.
    g.ax_cbar.set_position([0.25, 0.98, 0.5, 0.01])
    g.ax_cbar.tick_params(axis='x', length=3)

    ax = g.ax_heatmap
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Chromosome dividers + centred chromosome labels (no per-bin x labels).
    col_chroms = [str(c).split(':')[0] for c in mat_plot.columns]
    prev = None
    for i, c in enumerate(col_chroms):
        if prev is not None and c != prev:
            ax.axvline(i, color='black', linestyle='--', linewidth=0.8)
        prev = c
    pos = {}
    for i, c in enumerate(col_chroms):
        pos.setdefault(c, []).append(i)
    centers, labels = [], []
    for c in CHROM_ORDER:
        if c in pos:
            centers.append((pos[c][0] + pos[c][-1] + 1) / 2.0)
            labels.append(c.replace('chr', ''))
    if centers:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.tick_params(axis='x', length=0)

    if show_sample_labels:
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)

    if title:
        g.fig.suptitle(title, y=1.02)

    out_dir = os.path.dirname(os.path.abspath(output_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    #g.savefig(output_prefix + '.pdf', bbox_inches='tight')
    g.savefig(output_prefix + '.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    sys.stderr.write(f'Wrote {output_prefix}.png\n')
    #sys.stderr.write(f'Wrote {output_prefix}.pdf and {output_prefix}.png\n')


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

    rna_read_counts = None
    dna_read_counts = None
    if args.cell_read_counts:
        logging.info("Started loading per-cell RNA read counts for clustermap labels")
        rna_read_counts = load_cell_read_counts(
            args.cell_read_counts,
            cell_column=args.read_count_cell_column,
            value_column=args.read_count_value_column,
        )
        # Also load DNA (scWGS) read counts for the truth clustermap if a
        # DNA column is specified or can be auto-detected.
        if args.read_count_dna_value_column is not None:
            logging.info("Started loading per-cell DNA read counts for truth clustermap labels")
            dna_read_counts = load_cell_read_counts(
                args.cell_read_counts,
                cell_column=args.read_count_cell_column,
                value_column=args.read_count_dna_value_column,
            )
        else:
            # Auto-detect: try to find a DNA/scWGS column in the same file.
            # If the file has a recognised DNA column header, load it;
            # otherwise fall back to using RNA counts for both clustermaps.
            with _open_text(args.cell_read_counts) as handle:
                first_lines = [line.strip() for line in handle
                               if line.strip() and not line.lstrip().startswith("#")]
            if first_lines and "\t" in first_lines[0]:
                header_fields = first_lines[0].split("\t")
                normalised = [_normalise_header_name(x) for x in header_fields]
                for name in normalised:
                    if name in _DNA_READ_COUNT_COLUMN_ALIASES:
                        dna_col = header_fields[normalised.index(name)]
                        logging.info("Auto-detected DNA read-count column '%s'; "
                                     "loading DNA counts for truth clustermap", dna_col)
                        dna_read_counts = load_cell_read_counts(
                            args.cell_read_counts,
                            cell_column=args.read_count_cell_column,
                            value_column=dna_col,
                        )
                        break
            if dna_read_counts is None:
                logging.info("No DNA read-count column found; using RNA counts for "
                             "both truth and caller clustermaps")
                dna_read_counts = rna_read_counts

    # Load annotations first so ground-truth subsampling can preserve the
    # tumor/reference proportions instead of loading all ~3000 cell columns.
    logging.info('Started loading sample annotation')
    sample_annot     = load_sample_annotation(args.sample_annotation)
    dna_annot        = load_sample_annotation(args.dna_annotation)
    rna_refset_annot = load_sample_annotation(args.rna_annotation)
    rna_real_annot   = load_sample_annotation(args.rna_annotation + '.maybe_zero_refs')

    # If the author/config 'sample' annotation is a placeholder (all-unknown, i.e.
    # the config used 'Unknown' for every cell), defer to the DNA (Ginkgo) labeling
    # as the effective sample labeling.
    SAMPLE_IS_UNKNOWN = annotation_is_all_unknown(sample_annot)
    if SAMPLE_IS_UNKNOWN:
        logging.info(
            "Sample annotation is all-unknown (config celltypes were 'Unknown'); "
            "using the DNA (Ginkgo) annotation as the effective 'sample' labeling.")
        sample_annot = dna_annot

    # Keyed by normalised cell ID
    sample_annot_dict = build_annot_dict(sample_annot)
    dna_annot_dict    = build_annot_dict(dna_annot)
    rna_refset_annot_dict = build_annot_dict(rna_refset_annot)
    rna_real_annot_dict   = build_annot_dict(rna_real_annot)

    logging.info('Started loading and subsampling ground truth')
    sampling_annotation = (None if args.no_stratified_subsample
                           else sample_annot_dict)
    gt_by_cell, gt_cell_names = load_ground_truth_wide(
        args.ground_truth, max_cells=args.max_cells,
        seed=args.subsample_seed, annotation_dict=sampling_annotation)

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

    logging.info(
        "Loaded %d sampled ground-truth cells, %d caller cells, %d common IDs.",
        len(gt_cell_names), len(caller_cell_names), len(common_cells))

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

    logging.info("Matched %d cells after subsampling.", len(cell_pairs))

    # Build chromosome indexes once. Previously these dictionaries were rebuilt
    # and re-sorted inside compute_metrics_for_cell for every cell.
    logging.info("Indexing intervals for fast overlap evaluation")
    gt_interval_indexes = {
        gt_cell: index_intervals_by_chrom(gt_by_cell.get(gt_cell, []))
        for gt_cell, _pred_cell in cell_pairs
    }
    pred_interval_indexes = {
        pred_cell: index_intervals_by_chrom(caller_by_cell.get(pred_cell, []))
        for _gt_cell, pred_cell in cell_pairs
    }

    # REVISED NORMALIZATION: estimate the diploid baseline ONCE from the
    # reference (diploid) cells and reuse it for every cell, instead of taking
    # each cell's own median (which is biased for aneuploid cells). Falls back to
    # the fixed per-format baseline when no reference cells are present. The
    # 'reference' cells here are taken from the author-provided sample annotation
    # (the ground-truth tumor/reference labelling); switch to dna_annot_dict if
    # you prefer the DNA-derived labelling. See Song et al. 2025 (Brief.
    # Bioinform. bbaf076) and Schmid et al. 2025 (Nat. Commun. 16:8777).
    ref_baseline = compute_reference_baseline(
        cell_pairs, caller_by_cell, sample_annot_dict, caller_format)
    if ref_baseline is None:
        logging.info(
            "No 'reference' cells available for baseline estimation; using fixed "
            f"per-format diploid baseline {neutral_baseline_for_format(caller_format)} "
            f"for caller format '{caller_format}'.")
    else:
        logging.info(
            f"Reference-derived diploid baseline = {ref_baseline:.6f} "
            f"(fixed per-format default = {neutral_baseline_for_format(caller_format)}; "
            f"neutral band half-width = {neutral_delta_for_format(caller_format)}).")

    long_rows = []
    summary_rows = []

    # ── collect per-cell annotation tuples for the classification benchmarks ──
    cell_ct_sample, cell_ct_dna, cell_ct_rna, cell_ct_real_rna = [], [], [], []

    # Per-cell continuous aneuploidy score (MADD) from the scRNA-seq CNV profile,
    # collected in the SAME cell order as cell_ct_dna so the two line up for the
    # threshold-agnostic AUROC (scWGS labels vs RNA score) computed after the loop.
    madd_scores = []

    for gt_cell, pred_cell in cell_pairs:
        gt_id   = cellpath2id(gt_cell)
        pred_id = cellpath2id(pred_cell)

        celltype_sample = sample_annot_dict.get(gt_id, "unknown")
        celltype_dna    = dna_annot_dict.get(gt_id, "unknown")
        celltype_refset_rna = rna_refset_annot_dict.get(pred_id, "unknown")
        celltype_real_rna = rna_real_annot_dict.get(pred_id, "unknown")

        cell_ct_sample.append(celltype_sample)
        cell_ct_dna.append(celltype_dna)
        cell_ct_rna.append(celltype_refset_rna)
        cell_ct_real_rna.append(celltype_real_rna)

        logging.info(f"Processing {gt_cell} <-> {pred_cell} | "
                     f"sample={celltype_sample}  dna={celltype_dna}  rna={celltype_refset_rna}  real_rna={celltype_real_rna}")

        gt_intervals   = gt_by_cell.get(gt_cell, [])
        pred_intervals = caller_by_cell.get(pred_cell, [])

        # Continuous aneuploidy score (MADD) for this cell, using the same diploid
        # baseline as the gain/loss classification (reference-derived, else fixed).
        madd_baseline = (ref_baseline if ref_baseline is not None
                         else neutral_baseline_for_format(caller_format))
        madd_scores.append(compute_aneuploidy_score(pred_intervals, madd_baseline))

        metrics = compute_metrics_for_cell(
            gt_intervals=gt_intervals,
            pred_intervals=pred_intervals,
            caller_format=caller_format,
            exome_fraction=exome_fraction,
            baseline=ref_baseline,   # reference-derived (or None -> fixed per-format)
            gt_index=gt_interval_indexes.get(gt_cell),
            pred_index=pred_interval_indexes.get(pred_cell),
        )
        metrics["Fraction of the cells with inferred copy numbers"] = (
            len(common_cells) / float(len(cell_to_gt_name))
        )
        # Percent of the scWGS genome that is diploid, reported ONLY for cells the
        # scRNA caller labelled normal/reference (celltype_real_rna == 'reference');
        # NaN for every other cell so the heatmap's per-(dataset,method) mean is
        # taken over the scRNA-normal cells only.
        scwgs_diploid_fraction = metrics.pop("_scwgs_diploid_fraction", float("nan"))
        metrics["Fraction of the scWGS genome diploid in scRNA-normal cells"] = (
            scwgs_diploid_fraction if celltype_real_rna == "reference" else float("nan")
        )
        summary_rows.append((gt_cell, pred_cell, celltype_sample, metrics.get("n_overlap_pairs", 0)))

        for metric_name, metric_value in metrics.items():
            if metric_name == "n_overlap_pairs" or metric_name.startswith("_"):
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
        # dna_vs_sample is meaningless when sample==dna (all-unknown config).
        "dna_vs_sample": ({} if SAMPLE_IS_UNKNOWN
                          else compute_cell_classification_benchmark(cell_ct_dna, cell_ct_sample)),
    }

    # Threshold-agnostic tumor-vs-normal AUROC from the CONTINUOUS scRNA-seq
    # aneuploidy score (MADD), scored against the scWGS (Ginkgo) gold-standard
    # labels. This answers "can the RNA-seq CNV profile rank aneuploid (tumor)
    # cells above near-diploid (reference) cells?" and is reported separately from
    # rna_vs_dna's hard-label 'roc_auc'. n_cells counts cells usable for the AUROC
    # (labelled tumor/reference with a non-NaN score).
    _auroc_n = sum(
        1 for s, g in zip(madd_scores, cell_ct_dna)
        if g in ("tumor", "reference")
        and not (s is None or (isinstance(s, float) and math.isnan(s)))
    )
    clf_benchmarks["rna_score_vs_dna"] = {
        "n_cells":                _auroc_n,
        "aneuploidy_score_auroc": compute_aneuploidy_auroc(madd_scores, cell_ct_dna),
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

    if args.skip_boxplot:
        logging.info("Skipping boxplot (--skip-boxplot).")
    else:
        make_boxplot(long_rows, args.caller_name, plot_path)

    # ── NEW: Two CNV clustermaps (truth + caller) ────────────────────────
    # Adapted from cnv_clustermap.py: rows hierarchically clustered, columns
    # kept in genomic order; discrete integer-CN colormap for the Ginkgo truth
    # set (and for absolute-CN callers such as scevan_seg), continuous diverging
    # colormap (TwoSlopeNorm centred at the diploid baseline) for log-ratio /
    # discrete callers. Both plots use the SAME binning scheme so the X axes
    # line up; cells are taken from `cell_pairs` and labelled with cellpath2id
    # so the y-axis labels match across the two plots.
    if args.skip_clustermap:
        logging.info("Skipping CNV clustermaps (--skip-clustermap).")
    elif sns is None or pd is None or plt is None:
        sys.stderr.write(
            "WARNING: seaborn/pandas/matplotlib not installed; skipping CNV "
            "clustermaps.\n")
    else:
        clustermap_prefix = args.clustermap_prefix
        if clustermap_prefix is None:
            base, _ = os.path.splitext(args.output)
            clustermap_prefix = base + ".clustermap"

        chrom_sizes = None
        bin_size = args.clustermap_bin_size
        if bin_size > 0:
            # Fixed-bin mode: resolve chrom sizes from --fai, or infer from data.
            if args.fai:
                chrom_sizes = {c: s for c, s in parse_chrom_sizes(args.fai).items()
                               if c in CHROM_ORDER}
            else:
                # Auto-infer from the union of truth + caller intervals so the
                # script works out-of-the-box without a .fai file. The inferred
                # size is the furthest observed genomic position per chromosome,
                # which is sufficient to lay out fixed-size bins covering the data.
                inferred = infer_chrom_sizes_from_intervals(gt_by_cell, caller_by_cell)
                chrom_sizes = {c: s for c, s in inferred.items() if c in CHROM_ORDER}
                if chrom_sizes:
                    sys.stderr.write(
                        "INFO: --fai not given; inferred chromosome sizes from "
                        "the data (max end per chrom across truth + caller "
                        f"intervals): "
                        f"{dict(sorted(chrom_sizes.items(), key=lambda kv: chrom_sort_key(kv[0])))}\n")
            if not chrom_sizes:
                sys.stderr.write(
                    "WARNING: could not resolve chromosome sizes for fixed-bin "
                    "clustermap; falling back to per-chromosome means.\n")
                bin_size = 0
        # else: per-chromosome means (no --fai needed)

        # Use the matched cells (cell_pairs), keyed by cellpath2id so the
        # y-axis labels match across the two plots.
        gt_cells_for_clustermap = {
            cellpath2id(gt_cell): gt_by_cell.get(gt_cell, [])
            for gt_cell, _pred_cell in cell_pairs
        }
        caller_cells_for_clustermap = {
            cellpath2id(pred_cell): caller_by_cell.get(pred_cell, [])
            for _gt_cell, pred_cell in cell_pairs
        }

        gt_mat = build_clustermap_matrix(
            gt_cells_for_clustermap, bin_size=bin_size, chrom_sizes=chrom_sizes)
        caller_mat = build_clustermap_matrix(
            caller_cells_for_clustermap, bin_size=bin_size,
            chrom_sizes=chrom_sizes)

        show_labels = bool(args.clustermap_show_sample_labels)

        # Truth clustermap: always integer discrete colormap (Ginkgo integer CN).
        truth_out = clustermap_prefix + ".truth"
        make_cnv_clustermap(
            gt_mat, truth_out,
            title=f"CNV clustermap (truth, Ginkgo/scWGS): {args.caller_name}",
            discrete=True,
            vmin=args.clustermap_truth_vmin,
            vmax=args.clustermap_truth_vmax,
            center=args.clustermap_truth_center,
            cmap=args.clustermap_cmap,
            show_sample_labels=show_labels,
            read_counts=dna_read_counts,
            read_label_format=args.clustermap_read_label_format,
        )
        print(f"Truth clustermap written to: {truth_out}.png")

        # Caller clustermap: integer discrete for absolute-CN formats
        # (neutral_baseline_for_format == 2.0, e.g. scevan_seg), continuous
        # diverging for log-ratio / discrete -1/0/1 formats (baseline 0.0).
        caller_baseline = (ref_baseline if ref_baseline is not None
                           else neutral_baseline_for_format(caller_format))
        is_absolute_cn = (neutral_baseline_for_format(caller_format) == 2.0)
        caller_out = clustermap_prefix + ".caller"
        if is_absolute_cn:
            make_cnv_clustermap(
                caller_mat, caller_out,
                title=f"CNV clustermap (caller, {args.caller_name})",
                discrete=True,
                vmin=args.clustermap_truth_vmin,
                vmax=args.clustermap_truth_vmax,
                center=args.clustermap_truth_center,
                cmap=args.clustermap_cmap,
                show_sample_labels=show_labels,
                read_counts=rna_read_counts,
                read_label_format=args.clustermap_read_label_format,
            )
        else:
            # Auto-derive vmin/vmax from data percentiles if not provided.
            if caller_mat.empty:
                caller_vmin = caller_baseline - 1.5
                caller_vmax = caller_baseline + 1.5
            else:
                flat = caller_mat.to_numpy().flatten()
                flat = flat[~np.isnan(flat)]
                if flat.size == 0:
                    caller_vmin = caller_baseline - 1.5
                    caller_vmax = caller_baseline + 1.5
                else:
                    caller_vmin = (args.clustermap_caller_vmin
                                   if args.clustermap_caller_vmin is not None
                                   else float(np.nanpercentile(flat, 1)))
                    caller_vmax = (args.clustermap_caller_vmax
                                   if args.clustermap_caller_vmax is not None
                                   else float(np.nanpercentile(flat, 99)))
            # Guard: ensure vmin < baseline < vmax for TwoSlopeNorm.
            if caller_vmin >= caller_baseline:
                caller_vmin = caller_baseline - 1.5
            if caller_vmax <= caller_baseline:
                caller_vmax = caller_baseline + 1.5
            make_cnv_clustermap(
                caller_mat, caller_out,
                title=f"CNV clustermap (caller, {args.caller_name})",
                discrete=False,
                vmin=caller_vmin,
                vmax=caller_vmax,
                center=caller_baseline,
                cmap=args.clustermap_cmap,
                show_sample_labels=show_labels,
                read_counts=rna_read_counts,
                read_label_format=args.clustermap_read_label_format,
            )
        print(f"Caller clustermap written to: {caller_out}.png")

    print(f"Evaluation complete for {args.caller_name}")
    print(f"Caller format: {caller_format}")
    print(f"Matched cells by order: {len(cell_pairs)}")
    print(f"Exome coverage fraction: {exome_fraction:.6f}" if not math.isnan(exome_fraction) else "Exome coverage fraction: nan")
    print(f"Metrics written to: {args.output}")
    if not args.skip_boxplot:
        print(f"Boxplot written to: {plot_path}")
    for gt_cell, pred_cell, celltype, n_pairs in summary_rows[:5]:
        print(f"  {gt_cell} <-> {pred_cell} ({celltype}): n_overlap_pairs={n_pairs}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s')
    main()


