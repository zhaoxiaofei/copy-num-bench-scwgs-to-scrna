#!/usr/bin/env python3
"""
compute_dataset_metrics.py

Compute per-dataset descriptive / confounder metrics for the
copy-num-bench-scwgs-to-scrna benchmark.

For each dataset (= one YAML config) it produces five metrics:

  1. Tumor purity        -> fraction of malignant cells (from config labels)
  2. Sequencing depth    -> reads / cell and mean coverage, per modality (from BAMs)
  3. CNV burden + event  -> fraction-of-genome-altered and segment-size
     size distribution      distribution (from the Ginkgo SegCopy ground truth)
  4. Tumor ploidy        -> mean integer copy number over tumor cells (Ginkgo)
  5. Number of cells     -> scRNA cell count (from config; optional post-QC count)

Design notes
------------
* Two metrics (purity, scRNA cell count) need only the merged config.
* Two metrics (CNV burden / event sizes, ploidy) are read from the Ginkgo
  `SegCopy` matrix that the pipeline already produces as ground truth.
* One metric (depth) shells out to `samtools idxstats` (fast, needs .bai)
  and optionally `samtools depth` for true mean coverage.

The script runs either standalone (argparse) or from a Snakemake `script:`
directive (it auto-detects the injected `snakemake` object).

Outputs (written to --out-dir):
  dataset_metrics.tsv   one summary row for the dataset (the headline table)
  per_cell_metrics.tsv  one row per cell (depth, ploidy, CN burden)
  cnv_event_sizes.tsv   one row per CNV segment (cell, chrom, size_bp, cn, class)
"""

import argparse
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #
def load_config(config_paths):
    """Merge one or more YAML files (template + dataset config), last wins."""
    merged = {}
    for p in config_paths:
        with open(p) as fh:
            d = yaml.safe_load(fh) or {}
        merged.update(d)
    return merged


def parse_cells(config):
    """Return a DataFrame of cells from cellname_celltype_scWGS_scRNA_tup_list.

    Each entry is [cell_id, 'Tumor'/'Normal', scWGS_bam, scRNA_bam].
    The placeholder string entry ('etc.') and malformed rows are dropped.
    """
    rows = []
    for entry in config.get("cellname_celltype_scWGS_scRNA_tup_list", []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            continue
        cell_id, celltype, scwgs_bam, scrna_bam = entry[:4]
        rows.append(
            {
                "cell_id": str(cell_id),
                "celltype": str(celltype).strip().capitalize(),  # Tumor / Normal
                "scwgs_bam": str(scwgs_bam) if scwgs_bam else "",
                "scrna_bam": str(scrna_bam) if scrna_bam else "",
            }
        )
    print(f'Parsed {rows}\nfrom the config\n{config}.')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Metric 1 + 5: purity and cell counts (config only)
# --------------------------------------------------------------------------- #
def purity_and_counts(cells):
    is_tumor = cells["celltype"].str.lower().eq("tumor")
    is_normal = cells["celltype"].str.lower().eq("normal")
    n_tumor = int(is_tumor.sum())
    n_normal = int(is_normal.sum())
    n_total = n_tumor + n_normal
    purity = n_tumor / n_total if n_total else np.nan

    has_scrna = cells["scrna_bam"].str.len().gt(0)
    n_scrna = int(has_scrna.sum())
    n_scrna_tumor = int((has_scrna & is_tumor).sum())

    return {
        "n_cells_total": n_total,
        "n_tumor_cells": n_tumor,
        "n_normal_cells": n_normal,
        "tumor_purity_cellfraction": round(purity, 4) if n_total else np.nan,
        "n_cells_scRNA": n_scrna,
        "n_cells_scRNA_tumor": n_scrna_tumor,
    }


# --------------------------------------------------------------------------- #
# Metric 2: sequencing depth / reads per cell (BAMs)
# --------------------------------------------------------------------------- #
def _resolve(prefix, bam):
    if not bam:
        return ""
    return bam if os.path.isabs(bam) else os.path.join(prefix, bam)


def bam_mapped_reads(bam_path, samtools="samtools"):
    """Primary, mapped, non-dup, non-supplementary reads via idxstats (fast)."""
    if not bam_path or not os.path.exists(bam_path):
        return np.nan
    try:
        out = subprocess.run(
            [samtools, "idxstats", bam_path],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return np.nan
    mapped = 0
    for line in out.strip().splitlines():
        f = line.split("\t")
        if len(f) >= 3 and f[0] != "*":
            mapped += int(f[2])
    return mapped


def bam_mean_depth(bam_path, samtools="samtools"):
    """Mean coverage over covered positions (samtools depth). Slow; optional."""
    if not bam_path or not os.path.exists(bam_path):
        return np.nan
    try:
        proc = subprocess.run(
            f"{samtools} depth -a {bam_path} "
            "| awk '{s+=$3; n++} END{if(n>0) print s/n; else print 0}'",
            shell=True, check=True, capture_output=True, text=True,
        )
        return float(proc.stdout.strip() or "nan")
    except (subprocess.CalledProcessError, ValueError):
        return np.nan


def depth_per_cell(cells, prefix, samtools="samtools", with_coverage=False):
    recs = []
    for _, c in cells.iterrows():
        wgs = _resolve(prefix, c["scwgs_bam"])
        rna = _resolve(prefix, c["scrna_bam"])
        rec = {
            "cell_id": c["cell_id"],
            "scWGS_reads": bam_mapped_reads(wgs, samtools),
            "scRNA_reads": bam_mapped_reads(rna, samtools),
        }
        if with_coverage:
            rec["scWGS_mean_depth"] = bam_mean_depth(wgs, samtools)
        recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Metrics 3 + 4: CNV burden, event-size distribution, ploidy (Ginkgo SegCopy)
# --------------------------------------------------------------------------- #
def load_segcopy(ginkgo_dir, segcopy_name="SegCopy"):
    """Ginkgo SegCopy: columns CHR, START, END, then one int-CN column per cell."""
    path = os.path.join(ginkgo_dir, segcopy_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ginkgo SegCopy not found at {path}. "
            "Point --ginkgo-dir at the Ginkgo output for this dataset."
        )
    df = pd.read_csv(path, sep="\t")
    # Normalise the first three column names (Ginkgo uses CHR/START/END).
    df = df.rename(columns={df.columns[0]: "CHR",
                            df.columns[1]: "START",
                            df.columns[2]: "END"})
    return df


def cnv_metrics_from_segcopy(segcopy, tumor_cells, focal_max_bp=3_000_000):
    """Per-cell ploidy, fraction-of-genome-altered, and CNV segments.

    Baseline (neutral) copy number per cell = its modal integer CN.
    Altered bin = CN != baseline. FGA weights bins by genomic length.
    Events = maximal runs of identical altered CN within a chromosome.
    """
    bin_len = (segcopy["END"] - segcopy["START"]).to_numpy()
    genome_len = float(bin_len.sum())
    chrom = segcopy["CHR"].to_numpy()
    start = segcopy["START"].to_numpy()
    end = segcopy["END"].to_numpy()

    cell_cols = [c for c in segcopy.columns if c not in ("CHR", "START", "END")]
    tumor_set = set(tumor_cells)

    per_cell = []
    events = []
    for col in cell_cols:
        cn = segcopy[col].to_numpy()
        if np.all(np.isnan(cn)):
            continue
        baseline = Counter(cn[~np.isnan(cn)]).most_common(1)[0][0]
        altered = cn != baseline
        fga = float(bin_len[altered].sum() / genome_len) if genome_len else np.nan
        ploidy = float(np.nanmean(cn))
        per_cell.append({
            "cell_id": col,
            "is_tumor": col in tumor_set,
            "ploidy_meanCN": round(ploidy, 4),
            "baseline_CN": int(baseline),
            "frac_genome_altered": round(fga, 4),
        })

        # run-length encode events within each chromosome
        for ch in pd.unique(chrom):
            idx = np.where(chrom == ch)[0]
            if idx.size == 0:
                continue
            cnch = cn[idx]
            seg_start = 0
            for i in range(1, len(idx) + 1):
                if i == len(idx) or cnch[i] != cnch[seg_start]:
                    cn_val = cnch[seg_start]
                    if not np.isnan(cn_val) and cn_val != baseline:
                        s_bp = int(start[idx[seg_start]])
                        e_bp = int(end[idx[i - 1]])
                        size = e_bp - s_bp
                        events.append({
                            "cell_id": col,
                            "is_tumor": col in tumor_set,
                            "chrom": ch,
                            "start": s_bp,
                            "end": e_bp,
                            "size_bp": size,
                            "cn": int(cn_val),
                            "direction": "gain" if cn_val > baseline else "loss",
                            "class": "focal" if size < focal_max_bp else "broad",
                        })
                    seg_start = i

    per_cell_df = pd.DataFrame(per_cell)
    events_df = pd.DataFrame(events)
    return per_cell_df, events_df


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def summarise(dataset, counts, depth_df, cnv_cell_df, events_df):
    row = {"dataset": dataset}
    row.update(counts)

    if not depth_df.empty:
        row["scRNA_reads_per_cell_mean"] = round(np.nanmean(depth_df["scRNA_reads"]), 1)
        row["scRNA_reads_per_cell_median"] = np.nanmedian(depth_df["scRNA_reads"])
        row["scWGS_reads_per_cell_mean"] = round(np.nanmean(depth_df["scWGS_reads"]), 1)
        if "scWGS_mean_depth" in depth_df:
            row["scWGS_mean_depth"] = round(np.nanmean(depth_df["scWGS_mean_depth"]), 4)

    if not cnv_cell_df.empty:
        tum = cnv_cell_df[cnv_cell_df["is_tumor"]]
        ref = tum if not tum.empty else cnv_cell_df
        row["tumor_ploidy_mean"] = round(float(ref["ploidy_meanCN"].mean()), 3)
        row["cnv_burden_FGA_mean"] = round(float(ref["frac_genome_altered"].mean()), 4)

    if not events_df.empty:
        tev = events_df[events_df["is_tumor"]]
        ref = tev if not tev.empty else events_df
        row["n_cnv_events_per_tumor_cell_mean"] = round(
            ref.groupby("cell_id").size().mean(), 2)
        row["cnv_event_size_bp_median"] = int(ref["size_bp"].median())
        row["frac_events_focal"] = round((ref["class"] == "focal").mean(), 4)
        row["frac_events_broad"] = round((ref["class"] == "broad").mean(), 4)

    return pd.DataFrame([row])


def run(config_paths, ginkgo_dir, out_dir, bam_prefix=None,
        samtools="samtools", with_coverage=False, skip_bams=False,
        segcopy_name="SegCopy"):
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(config_paths)
    dataset = config.get("dataset", "unknown_dataset")
    prefix = bam_prefix or os.environ.get(
        "SCWGS_SCRNA_PREFIX_OVERRIDE", config.get("scWGS_scRNA_prefix", ""))

    cells = parse_cells(config)
    if cells.empty:
        sys.exit("No cells parsed from config; check "
                 "cellname_celltype_scWGS_scRNA_tup_list.")

    counts = purity_and_counts(cells)

    depth_df = pd.DataFrame()
    if not skip_bams:
        depth_df = depth_per_cell(cells, prefix, samtools, with_coverage)

    cnv_cell_df, events_df = pd.DataFrame(), pd.DataFrame()
    if ginkgo_dir:
        segcopy = load_segcopy(ginkgo_dir, segcopy_name)
        tumor_cells = cells.loc[cells["celltype"].str.lower() == "tumor",
                                "cell_id"].tolist()
        cnv_cell_df, events_df = cnv_metrics_from_segcopy(segcopy, tumor_cells)

    # per-cell table: merge depth + cnv
    per_cell = cells[["cell_id", "celltype"]]
    if not depth_df.empty:
        per_cell = per_cell.merge(depth_df, on="cell_id", how="left")
    if not cnv_cell_df.empty:
        per_cell = per_cell.merge(cnv_cell_df.drop(columns=["is_tumor"]),
                                  on="cell_id", how="left")

    summary = summarise(dataset, counts, depth_df, cnv_cell_df, events_df)

    summary.to_csv(os.path.join(out_dir, "dataset_metrics.tsv"),
                   sep="\t", index=False)
    per_cell.to_csv(os.path.join(out_dir, "per_cell_metrics.tsv"),
                    sep="\t", index=False)
    if not events_df.empty:
        events_df.to_csv(os.path.join(out_dir, "cnv_event_sizes.tsv"),
                         sep="\t", index=False)

    print(summary.to_string(index=False))
    return summary


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _from_snakemake(smk):
    cfgs = list(smk.input.get("configs", [])) or list(smk.params.get("config_paths", []))
    run(
        config_paths=cfgs,
        ginkgo_dir=smk.params.get("ginkgo_dir", ""),
        out_dir=os.path.dirname(smk.output[0]),
        bam_prefix=smk.params.get("bam_prefix"),
        samtools=smk.params.get("samtools", "samtools"),
        with_coverage=bool(smk.params.get("with_coverage", False)),
        skip_bams=bool(smk.params.get("skip_bams", False)),
        segcopy_name=smk.params.get("segcopy_name", "SegCopy"),
    )


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True,
                    help="config_template.yaml and the dataset YAML (template first)")
    ap.add_argument("--ginkgo-dir", default="",
                    help="Dir containing the Ginkgo SegCopy ground truth")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bam-prefix", default=None)
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--segcopy-name", default="SegCopy")
    ap.add_argument("--with-coverage", action="store_true",
                    help="Also compute scWGS mean depth (slow: samtools depth)")
    ap.add_argument("--skip-bams", action="store_true",
                    help="Skip all BAM I/O (purity/ploidy/CNV only)")
    a = ap.parse_args()
    run(a.config, a.ginkgo_dir, a.out_dir, a.bam_prefix,
        a.samtools, a.with_coverage, a.skip_bams, a.segcopy_name)


if __name__ == "__main__":
    if "snakemake" in globals():          # injected by Snakemake `script:`
        _from_snakemake(globals()["snakemake"])
    else:
        _cli()
