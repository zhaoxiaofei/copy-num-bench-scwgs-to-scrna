#!/usr/bin/env python3
"""
compute_dataset_metrics.py

Compute per-dataset descriptive / confounder metrics for the
copy-num-bench-scwgs-to-scrna benchmark.

Two independent tumor/normal labelings are used, and every label-dependent metric
is reported once per labeling, plus an agreement summary between the two:

  1. "sample"  annotation  -- the author/config-provided labels
                              data/{DATASET}_input/sample_annotation_1.txt
                              (written by rule create_sample_annotation_1)
  2. "dna"     annotation  -- labels inferred from the Ginkgo copy-number ground truth
                              data/{dataset}_ginkgo_groundtruth/dna_cell_annotation.txt
                              (written by separate_tumor_normal_from_ginkgo_segcopy.py)

Both files are 2-column, NO HEADER:  <sample_name>\t<tumor|reference>
(reference == normal). Any label not in _REF_CELLTYPES is treated as tumor.

Metrics produced
----------------
Per labeling (suffixed __sample / __dna in the headline row):
  * Tumor purity        -> fraction of malignant cells
  * Tumor ploidy        -> mean integer CN over that labeling's tumor cells (Ginkgo)
  * CNV burden (FGA)    -> mean fraction-of-genome-altered over tumor cells
  * CNV event sizes     -> per-segment size distribution over tumor cells

Labeling-independent:
  * Sequencing depth    -> reads / cell per modality, from the pipeline's *.bam.stats
  * Number of cells     -> total and scRNA cell counts

Agreement between the two labelings:
  * confusion counts (tumor/tumor, tumor/ref, ref/tumor, ref/ref), Cohen's kappa,
    and the list of discordant cells.

Pipeline paths this script relies on (see snakemake_pipeline/new_workflow.snake):
  scWGS / DNA BAM : results/{dataset}_bams/dna/{cell}_hg19.bam            (+ .bam.stats)
  scRNA BAM       : results/{dataset}_bams/rna/all_cells_grch38_{cell}.bam (+ .bam.stats)
  CNV ground truth: results/{dataset}_ginkgo/output/SegCopy_grch38.tsv

Runs standalone (argparse) or from a Snakemake `script:` directive (auto-detected).

Outputs (into --out-dir):
  dataset_metrics.tsv             one headline row (metrics under both labelings)
  per_cell_metrics.tsv            one row per cell (both labels, depth, ploidy, FGA)
  cnv_event_sizes.tsv             one row per CNV segment, tagged with both labels
  annotation_agreement.tsv        confusion / kappa summary
  annotation_discordant_cells.tsv cells where the two labelings disagree
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# Celltype classification -- kept in sync with new_workflow.snake
# --------------------------------------------------------------------------- #
_REF_CELLTYPES = {
    "n", "normal", "control", "ctrl", "reference", "ref",
    "diploid", "bl", "lymphocyte", "lymphocite", "blood",
}

# Raw labels that mean "no tumor/normal call" in an annotation file. Written by
# rule create_sample_annotation_1 as 'unknown' for every cell exactly when the
# config used 'Unknown' throughout (see new_workflow.snake).
_UNKNOWN_LABELS = {"unknown", "unk", "na", "n/a", "", "-", "."}

def _norm_cell_key(x):
    return str(x).strip().replace('/', '.').replace('_', '.').replace('-', '.')

def _is_ref_celltype(celltype):
    """True if the celltype string denotes a normal/reference (diploid control) cell."""
    return str(celltype).strip().lower() in _REF_CELLTYPES

def _is_unknown_celltype(celltype):
    return str(celltype).strip().lower() in _UNKNOWN_LABELS


def _label_of(celltype):
    """Normalise any celltype string to canonical 'reference' / 'tumor'."""
    if _is_ref_celltype(celltype): return "reference"
    if _is_unknown_celltype(celltype): return "unknown"
    return "tumor"


def _is_empty_fq(fq_value):
    """Match the Snakefile's placeholder handling for absent FASTQ mates."""
    if fq_value is None:
        return True
    return str(fq_value).strip() in ("", "-", ".")


# --------------------------------------------------------------------------- #
# Config + annotation parsing
# --------------------------------------------------------------------------- #
def load_config(config_paths):
    """Merge one or more YAML files (template + dataset config), last wins."""
    merged = {}
    for p in config_paths:
        with open(p) as fh:
            d = yaml.safe_load(fh) or {}
        merged.update(d)
    return merged


CONFIG_CELL_KEY = "cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list"


def parse_cells(config):
    """Return a DataFrame of cells from the 6-tuple FastQ-mode config list."""
    rows = []
    for entry in config.get(CONFIG_CELL_KEY, []):
        if not isinstance(entry, (list, tuple)) or len(entry) < 6:
            continue
        cell_id, celltype, dna_fq1, dna_fq2, rna_fq1, rna_fq2 = entry[:6]
        rows.append(
            {
                "cell_id": str(cell_id),
                "config_celltype": _label_of(str(celltype).strip()),
                "has_dna_fq": not _is_empty_fq(dna_fq1),
                "has_rna_fq": not _is_empty_fq(rna_fq1),
            }
        )
    return pd.DataFrame(rows)


def load_annotation(path, name):
    """Load a 2-column, NO-HEADER annotation file: <sample_name>\\t<tumor|reference>.

    Returns dict {sample_name: 'tumor'|'reference'}. Tab- or whitespace-delimited.
    Unknown/blank labels are skipped with a warning. `name` is used only in messages.
    """
    if not path or not os.path.exists(path):
        print(f"WARNING: {name} annotation not found: {path}", file=sys.stderr)
        return {}
    mapping = {}
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                print(f"WARNING: {name} annotation line {ln} has <2 columns; skipped",
                      file=sys.stderr)
                continue
            sample1, label = parts[0].strip(), parts[1].strip()
            sample = _norm_cell_key(sample1) # .replace('/', '.').replace('_', '.').replace('-', '.')
            print(f'Processing file={path} sample1={sample1} sample={sample}')
            assert sample not in mapping, f'{sample} is duplicated at {path}! Aborting!'
            mapping[sample] = _label_of(label)
    return mapping

def annotation_is_all_unknown(path):
    """True if every raw label in a NO-HEADER annotation file is an 'unknown' placeholder.

    Read BEFORE _label_of canonicalisation (which would turn 'unknown' into
    'tumor'). Used to decide whether the author/config 'sample' labeling is real
    or a placeholder deferring to the DNA (Ginkgo) labeling. A missing file counts
    as unknown.
    """
    if not path or not os.path.exists(path):
        return True
    saw_label = False
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            saw_label = True
            if parts[1].strip().lower() not in _UNKNOWN_LABELS:
                return False
    return not saw_label

# --------------------------------------------------------------------------- #
# Pipeline output BAM / stats paths (must match new_workflow.snake exactly)
# --------------------------------------------------------------------------- #
def scwgs_bam_for(dataset, cell_id, results_prefix=""):
    return os.path.join(results_prefix,
                        f"results/{dataset}_bams/dna/{cell_id}_hg19.bam")


def scrna_bam_for(dataset, cell_id, results_prefix=""):
    return os.path.join(results_prefix,
                        f"results/{dataset}_bams/rna/all_cells_grch38_{cell_id}.bam")


# --------------------------------------------------------------------------- #
# Sequencing depth / reads per cell, from `*.bam.stats` (labeling-independent)
# --------------------------------------------------------------------------- #
def parse_samtools_stats(stats_path):
    """Parse the SN (summary numbers) block of a `samtools stats` output file."""
    if not stats_path or not os.path.exists(stats_path):
        if stats_path:
            print(f"WARNING: stats file not found: {stats_path}", file=sys.stderr)
        return {}
    out = {}
    with open(stats_path) as fh:
        for line in fh:
            if not line.startswith("SN\t"):
                continue
            payload = line[3:]
            key, _, rest = payload.partition(":")
            value = rest.split("\t")[1] if "\t" in rest else rest.strip()
            out[key.strip()] = value.strip()
    return out


def _stats_int(stats, key):
    try:
        return int(stats[key])
    except (KeyError, ValueError):
        return np.nan


def depth_per_cell(cells, dataset, results_prefix=""):
    recs = []
    for _, c in cells.iterrows():
        cid = c["cell_id"]
        d = parse_samtools_stats(scwgs_bam_for(dataset, cid, results_prefix) + ".stats")
        r = parse_samtools_stats(scrna_bam_for(dataset, cid, results_prefix) + ".stats")
        recs.append({
            "cell_id": cid,
            "scWGS_total_reads": _stats_int(d, "raw total sequences"),
            "scWGS_reads_mapped": _stats_int(d, "reads mapped"),
            "scWGS_error_rate": float(d["error rate"]) if "error rate" in d else np.nan,
            "scRNA_total_reads": _stats_int(r, "raw total sequences"),
            "scRNA_reads_mapped": _stats_int(r, "reads mapped"),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Per-cell CNV profile (ploidy, FGA, events) from Ginkgo SegCopy_grch38
# This is labeling-INDEPENDENT; tumor/normal is applied later when aggregating.
# --------------------------------------------------------------------------- #
def load_segcopy(segcopy_path):
    if not segcopy_path or not os.path.exists(segcopy_path):
        raise FileNotFoundError(
            f"SegCopy ground truth not found at {segcopy_path!r}. "
            "Point --segcopy at results/<dataset>_ginkgo/output/SegCopy_grch38.tsv."
        )
    df = pd.read_csv(segcopy_path, sep=r"\s+", engine="python")
    df = df.rename(columns={df.columns[0]: "CHR",
                            df.columns[1]: "START",
                            df.columns[2]: "END"})
    df["CHR"] = df["CHR"].astype(str)
    df["START"] = pd.to_numeric(df["START"], errors="coerce")
    df["END"] = pd.to_numeric(df["END"], errors="coerce")
    return df


def cnv_profile_from_segcopy(segcopy, focal_max_bp=3_000_000):
    """Per-cell ploidy + FGA, and a per-segment events table. No tumor/normal here."""
    bin_len = (segcopy["END"] - segcopy["START"]).to_numpy(dtype=float)
    genome_len = float(np.nansum(bin_len))

    chrom = segcopy["CHR"].to_numpy()
    start = segcopy["START"].to_numpy()
    end = segcopy["END"].to_numpy()

    cell_cols = [c for c in segcopy.columns if c not in ("CHR", "START", "END")]

    per_cell = []
    events = []

    for col in cell_cols:
        cn = pd.to_numeric(segcopy[col], errors="coerce").to_numpy()
        if np.all(np.isnan(cn)):
            continue

        baseline = Counter(cn[~np.isnan(cn)]).most_common(1)[0][0]
        altered = (cn != baseline) & ~np.isnan(cn)
        fga = float(np.nansum(bin_len[altered]) / genome_len) if genome_len else np.nan
        ploidy = float(np.nanmean(cn))

        per_cell.append({
            "cell_id": str(col),
            "ploidy_meanCN": round(ploidy, 4),
            "baseline_CN": int(baseline),
            "frac_genome_altered": round(fga, 4) if not np.isnan(fga) else np.nan,
        })

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
                            "cell_id": str(col),
                            "chrom": ch,
                            "start": s_bp,
                            "end": e_bp,
                            "size_bp": size,
                            "cn": int(cn_val),
                            "direction": "gain" if cn_val > baseline else "loss",
                            "class": "focal" if size < focal_max_bp else "broad",
                        })
                    seg_start = i

    return pd.DataFrame(per_cell), pd.DataFrame(events)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _safe_mean(series):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan


def _safe_median(series):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmedian(arr)) if np.any(~np.isnan(arr)) else np.nan


def _round(x, n):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return round(x, n)


def purity_counts_for_labeling(cells, label_col):
    """Purity + cell counts under one labeling column ('tumor'/'reference')."""
    lab = cells[label_col].fillna("")
    is_tumor = lab.eq("tumor")
    is_ref = lab.eq("reference")
    n_tumor = int(is_tumor.sum())
    n_ref = int(is_ref.sum())
    n_lab = n_tumor + n_ref
    purity = n_tumor / n_lab if n_lab else np.nan
    n_rna_tumor = int((cells["has_rna_fq"] & is_tumor).sum())
    return {
        "n_tumor_cells": n_tumor,
        "n_normal_cells": n_ref,
        "n_labeled_cells": n_lab,
        "tumor_purity_cellfraction": _round(purity, 4),
        "n_cells_scRNA_tumor": n_rna_tumor,
    }


def cnv_summary_for_labeling(cells, label_col, cnv_cell_df, events_df,
                             focal_max_bp=3_000_000):
    """Ploidy / FGA / event-size summary over the tumor cells of one labeling."""
    out = {}
    tumor_ids = set(cells.loc[cells[label_col].eq("tumor"), "cell_id"].astype(str))

    if not cnv_cell_df.empty:
        sub = cnv_cell_df[cnv_cell_df["cell_id"].astype(str).isin(tumor_ids)]
        ref = sub if not sub.empty else cnv_cell_df  # fall back if none labeled tumor
        out["tumor_sample_ploidy_mean"] = _round(_safe_mean(ref["ploidy_meanCN"]), 3)
        out["cnv_burden_FGA_mean"] = _round(_safe_mean(ref["frac_genome_altered"]), 4)

    if not events_df.empty:
        sub = events_df[events_df["cell_id"].astype(str).isin(tumor_ids)]
        ref = sub if not sub.empty else events_df
        if not ref.empty:
            out["n_cnv_events_per_tumor_cell_mean"] = _round(
                float(ref.groupby("cell_id").size().mean()), 2)
            med = _safe_median(ref["size_bp"])
            out["cnv_event_size_bp_median"] = int(med) if not np.isnan(med) else np.nan
            out["frac_events_focal"] = _round(float((ref["class"] == "focal").mean()), 4)
            out["frac_events_broad"] = _round(float((ref["class"] == "broad").mean()), 4)
    return out


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two equal-length lists of categorical labels."""
    n = len(labels_a)
    if n == 0:
        return np.nan
    cats = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        m[idx[a], idx[b]] += 1
    po = np.trace(m) / n
    row = m.sum(axis=1) / n
    col = m.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def annotation_agreement(cells):
    """Confusion counts + kappa + discordant cells between the two labelings.

    Only cells present (non-null) in BOTH labelings are compared.
    Returns (summary_dict, discordant_dataframe).
    """
    both = cells.dropna(subset=["label_sample", "label_dna"])

    def cnt(x, y):
        return int(((both["label_sample"] == x) & (both["label_dna"] == y)).sum())

    tt, tr = cnt("tumor", "tumor"), cnt("tumor", "reference")
    rt, rr = cnt("reference", "tumor"), cnt("reference", "reference")
    n = len(both)
    concordant = tt + rr
    summary = {
        "n_cells_compared": n,
        "sampleTumor_dnaTumor": tt,
        "sampleTumor_dnaReference": tr,
        "sampleReference_dnaTumor": rt,
        "sampleReference_dnaReference": rr,
        "n_concordant": concordant,
        "n_discordant": n - concordant,
        "frac_concordant": _round(concordant / n, 4) if n else np.nan,
        "cohen_kappa": _round(cohen_kappa(both["label_sample"].tolist(),
                                          both["label_dna"].tolist()), 4),
    }
    disc = both[both["label_sample"] != both["label_dna"]][
        ["cell_id", "config_celltype", "label_sample", "label_dna"]
    ].copy()
    return summary, disc


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_headline(dataset, ref_cell_source, base_counts, depth_df, cells, cnv_cell_df, events_df,
                   focal_max_bp=3_000_000):
    """One wide row: shared metrics + per-labeling metrics suffixed __sample / __dna."""
    row = {"dataset": dataset, "ref_cell_identification": ref_cell_source}
    row.update(base_counts)

    if not depth_df.empty:
        row["scRNA_reads_per_cell_mean"] = _round(_safe_mean(depth_df["scRNA_reads_mapped"]), 1)
        row["scRNA_reads_per_cell_median"] = _safe_median(depth_df["scRNA_reads_mapped"])
        row["scWGS_reads_per_cell_mean"] = _round(_safe_mean(depth_df["scWGS_reads_mapped"]), 1)
        row["scWGS_reads_per_cell_median"] = _safe_median(depth_df["scWGS_reads_mapped"])
        if "scWGS_error_rate" in depth_df:
            row["scWGS_error_rate_mean"] = _round(_safe_mean(depth_df["scWGS_error_rate"]), 5)

    for tag, col in (("sample", "label_sample"), ("dna", "label_dna")):
        block = {}
        block.update(purity_counts_for_labeling(cells, col))
        block.update(cnv_summary_for_labeling(cells, col, cnv_cell_df, events_df,
                                              focal_max_bp=focal_max_bp))
        for k, v in block.items():
            row[f"{k}__{tag}"] = v

    return pd.DataFrame([row])


def run(config_paths, segcopy_path, out_dir,
        sample_annotation, dna_annotation,
        dataset=None, results_prefix="", skip_bams=False,
        focal_max_bp=3_000_000):
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    config = load_config(config_paths)
    if dataset is None:
        dataset = config.get("dataset", "unknown_dataset")

    cells = parse_cells(config)
    if cells.empty:
        sys.exit(f"No cells parsed from config; check '{CONFIG_CELL_KEY}'.")

    sample_map = load_annotation(sample_annotation, "sample")
    dna_map = load_annotation(dna_annotation, "dna")

    # When the author/config 'sample' labeling is a placeholder (all-'unknown',
    # i.e. the config used 'Unknown' for every cell), the tumor/normal statuses
    # were inferred from the scWGS (Ginkgo) DNA data. Use the DNA labeling as the
    # effective 'sample' labeling, and asterisk the dataset name to flag that the
    # '__sample' metrics are DNA-inferred rather than author-provided.
    sample_is_unknown = annotation_is_all_unknown(sample_annotation)
    if sample_is_unknown:
        print("[metrics] sample annotation is all-unknown (config celltypes were "
              "'Unknown'); using DNA (Ginkgo) labeling as the effective 'sample' "
              "labeling and asterisking the dataset name.", file=sys.stderr)
        sample_map = dict(dna_map)
        # dataset = f"{dataset}*"
    ref_cell_source = ('scWGS-coseq-data' if sample_is_unknown else 'prior-publication')
    # revised at: https://sorryios.ai/chat/69cd3b8b-b68a-479f-ae80-5269790fd414
    _key = cells["cell_id"].map(_norm_cell_key)
    cells["label_sample"] = _key.map(sample_map)
    cells["label_dna"]    = _key.map(dna_map)

    #cells["label_sample"] = cells["cell_id"].map(sample_map)
    #cells["label_dna"] = cells["cell_id"].map(dna_map)

    base_counts = {
        "n_cells_total": len(cells),
        "n_cells_scRNA": int(cells["has_rna_fq"].sum()),
        "n_cells_sample_labeled": int(cells["label_sample"].notna().sum()),
        "n_cells_dna_labeled": int(cells["label_dna"].notna().sum()),
    }

    depth_df = pd.DataFrame()
    if not skip_bams:
        depth_df = depth_per_cell(cells, dataset, results_prefix)

    cnv_cell_df, events_df = pd.DataFrame(), pd.DataFrame()
    if segcopy_path:
        segcopy = load_segcopy(segcopy_path)
        cnv_cell_df, events_df = cnv_profile_from_segcopy(segcopy, focal_max_bp=focal_max_bp)

    per_cell = cells[["cell_id", "config_celltype",
                      "label_sample", "label_dna",
                      "has_dna_fq", "has_rna_fq"]].copy()
    if not depth_df.empty:
        per_cell = per_cell.merge(depth_df, on="cell_id", how="left")
    if not cnv_cell_df.empty:
        per_cell = per_cell.merge(cnv_cell_df, on="cell_id", how="left")

    if not events_df.empty:
        events_df = events_df.merge(
            cells[["cell_id", "label_sample", "label_dna"]], on="cell_id", how="left")

    headline = build_headline(dataset, ref_cell_source, base_counts, depth_df, cells,
                              cnv_cell_df, events_df, focal_max_bp=focal_max_bp)

    agree_summary, discordant = annotation_agreement(cells)
    if sample_is_unknown:
        # With no independent author labeling, sample==dna: agreement is trivial.
        # Blank the confusion/kappa so it isn't mistaken for real concordance.
        agree_summary = {k: (np.nan if k != "n_cells_compared" else agree_summary.get(k, 0))
                         for k in agree_summary}
        discordant = discordant.iloc[0:0]

    agree_df = pd.DataFrame([{"dataset": dataset, **agree_summary}])

    headline.to_csv(os.path.join(out_dir, "dataset_metrics.tsv"), sep="\t", index=False)
    per_cell.to_csv(os.path.join(out_dir, "per_cell_metrics.tsv"), sep="\t", index=False)
    agree_df.to_csv(os.path.join(out_dir, "annotation_agreement.tsv"), sep="\t", index=False)
    discordant.to_csv(os.path.join(out_dir, "annotation_discordant_cells.tsv"),
                      sep="\t", index=False)
    if not events_df.empty:
        events_df.to_csv(os.path.join(out_dir, "cnv_event_sizes.tsv"), sep="\t", index=False)

    print("=== dataset_metrics (per labeling: __sample / __dna) ===")
    print(headline.to_string(index=False))
    print("\n=== annotation_agreement (sample vs dna) ===")
    print(agree_df.to_string(index=False))
    if not discordant.empty:
        print(f"\n{len(discordant)} discordant cell(s) -> annotation_discordant_cells.tsv")
    print(f"Stored at {out_dir}")

    return headline, agree_df


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def _get(obj, key, default=None):
    try:
        val = obj.get(key)
    except (AttributeError, TypeError):
        return default
    return default if val is None else val


def _from_snakemake(smk):
    cfgs = list(_get(smk.input, "configs", []) or [])
    if not cfgs:
        cfgs = list(_get(smk.params, "config_paths", []) or [])

    run(
        config_paths=cfgs,
        segcopy_path=_get(smk.input, "segcopy", None) or _get(smk.params, "segcopy", None),
        out_dir=os.path.dirname(smk.output[0]) or ".",
        sample_annotation=_get(smk.input, "sample_annot", None)
                          or _get(smk.params, "sample_annotation", None),
        dna_annotation=_get(smk.input, "dna_annot", None)
                       or _get(smk.params, "dna_annotation", None),
        dataset=_get(smk.params, "dataset", None),
        results_prefix=_get(smk.params, "results_prefix", "") or "",
        skip_bams=bool(_get(smk.params, "skip_bams", False)),
        focal_max_bp=int(_get(smk.params, "focal_max_bp", 3_000_000)),
    )
    print(f'Finished running with configs={cfgs} smk={str(smk)}')


def _cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", required=True,
                   help="config_template.yaml and the dataset YAML (template first)")
    p.add_argument("--sample-annotation", required=True,
                   help="Path to original author-provided sample annotation TSV "
                        "(NO HEADER): original_sample_name\\ttumor|reference")
    p.add_argument("--dna-annotation", required=True,
                   help="Path to DNA sample annotation TSV (NO HEADER): "
                        "DNA-inferred_sample_name\\ttumor|reference")
    p.add_argument("--segcopy", default="",
                   help="Lifted Ginkgo ground truth "
                        "(results/<dataset>_ginkgo/output/SegCopy_grch38.tsv)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset", default=None,
                   help="Dataset stem for results/<dataset>_bams/... "
                        "(defaults to config['dataset'])")
    p.add_argument("--results-prefix", default="",
                   help="Prefix prepended to results/ BAM/stats paths "
                        "(use when not running from the Snakemake workdir)")
    p.add_argument("--focal-max-bp", type=int, default=3_000_000)
    p.add_argument("--skip-bams", action="store_true",
                   help="Skip BAM stats I/O (purity/ploidy/CNV/agreement only)")
    a = p.parse_args()
    run(
        config_paths=a.config,
        segcopy_path=a.segcopy,
        out_dir=a.out_dir,
        sample_annotation=a.sample_annotation,
        dna_annotation=a.dna_annotation,
        dataset=a.dataset,
        results_prefix=a.results_prefix,
        skip_bams=a.skip_bams,
        focal_max_bp=a.focal_max_bp,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        _from_snakemake(globals()["snakemake"])
    else:
        _cli()
