#!/usr/bin/env python3
'''
https://sorryios.ai/chat/db9526db-f562-4ddc-9826-d8146c77363a
ls /nfs/wxz/zxf/copy-compass-sc-rna/sra-tables/DNTRseq_DNA_PRJNA603321_fqs/A375_*.fastq.gz | python3 coseq_configs/make_dntr_coseq_config.py -o coseq_configs/DNTR-seq_A375.yaml
ls /nfs/wxz/zxf/copy-compass-sc-rna/sra-tables/DNTRseq_DNA_PRJNA603321_fqs/HCT116_*.fastq.gz | python3 coseq_configs/make_dntr_coseq_config.py -o coseq_configs/DNTR-seq_HCT116.yaml
'''

"""
make_dntr_coseq_config.py

Build a coseq YAML config (same format as the
copy-num-bench-scwgs-to-scrna `coseq_configs/config_*.yaml` files) from a list
of DNTR-seq FASTQ filepaths.

DNTR-seq co-sequences DNA (GENOMIC) and RNA (TRANSCRIPTOMIC) from the *same*
single cell. The two libraries share a sample base such as `A375_HCA00101-A05`
(cellline_plate-well) and differ only by the GENOMIC / TRANSCRIPTOMIC token and
the SRR accession, e.g.:

    A375_HCA00101-A05_GENOMIC_SRR12860123_1.fastq.gz   <- DNA  read 1
    A375_HCA00101-A05_GENOMIC_SRR12860123_2.fastq.gz   <- DNA  read 2
    A375_HCA00101-A05_TRANSCRIPTOMIC_SRR12860229_1.fastq.gz  <- RNA read 1
    A375_HCA00101-A05_TRANSCRIPTOMIC_SRR12860229_2.fastq.gz  <- RNA read 2

Cells are paired on the sample base (the same logic as the reference
SRP245376_generate_commands.sh shell script: keep a cell only when both a
GENOMIC and a TRANSCRIPTOMIC library exist for it).

Output YAML structure (mirrors the reference template):

    dataset: <name>
    scWGS_scRNA_prefix: <prefix>/
    cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list:
    - - <cellname>
      - <celltype>
      - <DNA fq1 relative to prefix>
      - <DNA fq2 relative to prefix or '-'>
      - <RNA fq1 relative to prefix>
      - <RNA fq2 relative to prefix or '-'>
    - - ...

Usage examples
--------------
  # paths from a glob on the command line
  ls /nfs/.../DNTRseq_DNA_PRJNA603321_fqs/*.fastq.gz \
      | python3 make_dntr_coseq_config.py -o config_A375.yaml

  # paths listed in a file
  python3 make_dntr_coseq_config.py -i fastq_list.txt -o config_A375.yaml

  # paths as positional arguments
  python3 make_dntr_coseq_config.py /path/*.fastq.gz -o config_A375.yaml

  # one YAML per plate instead of a single combined file
  python3 make_dntr_coseq_config.py -i fastq_list.txt --split-by-plate -O configs/
"""

import argparse
import os
import re
import sys

import yaml

# Placeholder used by the reference configs for an absent FASTQ (e.g. a
# single-end library has no read 2).
MISSING = "-"

# DNTR-seq FASTQ filename:
#   <sample>_<GENOMIC|TRANSCRIPTOMIC>_<accession>_<1|2>.fastq[.gz]
# <sample> may itself contain underscores (e.g. A375_HCA00101-A05), so the
# library-type token anchors the split.
FASTQ_RE = re.compile(
    r"^(?P<sample>.+)_(?P<libtype>GENOMIC|TRANSCRIPTOMIC)_"
    r"(?P<acc>[SED]RR\d+)_(?P<read>[12])\.fastq(?:\.gz)?$"
)

# A sample base looks like <cellline>_<plate>-<well>, e.g. A375_HCA00101-A05.
# Captured loosely so non-standard names still parse; the plate is only used
# for --split-by-plate and for deriving a default dataset name.
SAMPLE_RE = re.compile(r"^(?P<cellline>.+)_(?P<plate>[^_-]+)-(?P<well>[^_-]+)$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a DNTR-seq coseq YAML config from FASTQ filepaths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input (choose one; defaults to stdin)")
    src.add_argument(
        "fastqs",
        nargs="*",
        help="FASTQ filepaths as positional arguments.",
    )
    src.add_argument(
        "-i",
        "--input-list",
        help="Text file with one FASTQ filepath per line.",
    )

    p.add_argument(
        "-o",
        "--output",
        help="Output YAML path (single combined config). "
        "Defaults to stdout when not splitting by plate.",
    )
    p.add_argument(
        "-O",
        "--output-dir",
        help="Output directory (used with --split-by-plate).",
    )
    p.add_argument(
        "--split-by-plate",
        action="store_true",
        help="Write one YAML per plate (matches the per-chip reference configs) "
        "instead of a single combined file.",
    )

    p.add_argument(
        "--prefix",
        default="/nfs/wxz/zxf/copy-compass-sc-rna/sra-tables/",
        help="scWGS_scRNA_prefix; FASTQ paths under it are stored relative to it.",
    )
    p.add_argument(
        "--dataset",
        help="dataset name. Default is derived from the data "
        "(cellline_plate[_project] when unambiguous, else 'DNTRseq_dataset').",
    )
    p.add_argument(
        "--celltype",
        default="Tumor",
        help="celltype label written for every cell.",
    )
    p.add_argument(
        "--cellname-mode",
        choices=["sample", "plate-well", "well"],
        default="sample",
        help="How to name each cell: full sample base (A375_HCA00101-A05), "
        "plate-well (HCA00101-A05), or well (A05).",
    )
    p.add_argument(
        "--keep-unpaired",
        action="store_true",
        help="Also emit cells that have only DNA or only RNA "
        "(missing side filled with '-'). Default drops them.",
    )
    return p.parse_args(argv)


def read_fastq_paths(args):
    """Collect FASTQ paths from positionals, a list file, or stdin."""
    paths = list(args.fastqs)
    if args.input_list:
        with open(args.input_list) as fh:
            paths += [ln.strip() for ln in fh if ln.strip()]
    if not paths and not sys.stdin.isatty():
        paths += [ln.strip() for ln in sys.stdin if ln.strip()]
    # De-duplicate while preserving order.
    seen, uniq = set(), []
    for path in paths:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def rel_to_prefix(path, prefix):
    """Path relative to prefix; if not under prefix, return it unchanged."""
    norm_prefix = prefix if prefix.endswith("/") else prefix + "/"
    if path.startswith(norm_prefix):
        return path[len(norm_prefix):]
    # Fall back to a normalized relpath so the YAML is still usable.
    try:
        rel = os.path.relpath(path, norm_prefix)
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass
    return path


def cellname_for(sample, mode):
    m = SAMPLE_RE.match(sample)
    if not m or mode == "sample":
        return sample
    if mode == "plate-well":
        return f"{m.group('plate')}-{m.group('well')}"
    if mode == "well":
        return m.group("well")
    return sample


def build_cells(paths, args):
    """
    Parse paths and group them into per-cell records.

    Returns (cells, warnings) where cells is an ordered list of dicts with keys:
      sample, plate, cellline, dna (r1,r2), rna (r1,r2).
    """
    # sample -> {'dna': {1: path, 2: path}, 'rna': {...}, 'plate':, 'cellline':}
    groups = {}
    order = []
    warnings = []

    for path in paths:
        fname = os.path.basename(path)
        m = FASTQ_RE.match(fname)
        if not m:
            warnings.append(f"skip (unrecognized FASTQ name): {path}")
            continue
        sample = m.group("sample")
        libtype = m.group("libtype")
        read = int(m.group("read"))
        side = "dna" if libtype == "GENOMIC" else "rna"

        rec = groups.get(sample)
        if rec is None:
            sm = SAMPLE_RE.match(sample)
            rec = {
                "sample": sample,
                "plate": sm.group("plate") if sm else None,
                "cellline": sm.group("cellline") if sm else None,
                "dna": {},
                "rna": {},
            }
            groups[sample] = rec
            order.append(sample)

        if read in rec[side]:
            warnings.append(
                f"duplicate {libtype} read{read} for {sample}; "
                f"keeping first, ignoring: {path}"
            )
        else:
            rec[side][read] = rel_to_prefix(path, args.prefix)

    cells = []
    for sample in order:
        rec = groups[sample]
        has_dna = bool(rec["dna"])
        has_rna = bool(rec["rna"])
        if not (has_dna and has_rna) and not args.keep_unpaired:
            missing = "RNA" if has_dna else "DNA"
            warnings.append(f"drop {sample}: no {missing} library (use --keep-unpaired to keep)")
            continue
        cells.append(rec)
    return cells, warnings


def cell_to_tuple(rec, args):
    name = cellname_for(rec["sample"], args.cellname_mode)
    dna = rec["dna"]
    rna = rec["rna"]
    return [
        name,
        args.celltype,
        dna.get(1, MISSING),
        dna.get(2, MISSING),
        rna.get(1, MISSING),
        rna.get(2, MISSING),
    ]


def derive_dataset_name(cells, project_hint=None):
    """Build a default dataset name from the cells when not given explicitly."""
    celllines = {c["cellline"] for c in cells if c["cellline"]}
    plates = {c["plate"] for c in cells if c["plate"]}
    parts = []
    if len(celllines) == 1:
        parts.append(next(iter(celllines)))
    if len(plates) == 1:
        parts.append(next(iter(plates)))
    if project_hint:
        parts.append(project_hint)
    return "_".join(parts) if parts else "DNTRseq_dataset"


def project_from_paths(paths):
    """Grab a PRJNA/PRJEB/PRJDB accession from any path, if present."""
    for path in paths:
        m = re.search(r"PRJ(?:NA|EB|DB)\d+", path)
        if m:
            return m.group(0)
    return None


def dump_config(dataset, prefix, tuples):
    data = {
        "dataset": dataset,
        "scWGS_scRNA_prefix": prefix if prefix.endswith("/") else prefix + "/",
        "cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list": tuples,
    }
    return yaml.safe_dump(
        data, default_flow_style=False, sort_keys=False, width=4096
    )


def write_output(text, out_path):
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(text)
        sys.stderr.write(f"wrote {out_path}\n")
    else:
        sys.stdout.write(text)


def main(argv=None):
    args = parse_args(argv)
    paths = read_fastq_paths(args)
    if not paths:
        sys.exit("error: no FASTQ filepaths provided (positional, -i, or stdin).")

    cells, warnings = build_cells(paths, args)
    for w in warnings:
        sys.stderr.write(f"[warn] {w}\n")
    if not cells:
        sys.exit("error: no cells with the required libraries were found.")

    project = project_from_paths(paths)

    if args.split_by_plate:
        out_dir = args.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        # Group cells by plate, preserving first-seen order.
        by_plate = {}
        for rec in cells:
            by_plate.setdefault(rec["plate"], []).append(rec)
        for plate, recs in by_plate.items():
            dataset = args.dataset or derive_dataset_name(recs, project)
            tuples = [cell_to_tuple(r, args) for r in recs]
            text = dump_config(dataset, args.prefix, tuples)
            plate_tag = plate if plate else "unknown_plate"
            write_output(text, os.path.join(out_dir, f"config_{dataset}.yaml"
                                            if args.dataset is None
                                            else f"config_{dataset}_{plate_tag}.yaml"))
        sys.stderr.write(f"done: {len(cells)} cells across {len(by_plate)} plate(s)\n")
        return

    dataset = args.dataset or derive_dataset_name(cells, project)
    tuples = [cell_to_tuple(r, args) for r in cells]
    text = dump_config(dataset, args.prefix, tuples)
    write_output(text, args.output)
    sys.stderr.write(f"done: {len(cells)} cells, dataset={dataset}\n")


if __name__ == "__main__":
    main()
