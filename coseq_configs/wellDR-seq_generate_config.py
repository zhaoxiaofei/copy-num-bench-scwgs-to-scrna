#!/usr/bin/env python3

'''
I first wrote this by designing a prompt to LLM agents.
  prompt private link: https://agent.minimaxi.com/chat?id=391860940923152
  prompt public link: https://agent.minimaxi.com/share/391996227695418?chat_type=2
Then, I revised the code manually
'''

"""
Generate Snakemake YAML configuration files from wellDR-seq data.

This script generates YAML config files (one per DNA-BioSample/RNA-SRA-RunID combination)
following the cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list format.

Usage:
    python generate_snakemake_configs.py \
        --wafer-match wafer_match_list.csv \
        --dna-run-table wellDRseq_DNA_PRJNA1086561_SraRunTable.csv \
        --rna-run-table wellDRseq_RNA_PRJNA1088478_SraRunTable.csv \
        --rna-mapping wellDRseq_RNA_mapping.txt \
        --dna-fastq-dir wellDRseq_DNA_PRJNA1086561_fqs \
        --rna-fastq-dir wellDRseq_RNA_PRJNA1088478_fqs \
        --output-dir output_configs
"""

import os
import itertools
import csv
import argparse
import yaml
from pathlib import Path
from collections import defaultdict


def parse_wafer_match_list(filepath):
    """
    Parse wafer_match_list.csv
    Returns list of dicts with keys: Cell, DNA_Barcode, RNA, RNA_Barcode, DNA_Cell_Name, Row, Col, SampleSourceWell, coor
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up column names (remove quotes)
            cleaned = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
            records.append(cleaned)
    return records


def parse_dna_run_table(filepath):
    """
    Parse DNA SRA run table.
    Returns dict: LibraryName -> dict with Run, BioSample, LibraryLayout, etc.
    """
    dna_runs = {}
    dna_runs2 = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            library_name = row.get('LibraryName', '').strip()
            run_id = row.get('Run', '').strip()
            biosample = row.get('BioSample', '').strip()
            library_layout = row.get('LibraryLayout', '').strip()
            sample_name = row.get('SampleName', '').strip()
            cell_name = extract_cell_name_from_library(library_name)
            if library_name and run_id:
                dna_run_key = (sample_name, library_name)
                assert dna_run_key not in dna_runs, f'The (sample_name, library_name) {dna_run_key} was already seen!'
                dna_runs[dna_run_key] = {
                    'Run': run_id,
                    'BioSample': biosample,
                    'LibraryLayout': library_layout,
                    'SampleName': sample_name,
                }
                if sample_name not in dna_runs2: dna_runs2[sample_name] = {}
                dna_runs2[sample_name][cell_name] = library_name
    return dna_runs, dna_runs2


def parse_rna_run_table(filepath):
    """
    Parse RNA SRA run table.
    Returns dict: Run -> dict with BioSample, SampleName, LibraryLayout, etc.
    """
    rna_runs = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get('Run', '').strip()
            biosample = row.get('BioSample', '').strip()
            sample_name = row.get('SampleName', '').strip()
            library_layout = row.get('LibraryLayout', '').strip()
            if run_id:
                rna_runs[run_id] = {
                    'BioSample': biosample,
                    'SampleName': sample_name,
                    'LibraryLayout': library_layout
                }
    return rna_runs


def parse_rna_mapping(filepath):
    """
    Parse RNA mapping file (tab-separated).
    Returns dict: SampleName -> SRA_runID
    """
    rna_mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample_name = row.get('SampleName', '').strip()
            sra_run_id = row.get('SRA_runID', '').strip()
            if sample_name and sra_run_id:
                rna_mapping[sample_name] = sra_run_id
    return rna_mapping


def find_dna_fastq_files(fastq_dir, run_id, layout):
    """
    Find DNA FastQ files for a given Run ID.
    Looks for files like: SRR28293679_1.fastq.gz, SRR28293679_2.fastq.gz
    Returns (fq1, fq2) where fq2 may be None for single-end data.
    """
    dir_path = Path(fastq_dir)
    if not dir_path.exists():
        return None, None

    # Look for paired-end files
    fq1 = dir_path / f"{run_id}_1.fastq.gz"
    fq2 = dir_path / f"{run_id}_2.fastq.gz"

    assert fq1.exists(), f"The file {fq1} does not exist!"
    if layout == 'SINGLE':
        return str(fq1), '-'
    elif layout == 'PAIRED':
        assert fq2.exists(), f"The file {fq1} does not exist!"
        return str(fq1), str(fq2)
    else: raise RuntimeError(f"The layout {layout} is invalid for {run_id} in {fastq_dir}")


def find_rna_fastq_files(fastq_dir, run_id, rna_barcode):
    """
    Find RNA FastQ file for a given Run ID and barcode.
    Looks in subdirectory: SRR28357487_2_fq_list_rna_m/*.fq.gz
    where filename (without extension) is the barcode.
    Returns (fq1, fq2) - fq2 is None since scRNA is single cell (one file per barcode)
    """
    dir_path = Path(fastq_dir)
    subdir = dir_path / f"{run_id}_2_fq_list_rna_m"
    barcode_file = subdir / f"{rna_barcode}.fq.gz"
    assert barcode_file.exists(), f"The file {barcode_file} does not exist!"
    return str(barcode_file), None

def extract_cell_name_from_library(library_name):
    """
    Extract cell name (e.g., C3367) from LibraryName (e.g., WDR_revision_ECIS57T_Chip1_DNA_C3367)
    The cell name appears to be the last part after 'DNA_' or just the suffix.
    """
    if '_DNA_' in library_name:
        return library_name.split('_DNA_')[-1].split('_')[0]
    elif '_RNA_' in library_name:
        return library_name.split('_RNA_')[-1].split('_')[0]
    else:
        # Try to extract last alphanumeric segment
        parts = library_name.split('_')
        for part in reversed(parts):
            if part and part[0] == 'C':
                return part
        for part in reversed(parts):
            if part and part[0].isalpha():
                return part
        return library_name

def main():
    parser = argparse.ArgumentParser(
        description='Generate Snakemake YAML configs from wellDR-seq data'
    )
    parser.add_argument('--wafer-match', required=True,
                        help='Path to wafer_match_list.csv')
    parser.add_argument('--dna-run-table', required=True,
                        help='Path to DNA SRA run table CSV')
    parser.add_argument('--rna-run-table', required=True,
                        help='Path to RNA SRA run table CSV')
    parser.add_argument('--rna-mapping', required=True,
                        help='Path to RNA mapping TXT')
    parser.add_argument('--dna-fastq-dir', required=True,
                        help='Directory containing DNA FastQ files')
    parser.add_argument('--rna-fastq-dir', required=True,
                        help='Directory containing RNA FastQ files')
    parser.add_argument('--output-dir', default='output_configs',
                        help='Output directory for YAML files')
    parser.add_argument('--base-prefix', default='',
                        help='Base prefix to strip from FastQ paths')

    args = parser.parse_args()

    # Parse input files
    print("Parsing input files...")
    wafer_records = parse_wafer_match_list(args.wafer_match)
    print(f"  Loaded {len(wafer_records)} wafer records")

    dna_runs, dna_sample2cell2lib = parse_dna_run_table(args.dna_run_table)
    print(f"  Loaded {len(dna_runs)} DNA runs")

    rna_runs = parse_rna_run_table(args.rna_run_table)
    print(f"  Loaded {len(rna_runs)} RNA runs")

    rna_mapping = parse_rna_mapping(args.rna_mapping)
    print(f"  Loaded {len(rna_mapping)} RNA mappings")

    # Build cell records with enriched metadata
    print("\nBuilding cell records...")
    enriched_records = []

    for wafer, (sample_name, cell2dnalib) in itertools.product(wafer_records, dna_sample2cell2lib.items()):
        # lib_name, run_info, and dna_runs are missing
        print(f"Processing wafer={wafer}, sample_name={sample_name}")
        cell_id = wafer.get('Cell', '').strip()
        dna_cell_name = wafer.get('DNA_Cell_Name', '').strip()
        rna_barcode = wafer.get('RNA_Barcode', '').strip()

        if not cell_id:
            print(f" Warning: skipping the malformed line record {wafer}!")
            continue

        lib_name = cell2dnalib.get(cell_id, '')
        if not lib_name:
            print(f" Warning: skipping the lib_name {lib_name}!")
            continue

        run_info = dna_runs[(sample_name, lib_name)]

        # Determine cell type
        celltype = 'Tumor'

        # Find DNA run info
        dna_run_id = None
        dna_biosample = None
        dna_samplename = None
        
        cell_name_in_lib = extract_cell_name_from_library(lib_name)
        assert cell_name_in_lib.startswith('C'), f'The lib_name {lib_name} does not have any valid cell name (extracted name is {cell_name_in_lib})!'
        assert cell_name_in_lib == cell_id, f"{cell_name_in_lib} == {cell_id} failed"

        dna_run_id = run_info['Run']
        dna_biosample = run_info['BioSample']
        dna_samplename = run_info['SampleName']
        dna_layout = run_info['LibraryLayout']

        if not dna_run_id:
            print(f"  Warning: No DNA run for cell {cell_id} (DNA_Barcode: {dna_cell_name})")
            continue

        # Find DNA FastQ files
        dna_fq1, dna_fq2 = find_dna_fastq_files(args.dna_fastq_dir, dna_run_id, dna_layout)
        if not dna_fq1:
            print(f"  Warning: No DNA FastQ for Run {dna_run_id}")
            continue

        # Find RNA runs for this barcode
        for sample_name, sra_run_id in rna_mapping.items():
            if sample_name  != dna_samplename: continue
            print(f"  DNA-RNA-match-found: DNA_sample_name={dna_samplename} RNA_sample_name={sample_name} cell_id={cell_id}")

            rna_fq1, rna_fq2 = find_rna_fastq_files(args.rna_fastq_dir, sra_run_id, rna_barcode)
            if rna_fq1:
                rna_biosample = rna_runs.get(sra_run_id, {}).get('BioSample', '')

                enriched_records.append({
                    'cell_id': cell_id,
                    'celltype': celltype,
                    'sample_name': sample_name,
                    'dna_biosample': dna_biosample,
                    'rna_biosample': rna_biosample,
                    'rna_run_id': sra_run_id,
                    'dna_fq1': dna_fq1,
                    'dna_fq2': dna_fq2,
                    'rna_fq1': rna_fq1,
                    'rna_fq2': rna_fq2
                })
            else:
                print(f"  RNA-fastq-not-found for DNA_sample_name={dna_samplename} RNA_sample_name={sample_name}"
                    + f"cell_id={cell_id} fq_dir={args.rna_fastq_dir} run_id={sra_run_id} rna_barcode={rna_barcode}!")

    print(f"  Built {len(enriched_records)} enriched records")

    # Group by (DNA_BioSample, RNA_SRA_RunID)
    print("\nGrouping records by (DNA_BioSample, RNA_SRA_RunID)...")
    grouped = defaultdict(list)
    for rec in enriched_records:
        key = (rec['sample_name'], rec['dna_biosample'], rec['rna_run_id'])
        record_list = [
            rec['cell_id'],
            rec['celltype'],
            rec['dna_fq1'],
            rec['dna_fq2'] if rec['dna_fq2'] else '-',
            rec['rna_fq1'],
            rec['rna_fq2'] if rec['rna_fq2'] else '-'
        ]
        grouped[key].append(record_list)

    print(f"  Found {len(grouped)} unique (DNA_BioSample, RNA_SRA_RunID) combinations")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine base prefix for all FastQ paths
    all_fq_paths = []
    for rec in enriched_records:
        all_fq_paths.extend([rec['dna_fq1'], rec['dna_fq2'], rec['rna_fq1'], rec['rna_fq2']])
    all_fq_paths = [p for p in all_fq_paths if p]

    common_prefix = os.path.commonpath(all_fq_paths)
    if not args.base_prefix:
        base_prefix = common_prefix
    else:
        base_prefix = args.base_prefix

    print(f"\nUsing base prefix: '{base_prefix}'")

    # Generate YAML files
    print("\nGenerating YAML files...")
    for (sample_name, dna_biosample, rna_run_id), records in sorted(grouped.items()):
        # Calculate relative paths
        relative_records = []
        for rec in records:
            cell_id, celltype, dna_fq1, dna_fq2, rna_fq1, rna_fq2 = rec

            rel_dna_fq1 = dna_fq1[len(base_prefix):].lstrip('/') if dna_fq1.startswith(base_prefix) else dna_fq1
            rel_dna_fq2 = dna_fq2[len(base_prefix):].lstrip('/') if dna_fq2 and dna_fq2.startswith(base_prefix) else dna_fq2
            rel_rna_fq1 = rna_fq1[len(base_prefix):].lstrip('/') if rna_fq1.startswith(base_prefix) else rna_fq1
            rel_rna_fq2 = rna_fq2[len(base_prefix):].lstrip('/') if rna_fq2 and rna_fq2.startswith(base_prefix) else rna_fq2

            relative_records.append([cell_id, celltype, rel_dna_fq1, rel_dna_fq2, rel_rna_fq1, rel_rna_fq2])

        # Create config dict
        config = {
            'dataset': f'{sample_name}_{dna_biosample}_{rna_run_id}',
            'scWGS_scRNA_prefix': base_prefix,
            'cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list': relative_records
        }

        # Write YAML file
        output_file = output_dir / f'config_{sample_name}_{dna_biosample}_{rna_run_id}.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"  Generated: {output_file.name} ({len(records)} cells)")

    print(f"\nDone! Generated {len(grouped)} YAML files in {output_dir}")


if __name__ == '__main__':
    main()
