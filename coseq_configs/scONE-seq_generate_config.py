#!/usr/bin/env python3
"""
Generate Snakemake YAML config files for scONE-seq data in FastQ format.

This script generates config files with the format:
    cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list

The config contains a list of 6-element lists:
    [cell_id, celltype, DNAseqFQ1, DNAseqFQ2, RNAseqFQ1, RNAseqFQ2]

If FQ2 is not available, FQ2 will be set to "-".

Usage:
    python scONE-seq_generate_config.py \
        --fq-list fq_file_list.txt \
        --output-dir output_configs \
        --scWGS_scRNA_prefix /common/path/prefix
"""

import os
import re
import argparse
import itertools
import yaml
from collections import defaultdict
from pathlib import Path


def parse_fastq_file_list(fq_list_path):
    """
    Parse the FastQ file list and separate into DNA and RNA files.

    DNA files: contain 'dedup_dna' in path and pattern:
        SRR*_GSM*_<SampleName>_scONEseq_<Well>_..._dedup_R1/R2.fastq.gz

    RNA files: contain 'dedup_rna' in path and pattern:
        SRR*_GSM*_<SampleName>_scONE-seq_<Well>_..._dedup_R1/R2.fastq.gz

    Returns:
        dict: {
            'dna': {sample_name: {'R1': [files], 'R2': [files]}},
            'rna': {sample_name: {'R1': [files], 'R2': [files]}}
        }
    """
    dna_files = defaultdict(lambda: {'R1': [], 'R2': []})
    rna_files = defaultdict(lambda: {'R1': [], 'R2': []})
    fq_paths = []

    with open(fq_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fq_paths.append(line)

    fq_prefix = os.path.commonpath(fq_paths)

    for line in fq_paths:
        # Determine if DNA or RNA file based on path
        if '/dedup_dna/' in line:
            file_type = 'dna'
            files_dict = dna_files
        elif '/dedup_rna/' in line:
            file_type = 'rna'
            files_dict = rna_files
        else:
            continue

        # Determine R1 or R2
        if '_R1.fastq.gz' in line:
            read_num = 'R1'
        elif '_R2.fastq.gz' in line:
            read_num = 'R2'
        else:
            continue

        # Extract sample name and well position
        match = re.match(
            r'(SRR\d+)_GSM\d+_([^ ]+)_scONEseq_(\d+)_.*_dedup_R\d+\.fastq\.gz',
            os.path.basename(line)
        )
        if match:
            srr_id = match.group(1)
            sample_name = match.group(2)
            well_pos = match.group(3)

            # Create cell_id from sample_name and well position
            cell_id = f"{sample_name}_{well_pos}"

            files_dict[sample_name][read_num].append({
                'file': line.removeprefix(fq_prefix),
                'srr_id': srr_id,
                'well': well_pos,
                'cell_id': cell_id
            })

    return {'dna': dna_files, 'rna': rna_files}, fq_prefix

def determine_celltype(sample_name):
    if 'HUVEC' in sample_name: return 'N'
    return 'T'

def match_dna_rna_by_sample(fq_data):
    """
    Match DNA and RNA files by sample name.

    For each sample, create cell entries with:
        [cell_id, celltype, DNA_FQ1, DNA_FQ2, RNA_FQ1, RNA_FQ2]
    """
    dna_by_sample = fq_data['dna']
    rna_by_sample = fq_data['rna']

    all_samples = set(dna_by_sample.keys()) | set(rna_by_sample.keys())

    celltype2samplename2record = defaultdict(dict) #matched_cells = []

    for sample_name in sorted(all_samples):
        celltype = determine_celltype(sample_name)

        # Get DNA files for this sample
        dna_r1_files = dna_by_sample.get(sample_name, {}).get('R1', [])
        dna_r2_files = dna_by_sample.get(sample_name, {}).get('R2', [])

        # Get RNA files for this sample
        rna_r1_files = rna_by_sample.get(sample_name, {}).get('R1', [])
        rna_r2_files = rna_by_sample.get(sample_name, {}).get('R2', [])

        # Create lookup by well position
        dna_by_well = {}
        for f in dna_r1_files + dna_r2_files:
            well = f['well']
            if well not in dna_by_well:
                dna_by_well[well] = {'R1': None, 'R2': None}
            dna_by_well[well][f['file'].endswith('R1.fastq.gz') and 'R1' or 'R2'] = f['file']

        rna_by_well = {}
        for f in rna_r1_files + rna_r2_files:
            well = f['well']
            if well not in rna_by_well:
                rna_by_well[well] = {'R1': None, 'R2': None}
            rna_by_well[well][f['file'].endswith('R1.fastq.gz') and 'R1' or 'R2'] = f['file']

        # Get all unique wells
        all_wells = set(dna_by_well.keys()) | set(rna_by_well.keys())

        for well in sorted(all_wells, key=lambda x: int(x)):
            cell_id = f"{sample_name}_{well}"

            dna_fq1 = dna_by_well.get(well, {}).get('R1', '-')
            dna_fq2 = dna_by_well.get(well, {}).get('R2', '-')
            rna_fq1 = rna_by_well.get(well, {}).get('R1', '-')
            rna_fq2 = rna_by_well.get(well, {}).get('R2', '-')

            matched_cell = [cell_id, celltype.replace('T', 'Tumor').replace('N', 'Normal'), dna_fq1, dna_fq2, rna_fq1, rna_fq2]
            if dna_fq1 != '-' and rna_fq1 != '-': 
                # Both available
                if celltype not in celltype2samplename2record: celltype2samplename2record[celltype] = defaultdict(list)
                celltype2samplename2record[celltype][sample_name].append(matched_cell)
            else:
                print(f"  Warning: The record {matched_cell} does not have both DNA and RNA sequencing data available and is therefore ignored!")
    return celltype2samplename2record # matched_cells


def generate_config_yaml(
        celltype2samplename2record, 
        output_path, scWGS_scRNA_prefix):
    """
    Generate YAML config file with the cell list.
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for tumor_sample, normal_sample in itertools.product(celltype2samplename2record['T'].keys(), celltype2samplename2record['N'].keys()):
        tumor_cell_list = sorted(celltype2samplename2record['T'][tumor_sample])
        normal_cell_list = sorted(celltype2samplename2record['N'][normal_sample])
        cell_list = tumor_cell_list + normal_cell_list
        config = {
            # The reference to the ploidy range of 1.5 to 2.5.
            # private link: https://chat.deepseek.com/a/chat/s/5042e866-ebfe-43b5-b6f2-3053f25190ee
            # public link: https://chat.deepseek.com/share/s0mvmnkscyjwzgn76k
            'ginkgo_binning': 'variable_500000_48_bwa', # 500-kb window size from https://www.science.org/doi/10.1126/sciadv.abp8901
            'ginkgo_extra_cmd_line_params': '--ploidy 1.5-2.5',
            'dataset': f'scONE-seq_{tumor_sample}_{normal_sample}_as_T_N',
            'scWGS_scRNA_prefix': scWGS_scRNA_prefix,
            'cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list': cell_list
        }
        with open(f'{output_path}_{tumor_sample}_{normal_sample}_as_T_N.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f'Generated {len(tumor_cell_list)} tumor cells from {tumor_sample} and {len(normal_cell_list)} normal cells from {normal_sample}.')
    for celltype in ['T', 'N']:
        sample2record = celltype2samplename2record[celltype]
        for sample, cell_list in sample2record.items():
            config = {
                'ginkgo_binning': 'variable_500000_48_bwa',
                'ginkgo_extra_cmd_line_params': '--ploidy 1.5-2.5',
                'dataset': f'scONE-seq_{sample}',
                'scWGS_scRNA_prefix': scWGS_scRNA_prefix,
                'cellname_celltype_DNAseqFQ1_DNAseqFQ2_RNAseqFQ1_RNAseqFQ2_tup_list': cell_list
            }
            with open(f'{output_path}_{sample}.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def main():
    parser = argparse.ArgumentParser(
        description='Generate Snakemake YAML config files for scONE-seq FastQ data'
    )
    parser.add_argument(
        '--fq-list', '-f',
        required=True,
        help='Path to the FastQ file list (one file per line)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output_configs',
        help='Output directory for YAML config files (default: output_configs)'
    )
    parser.add_argument(
        '--scWGS_scRNA_prefix',
        default='/stor/zxf/cnv/cnvguider-sc-rna/data/scONE-seq-data-processing/fq-symlink-dir/umidedup/dedup_fastq',
        help='Common prefix path for FastQ files'
    )
    parser.add_argument(
        '--combined-output',
        default=None,
        help='Output single combined config file (optional)'
    )

    args = parser.parse_args()

    print(f"Parsing FastQ file list: {args.fq_list}")
    fq_data, fq_prefix_path = parse_fastq_file_list(args.fq_list)

    # Summary statistics
    dna_samples = list(fq_data['dna'].keys())
    rna_samples = list(fq_data['rna'].keys())

    print(f"\nSummary:")
    print(f"  DNA samples: n={len(dna_samples)} ({dna_samples})")
    print(f"  RNA samples: n={len(rna_samples)} ({rna_samples})")

    dna_total = sum(len(f['R1']) + len(f['R2']) for f in fq_data['dna'].values())
    rna_total = sum(len(f['R1']) + len(f['R2']) for f in fq_data['rna'].values())

    print(f"  Total DNA FastQ files: {dna_total}")
    print(f"  Total RNA FastQ files: {rna_total}")

    celltype2samplename2record = match_dna_rna_by_sample(fq_data)
    print(f"\nGenerating combined config: {args.output_dir}")
    generate_config_yaml(celltype2samplename2record, args.output_dir, fq_prefix_path)

if __name__ == '__main__':
    main()

