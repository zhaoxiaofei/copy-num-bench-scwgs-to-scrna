#!/usr/bin/env python3
"""Synthetic end-to-end test for stat_tests.py (scRNA-seq CNV-caller benchmark).

Part A  Library API, default material clusters: 12 datasets built from 7
        materials (patient chips, one cell line across technologies and
        reference-cell configurations, cell-line mixing experiments), 4
        methods, 2 per-cell metrics + 1 dataset-level classification metric,
        with material-shared method effects - the correlation structure the
        real benchmark has.  Verifies the pairwise TSV (incl. the 95% CI of
        the rank-biserial effect size r), the Friedman levels, the JSON
        cluster record and the booktabs LaTeX table with EXACTLY n, p, r,
        95% CI per row.
Part B  Sensitivity modes: --cluster-key dataset (12 units) and
        --cluster-key none (naive, flagged).
Part C  MATERIAL_DATASETS table checks (primary sample -> derived dataset
        names; raw and display spellings; unknown datasets kept separate).
Part D  CLI round trip on a synthetic results/ tree of evaluation TSVs:
        stat_tests.py -i ... -o ... and --latex-table, plus the
        plot_cnv_heatmaps.py integration (stats tables written by default,
        --no_stats skips them, a bogus reference warns without breaking the
        figures).
Part E  winner_analysis.py (ordering / unique-winner / top-2 tests) on a
        20-material / 5-method synthetic benchmark with three planted
        ground truths: a metric with a UNIQUE WINNER, a metric with a TOP-2
        group (two indistinguishable leaders ahead of the rest), and a null
        metric with NO separation; plus a lower-is-better metric (runtime)
        whose ordering must flip.  Verifies the verdicts, the order/stepdown
        TSVs, the JSON consensus, the CLI round trip, the plot_cnv_heatmaps
        integration (winner tables by default, with a bogus --stats_reference,
        and skipped by --no_winner_analysis / --no_stats) and the booktabs
        step-down LaTeX table with EXACTLY n, p, r, 95% CI per row.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
OUT = os.path.join(REPO, 'work_test_out')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import stat_tests  # noqa: E402

rng = np.random.default_rng(7)

# ------------------------------------------------------------------ Part A --
method_effect = {'infercnv': 0.05, 'copykat': 0.0, 'casper': -0.06, 'scevan': 0.02}
datasets = {
    'BCIS106T_chip1_SAMN48409192_SRR33511671': 'BCIS106T',
    'BCIS106T_chip2_SAMN48409193_SRR33511670': 'BCIS106T',
    'ECIS44T_chip1_SAMN40389277_SRR28357495': 'ECIS44T',
    'ECIS44T_chip2_SAMN40389278_SRR28357494': 'ECIS44T',
    'MDA231_chip1_SAMN40389268_SRR33482482': 'MDA231',
    'scONE-seq_HCT116': 'HCT116',
    'scONE-seq_HCT116_HUVEC_H9_as_T_N': 'HCT116',
    'scONE-seq_NPC43': 'NPC43',
    'scONE-seq_NPC43_HUVEC_H9_as_T_N': 'NPC43',
    'HCT116_PRJNA603321': 'HCT116',
    'Cellline_mixing_experiment_1_1_SAMN48409183_SRR33511680': 'CM1_1',
    'Cellline_mixing_experiment_2_SAMN48409185_SRR33511678': 'CM2',
}
metrics = ['Pearson Correlation Coefficient', 'CopyNumber gain ROC-AUC',
           'Tumor/normal classification accuracy (scRNA vs scWGS)']
mat_shock = {m: rng.normal(0, 0.06) for m in set(datasets.values())}
rows = []
for ds, mat in datasets.items():
    for meth, eff in method_effect.items():
        val = float(np.clip(0.8 + mat_shock[mat] + eff + rng.normal(0, 0.05), 0, 1))
        rows.append({'dataset': ds, 'method': meth, 'metric': metrics[2],
                     'value': val, 'ground_truth_cell': np.nan})
        for c in range(30):
            cell_shock = rng.normal(0, 0.03)
            for mname in metrics[:2]:
                v = float(np.clip(0.7 + mat_shock[mat] + eff + cell_shock
                                  + rng.normal(0, 0.04), 0, 1))
                rows.append({'dataset': ds, 'method': meth, 'metric': mname,
                             'value': v, 'ground_truth_cell': F'{ds}_cell{c}'})
df = pd.DataFrame(rows)
print(F'Part A: synthetic frame {len(df)} rows, {df["dataset"].nunique()} datasets, '
      F'{df["method"].nunique()} methods, {len(metrics)} metrics')

prefix = os.path.join(OUT, 'figA')
settings = stat_tests.run_scrna_benchmark_stats(
    df, prefix, reference='infercnv', n_resamples=400, seed=2)
assert settings is not None
pw = pd.read_csv(prefix + '.pairwise.tsv', sep='\t')
fr = pd.read_csv(prefix + '.friedman.tsv', sep='\t')
need_cols = ['inference_level', 'cluster_key', 'n_clusters', 'n_units_paired',
             'n_pairs', 'pvalue_two_sided', 'pvalue_sign_test', 'pvalue_holm',
             'rank_biserial_r', 'ci95_r_low', 'ci95_r_high', 'ci_r_method',
             'cl_effect_paired', 'ci95_median_diff_low', 'ci95_median_diff_high',
             'pvalue_unit_naive', 'rank_biserial_r_unit_naive',
             'icc_within_cluster_d', 'design_effect', 'n_effective_units',
             'p_inflation_ratio']
missing = [c for c in need_cols if c not in pw.columns]
assert not missing, F'pairwise TSV missing columns: {missing}'
# 7 materials: BCIS106T, ECIS44T, MDA231, HCT116, NPC43, CM1_1, CM2
assert pw['n_clusters'].eq(7).all(), pw['n_clusters'].unique()
assert set(pw['inference_level']) == {'cluster'}
assert pw['ci95_r_low'].notna().all() and pw['ci95_r_high'].notna().all()
assert pw['ci95_r_low'].between(-1.01, 1.01).all()
assert (pw['ci95_r_low'] <= pw['ci95_r_high']).all()
_brackets = ((pw['ci95_r_low'] <= pw['rank_biserial_r'])
             & (pw['rank_biserial_r'] <= pw['ci95_r_high']))
assert _brackets.mean() >= 0.5, F'CI of r brackets the estimate only in {_brackets.mean() * 100:.0f}% of rows'
# direction: casper (negative effect) -> positive r (reference better)
assert (pw.loc[pw['method_b'] == 'casper', 'rank_biserial_r'] > 0).all()
assert fr['level'].eq('cluster').any() and fr['level'].eq('unit (naive)').any()
with open(prefix + '.json') as fh:
    js = json.load(fh)
assert js['independence']['cluster_key'] == ['material']
assert js['independence']['n_clusters'] == 7
print(F'OK Part A: {len(pw)} pairwise rows, n=7 materials, CI of r present '
      F'({_brackets.mean() * 100:.0f}% bracketed)')

tex = prefix + '.pairwise.tex'
assert os.path.isfile(tex)
text = open(tex).read()
assert '$n$ & $p$ & $r$ & 95\\% CI' in text
assert 'Median diff' not in text and 'Paired $n$' not in text
body = [ln for ln in text.splitlines()
        if ln.startswith('    ') and ln.rstrip().endswith('\\')
        and '$n$ & $p$' not in ln and 'toprule' not in ln
        and 'midrule' not in ln and 'bottomrule' not in ln]
assert body and all(ln.count('&') == 5 for ln in body), \
    'rows must have exactly 6 cells (metric, method_b, n, p, r, CI)'
assert len(re.findall(r'\[-?\d+\.\d{2}, -?\d+\.\d{2}\]', text)) >= len(body) - 1
assert 'CopyKAT 2021' in text and 'CaSpER 2020' in text
print(F'OK Part A: LaTeX table = exactly n, p, r, CI ({len(body)} rows)')

# ------------------------------------------------------------------ Part B --
stat_tests.run_scrna_benchmark_stats(df, prefix + '.naive', reference='infercnv',
                                     cluster_key_cols=[], n_resamples=300, seed=2)
pwn = pd.read_csv(prefix + '.naive.pairwise.tsv', sep='\t')
assert set(pwn['inference_level']) == {'unit (naive)'}
assert pwn['note'].str.contains('NAIVE').any()
percell = pwn[pwn['metric'] != metrics[2]]
perds = pwn[pwn['metric'] == metrics[2]]
assert percell['n_pairs'].eq(360).all()          # 12 datasets x 30 paired cells
assert perds['n_pairs'].eq(12).all()             # one pair per dataset
print('OK Part B: naive mode flagged and complete')

stat_tests.run_scrna_benchmark_stats(df, prefix + '.ds', reference='infercnv',
                                     cluster_key_cols=['dataset'],
                                     n_resamples=300, seed=2)
pwd = pd.read_csv(prefix + '.ds.pairwise.tsv', sep='\t')
assert pwd['n_clusters'].eq(12).all()
print('OK Part B: dataset-level sensitivity mode (n=12)')

# ------------------------------------------------------------------ Part C --
cases = {
    # raw folder names (mapped through dataset_materials.DATASET_NAME_ALIASES)
    'BCIS106T_chip1_SAMN48409192_SRR33511671': 'BCIS106T',
    'ECIS44T_chip3_SAMN40389279_SRR28357493': 'ECIS44T',
    'MDA231_chip2_SAMN40389269_SRR33482483': 'MDA231',
    'scONE-seq_HCT116': 'HCT116',
    'HCT116_PRJNA603321': 'HCT116',
    'wellDR3_SAMN48409182_SRR33511681': 'wellDR3',
    'A375_HCA00102_PRJNA603321': 'A375',
    # display dataset names (as written by plot_cnv_heatmaps.py): every
    # spelling of one primary sample must map to that primary sample
    'DNTR-seq_PRJNA603321_HCT116_noRefCells': 'HCT116',
    'scONE-seq_PRJNA768428_HCT116_noRefCells': 'HCT116',
    'scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef': 'HCT116',
    'scONE-seq_PRJNA768428_NPC43_noRefCells': 'NPC43',
    'scONE-seq_PRJNA768428_NPC43_asTumor_HUVEC-H9_asRef': 'NPC43',
    'scONE-seq_HUVEC_H9': 'HUVEC_H9',
    'wellDR-seq_ECIS44T_chip4_SRR28357492_tumorNormalMix': 'ECIS44T',
    'wellDR-seq_Cellline_mixing_experiment_1_1_SRR33511680_noRefCells':
        'Cellline_mixing_experiment_1_1',
    'wellDR-seq_Cellline_mixing_experiment_2_SRR33511678_noRefCells':
        'Cellline_mixing_experiment_2',
    'wellDR-seq_BCIS106T_chip2_SRR33511670_tumorNormalMix$^B$': 'BCIS106T',
}
for ds, want in cases.items():
    assert stat_tests.material_from_dataset(ds) == want, ds

# The table itself: 21 primary samples covering the 41 benchmark datasets.
from dataset_materials import DATASET_MATERIAL, MATERIAL_DATASETS  # noqa: E402
assert len(MATERIAL_DATASETS) == 21, len(MATERIAL_DATASETS)
assert len(DATASET_MATERIAL) == 41, len(DATASET_MATERIAL)
assert len(MATERIAL_DATASETS['HCT116']) == 3
assert len(MATERIAL_DATASETS['NPC43']) == 2
assert len(MATERIAL_DATASETS['ECIS44T']) == 5
assert len(MATERIAL_DATASETS['ECIS25T']) == 3

# A dataset that is not listed is kept as its own material and reported (the
# conservative choice), never silently merged into another material.
unknown = 'wellDR-seq_NEWDATASET_SRR99999999_tumorNormalMix'
assert stat_tests.material_from_dataset(unknown) == unknown
assert stat_tests.material_from_dataset('DNTR-seq_PRJNA603321_HCT116_noRefCells') == 'HCT116'
print('OK Part C: MATERIAL_DATASETS table (21 primary samples / 41 datasets), '
      'raw+display spellings, unknown datasets kept separate')

# Part C.2: the same check end-to-end on the real names - 7 datasets but only
# 3 independent materials (HCT116 x3, NPC43 x2, ECIS44T x2 chips).
real_cases = [
    'DNTR-seq_PRJNA603321_HCT116_noRefCells',
    'scONE-seq_PRJNA768428_HCT116_noRefCells',
    'scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef',
    'scONE-seq_PRJNA768428_NPC43_noRefCells',
    'scONE-seq_PRJNA768428_NPC43_asTumor_HUVEC-H9_asRef',
    'wellDR-seq_ECIS44T_chip1_SRR28357495_tumorNormalMix',
    'wellDR-seq_ECIS44T_chip2_SRR28357494_tumorNormalMix',
]
rows_real = []
for ds in real_cases:
    for meth, eff in method_effect.items():
        for c in range(6):
            rows_real.append({
                'dataset': ds, 'method': meth,
                'metric': 'Pearson Correlation Coefficient',
                'value': float(np.clip(0.7 + eff + rng.normal(0, 0.04), 0, 1)),
                'ground_truth_cell': F'{ds}_cell{c}'})
df_real = pd.DataFrame(rows_real)
prefix_real = os.path.join(OUT, 'figC2')
stat_tests.run_scrna_benchmark_stats(
    df_real, prefix_real, reference='infercnv', n_resamples=200, seed=3)
pw_real = pd.read_csv(prefix_real + '.pairwise.tsv', sep='\t')
assert pw_real['n_clusters'].eq(3).all(), pw_real['n_clusters'].unique()
with open(prefix_real + '.json') as fh:
    js_real = json.load(fh)
assert js_real['independence']['n_clusters'] == 3
print('OK Part C.2: 7 real cell-line datasets -> 3 materials (HCT116, NPC43, '
      'ECIS44T) in the end-to-end tests')

# Part C.3: a metric on which the reference method has no evaluable value
# (infercnv on the scWGS-diploid-fraction metric in the real data) must not
# abort the pairwise table: only that metric's reference-vs-rest family is
# skipped and recorded; its Friedman row and all other metrics stay.
df_nanref = df.copy()
skip_metric = metrics[1]
df_nanref.loc[(df_nanref['metric'] == skip_metric)
              & (df_nanref['method'] == 'infercnv'), 'value'] = np.nan
prefix_nanref = os.path.join(OUT, 'figC3')
stat_tests.run_scrna_benchmark_stats(
    df_nanref, prefix_nanref, reference='infercnv', n_resamples=200, seed=4)
pw_nr = pd.read_csv(prefix_nanref + '.pairwise.tsv', sep='\t')
assert skip_metric not in set(pw_nr['metric'])
fr_nr = pd.read_csv(prefix_nanref + '.friedman.tsv', sep='\t')
assert fr_nr['metric'].eq(skip_metric).any()
with open(prefix_nanref + '.json') as fh:
    js_nr = json.load(fh)
assert js_nr['metrics_skipped_no_reference'] == [skip_metric]
print('OK Part C.3: metric without reference values is skipped, not fatal')

# ------------------------------------------------------------------ Part D --
TREE = os.path.join(OUT, 'results')
if os.path.exists(TREE):
    shutil.rmtree(TREE)
small = {'BCIS106T_chip1_SAMN48409192_SRR33511671': 'BCIS106T',
         'BCIS106T_chip2_SAMN48409193_SRR33511670': 'BCIS106T',
         'ECIS44T_chip1_SAMN40389277_SRR28357495': 'ECIS44T',
         'MDA231_chip1_SAMN40389268_SRR33482482': 'MDA231'}
for ds, mat in small.items():
    evaldir = os.path.join(TREE, F'{ds}_solo-genefull_output', 'evaluation')
    os.makedirs(evaldir, exist_ok=True)
    for meth, eff in method_effect.items():
        erows = []
        for c in range(20):
            cell_shock = rng.normal(0, 0.03)
            for m in metrics[:2]:
                v = float(np.clip(0.75 + mat_shock[mat] + eff + cell_shock
                                  + rng.normal(0, 0.04), 0, 1))
                erows.append({'caller': meth, 'ground_truth_cell': F'{ds}_cell{c}',
                              'caller_cell': F'{ds}_cell{c}',
                              'celltype': 'tumor' if c % 3 else 'reference',
                              'celltype_dna': 'tumor' if c % 3 else 'reference',
                              'metric': m, 'value': v})
        pd.DataFrame(erows).to_csv(
            os.path.join(evaldir, F'bench_{meth}_without_preclassified_cells.tsv'),
            sep='\t', index=False)
        clf = [{'caller': meth, 'comparison': 'rna_vs_dna', 'metric': 'accuracy',
                'value': float(np.clip(0.8 + mat_shock[mat] + eff / 2
                                       + rng.normal(0, 0.05), 0, 1))},
               {'caller': meth, 'comparison': 'rna_score_vs_dna',
                'metric': 'aneuploidy_score_auroc',
                'value': float(np.clip(0.85 + mat_shock[mat] + eff / 2
                                       + rng.normal(0, 0.05), 0, 1))}]
        pd.DataFrame(clf).to_csv(
            os.path.join(evaldir, F'bench_{meth}.cell_classification.tsv'),
            sep='\t', index=False)

cli_prefix = os.path.join(OUT, 'figD')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'stat_tests.py'),
                      '-i', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '-o', cli_prefix, '--reference', 'infercnv',
                      '--boot', '300'], capture_output=True, text=True, cwd=HERE)
print('Part D CLI exit code:', ret.returncode)
if ret.returncode != 0:
    print(ret.stdout[-2000:]); print(ret.stderr[-2000:]); sys.exit(1)
pwd_cli = pd.read_csv(cli_prefix + '.pairwise.tsv', sep='\t')
assert pwd_cli['n_clusters'].eq(3).all()    # BCIS106T, ECIS44T, MDA231
assert os.path.isfile(cli_prefix + '.pairwise.tex')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'stat_tests.py'),
                      '-o', cli_prefix, '--latex-table', '--reference', 'infercnv'],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0 and '$n$ & $p$ & $r$ & 95\\% CI' in ret.stdout
print('OK Part D: CLI round trip (loader -> stats -> LaTeX table)')

# plot_cnv_heatmaps.py integration
hm_out = os.path.join(OUT, 'hmD')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'plot_cnv_heatmaps.py'),
                      '--input_glob', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '--outdir', hm_out, '--dpi', '72', '--stats_boot', '300',
                      '--no_normal_cells_glob', ''],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, ret.stderr[-2000:]
for f in ['stats.pairwise.tsv', 'stats.friedman.tsv', 'stats.json',
          'stats.pairwise.tex', 'winner.order.tsv', 'winner.stepdown.tsv',
          'winner.stepdown.tex', 'winner.json', 'aggregated_results.tsv']:
    assert os.path.isfile(os.path.join(hm_out, f)), F'MISSING {f}'
hm_pw = pd.read_csv(os.path.join(hm_out, 'stats.pairwise.tsv'), sep='\t')
assert hm_pw['n_clusters'].eq(3).all()
hm_tex = open(os.path.join(hm_out, 'stats.pairwise.tex')).read()
assert '$n$ & $p$ & $r$ & 95\\% CI' in hm_tex
hm_win_tex = open(os.path.join(hm_out, 'winner.stepdown.tex')).read()
assert '$n$ & $p$ & $r$ & 95\\% CI' in hm_win_tex

hm_out2 = os.path.join(OUT, 'hmD_nostats')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'plot_cnv_heatmaps.py'),
                      '--input_glob', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '--outdir', hm_out2, '--dpi', '72', '--no_stats',
                      '--no_normal_cells_glob', ''],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0
assert not os.path.exists(os.path.join(hm_out2, 'stats.pairwise.tsv'))
assert not os.path.exists(os.path.join(hm_out2, 'winner.stepdown.tsv'))
assert os.path.isfile(os.path.join(hm_out2, 'aggregated_results.tsv'))

hm_out3 = os.path.join(OUT, 'hmD_badref')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'plot_cnv_heatmaps.py'),
                      '--input_glob', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '--outdir', hm_out3, '--dpi', '72',
                      '--stats_reference', 'no_such_method',
                      '--no_normal_cells_glob', ''],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0 and '[stats] skipped' in ret.stdout
assert os.path.isfile(os.path.join(hm_out3, 'aggregated_results.tsv'))
# the winner analysis needs NO reference method: it must still have run
for f in ['winner.order.tsv', 'winner.stepdown.tsv', 'winner.stepdown.tex',
          'winner.json']:
    assert os.path.isfile(os.path.join(hm_out3, f)), F'MISSING {f} (bogus-reference run)'
print('OK Part D: plot_cnv_heatmaps.py integration (default on, --no_stats, '
      'bogus reference warns but the winner analysis still runs)')

# ------------------------------------------------------------------ Part E --
import winner_analysis  # noqa: E402

rngE = np.random.default_rng(11)
methodsE = ['infercnv', 'copykat', 'casper', 'scevan', 'conicsmat']
# 20 primary samples of the real benchmark, addressed by their real derived
# dataset names, so the default MATERIAL_DATASETS clustering applies to this
# synthetic frame (and to the CLI subprocesses that re-read it).
matsE = [m for m in sorted(MATERIAL_DATASETS) if m != 'HUVEC_H9']
assert len(matsE) == 20, matsE
eff_win = {'infercnv': 0.30, 'copykat': 0.18, 'casper': 0.10,
           'scevan': 0.04, 'conicsmat': 0.00}
eff_top2 = {'infercnv': 0.20, 'copykat': 0.20, 'casper': 0.06,
            'scevan': 0.03, 'conicsmat': 0.00}
rowsE = []
for mat in matsE:
    shock = rngE.normal(0, 0.03)
    for ds in MATERIAL_DATASETS[mat]:
        for meth in methodsE:
            v_win = 0.20 + eff_win[meth] + shock + rngE.normal(0, 0.05)
            v_top2 = 0.20 + eff_top2[meth] + shock + rngE.normal(0, 0.05)
            v_flat = 0.50 + shock + rngE.normal(0, 0.08)
            v_time = 100.0 * (0.32 - eff_win[meth]) + rngE.normal(0, 4.0)
            for c in range(8):
                cell = F'{ds}_c{c}'
                rowsE.append({'dataset': ds, 'method': meth, 'metric': 'm_win',
                              'value': float(np.clip(v_win + rngE.normal(0, 0.01), 0, 1)),
                              'ground_truth_cell': cell})
                rowsE.append({'dataset': ds, 'method': meth, 'metric': 'm_top2',
                              'value': float(np.clip(v_top2 + rngE.normal(0, 0.01), 0, 1)),
                              'ground_truth_cell': cell})
                rowsE.append({'dataset': ds, 'method': meth, 'metric': 'm_flat',
                              'value': float(np.clip(v_flat + rngE.normal(0, 0.01), 0, 1)),
                              'ground_truth_cell': cell})
                rowsE.append({'dataset': ds, 'method': meth, 'metric': 'm_runtime',
                              'value': max(1.0, v_time + rngE.normal(0, 1.0)),
                              'ground_truth_cell': cell})
dfE = pd.DataFrame(rowsE)
print(F'Part E: synthetic frame {len(dfE)} rows, 20 primary samples (real '
      F'dataset names), '
      F'{len(methodsE)} methods, 4 metrics (incl. lower-is-better m_runtime)')

prefixE = os.path.join(OUT, 'figE')
sE = winner_analysis.run_scrna_winner_analysis(
    dfE, prefixE, n_resamples=300, stability_resamples=200, seed=3,
    metric_directions={'m_runtime': 'lower'})
assert sE is not None
verd = {v['metric']: v for v in sE['verdicts']}

# m_win: clean unique winner, strict top separation
assert verd['m_win']['unique_winner'] is True
assert verd['m_win']['unique_winner_method'] == 'infercnv'
assert verd['m_win']['leading_group'] == ['infercnv']
assert verd['m_win']['ordering_supported'] is True
assert verd['m_win']['leading_group_homogeneous'] is True

# m_top2: two indistinguishable leaders, both ahead of the rest
assert verd['m_top2']['unique_winner'] is False
assert verd['m_top2']['top2_exists'] is True
assert set(verd['m_top2']['top2']) == {'infercnv', 'copykat'}
assert verd['m_top2']['approximate_ordering_exists'] is True
assert verd['m_top2']['cross_boundary_min_r'] > 0

# m_flat: nothing separates the methods
assert verd['m_flat']['unique_winner'] is False
assert verd['m_flat']['top2_exists'] is False
assert verd['m_flat']['no_separation'] is True
assert verd['m_flat']['ordering_supported'] is False

# m_runtime: lower-is-better; the lowest-runtime method must rank first
assert verd['m_runtime']['direction'] == 'lower'
assert verd['m_runtime']['ordering'][0] == 'infercnv'
assert verd['m_runtime']['unique_winner'] is True

# step-down TSV: structure + the decisive rows
sd = pd.read_csv(prefixE + '.stepdown.tsv', sep='\t')
need_sd = ['metric', 'step', 'method_a', 'method_b', 'n', 'pvalue_holm',
           'rank_biserial_r', 'ci95_r_low', 'ci95_r_high', 'a_beats_b',
           'is_boundary', 'is_shown', 'leading_group_size', 'metric_verdict']
missing = [c for c in need_sd if c not in sd.columns]
assert not missing, F'stepdown TSV missing columns: {missing}'
assert sd['n'].eq(20).all()                       # 20 independent materials
assert sd['ci95_r_low'].notna().all() and sd['ci95_r_high'].notna().all()
assert (sd['ci95_r_low'] <= sd['ci95_r_high']).all()
top2_rows = sd[sd['metric'] == 'm_top2'].sort_values('step')
assert not top2_rows.iloc[0]['a_beats_b']        # M1 vs M2 not separable
assert bool(top2_rows.iloc[1]['is_boundary'])    # M2 vs M3 = the boundary
assert top2_rows['is_shown'].sum() == 2          # chain stops at the boundary
win_rows = sd[sd['metric'] == 'm_win']
assert win_rows['is_shown'].sum() == 1 and bool(win_rows.iloc[0]['is_boundary'])

# order.tsv: rank position + memberships
ordr = pd.read_csv(prefixE + '.order.tsv', sep='\t')
assert list(ordr.columns[:4]) == ['metric', 'direction', 'method', 'rank']
rt = ordr[ordr['metric'] == 'm_runtime']
assert rt.sort_values('rank').iloc[0]['method'] == 'infercnv'
assert ordr[ordr['metric'] == 'm_win'].sort_values('rank').iloc[0]['method'] == 'infercnv'

# JSON consensus
with open(prefixE + '.json') as fh:
    jsE = json.load(fh)
cons = jsE['consensus']
assert cons['n_metrics_analysed'] == 4
assert 'm_win' in cons['metrics_unique_winner']
assert 'm_runtime' in cons['metrics_unique_winner']
assert 'm_top2' in cons['metrics_top2']
assert 'm_flat' in cons['metrics_no_separation']
pm = cons['per_method']
assert pm['infercnv']['n_unique_winner'] >= 2
assert pm['copykat']['n_leading_group'] >= 2
print(F'OK Part E: verdicts correct (unique winner / top-2 / flat / lower-is-better); '
      F'consensus: {cons["summary"]}')

# LaTeX step-down table: EXACTLY n, p, r, 95% CI per row
texE = open(prefixE + '.stepdown.tex').read()
assert 'Metric & Method $a$ & Method $b$ & $n$ & $p$ & $r$ & 95\\% CI' in texE
bodyE = [ln for ln in texE.splitlines()
         if ln.startswith('    ') and ln.rstrip().endswith('\\\\')
         and 'toprule' not in ln and 'midrule' not in ln
         and 'bottomrule' not in ln and '$n$ &' not in ln]
assert bodyE and all(ln.count('&') == 6 for ln in bodyE), \
    'step-down rows must have exactly 7 cells (metric, a, b, n, p, r, CI)'
assert len(re.findall(r'\[-?\d+\.\d{2}, -?\d+\.\d{2}\]', texE)) >= len(bodyE) - 1
assert 'unique winner' in texE and 'top-2 group' in texE   # verdicts in the caption
print(F'OK Part E: LaTeX step-down table = exactly n, p, r, CI ({len(bodyE)} rows)')

# CLI round trip (loader -> winner analysis -> LaTeX table)
long_tsv = os.path.join(OUT, 'figE_long.tsv')
dfE.to_csv(long_tsv, sep='\t', index=False)
cliE = os.path.join(OUT, 'figEcli')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'winner_analysis.py'),
                      '-i', long_tsv, '-o', cliE, '--boot', '300',
                      '--stability-boot', '200', '--lower-is-better', 'm_runtime'],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, (ret.stdout[-1500:] + ret.stderr[-1500:])
assert os.path.isfile(cliE + '.stepdown.tsv')
assert os.path.isfile(cliE + '.stepdown.tex')
sd_cli = pd.read_csv(cliE + '.stepdown.tsv', sep='\t')
assert sd_cli['n'].eq(20).all()
rt_cli = sd_cli[sd_cli['metric'] == 'm_runtime'].sort_values('step')
assert rt_cli.iloc[0]['method_a'] == 'infercnv'  # best (lowest) runtime first
assert (rt_cli['method_a'] != rt_cli['method_b']).all()
ret = subprocess.run([sys.executable, os.path.join(HERE, 'winner_analysis.py'),
                      '-o', cliE, '--latex-table'],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0 and '$n$ & $p$ & $r$ & 95\\% CI' in ret.stdout
print('OK Part E: CLI round trip (loader -> winner analysis -> LaTeX table)')

# stat_tests.py CLI on the SAME unified long TSV (documented input kind; the
# loader must take it as-is instead of feeding it to the figure loader)
cliE_stats = os.path.join(OUT, 'figEstats')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'stat_tests.py'),
                      '-i', long_tsv, '-o', cliE_stats, '--reference', 'infercnv',
                      '--boot', '300'], capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, (ret.stdout[-1500:] + ret.stderr[-1500:])
pwE = pd.read_csv(cliE_stats + '.pairwise.tsv', sep='\t')
assert pwE['n_clusters'].eq(20).all()          # 20 independent materials
assert 'unified long TSV' in ret.stdout
print('OK Part E: stat_tests.py CLI accepts the unified long TSV (n=20 materials)')

# plot_cnv_heatmaps.py: --no_winner_analysis skips ONLY the winner part
hm_out4 = os.path.join(OUT, 'hmE_nowinner')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'plot_cnv_heatmaps.py'),
                      '--input_glob', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '--outdir', hm_out4, '--dpi', '72', '--stats_boot', '300',
                      '--no_winner_analysis', '--no_normal_cells_glob', ''],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, ret.stderr[-2000:]
assert os.path.isfile(os.path.join(hm_out4, 'stats.pairwise.tsv'))
assert not os.path.exists(os.path.join(hm_out4, 'winner.stepdown.tsv'))
print('OK Part E: plot_cnv_heatmaps.py --no_winner_analysis skips only the winner part')

# ------------------------------------------------------------------ Part F --
# [REV] Hsu's MCB (comparison with the best) - mcb.py.  The interval-based
# counterpart of the winner analysis, on the SAME planted frames of Part E:
# F.1 planted verdicts (unique winner / top-2 / flat / lower-is-better),
# F.2 CLI round trip on the unified long TSV,
# F.3 plot_cnv_heatmaps.py integration (mcb by default, reference-free, the
#     Fig. 5 source data written by the swarm grid),
# F.4 LaTeX regression: exactly n, p, r, 95% CI per row (+ tectonic compile).
import mcb  # noqa: E402

prefixF = os.path.join(OUT, 'figF')
sF = mcb.run_scrna_mcb(dfE, prefixF, n_resamples=300, seed=3,
                       lower_is_better=('m_runtime',))
assert sF is not None
verdF = {v['family']: v for v in sF['per_family_verdicts']}

# m_win: the planted unique winner is the only method not significantly inferior
assert verdF['m_win']['sample_best'] == 'infercnv'
assert verdF['m_win']['unique_winner'] is True
assert verdF['m_win']['leading_group'] == ['infercnv']
# m_top2: planted top-2 = the MCB leading group of size 2
assert verdF['m_top2']['unique_winner'] is False
assert verdF['m_top2']['leading_group_size'] == 2
assert set(verdF['m_top2']['leading_group']) == {'infercnv', 'copykat'}
# m_flat: nothing separates the methods (no one significantly inferior)
assert verdF['m_flat']['inferior'] == []
assert verdF['m_flat']['leading_group_size'] == len(methodsE)
# m_runtime (lower-is-better): the lowest runtime is the sample-best after negation
assert verdF['m_runtime']['sample_best'] == 'infercnv'
print(F'OK F.1: MCB verdicts match the plants (m_win unique winner {verdF["m_win"]["sample_best"]}; '
      F'm_top2 leading group {verdF["m_top2"]["leading_group"]}; m_flat g = '
      F'{verdF["m_flat"]["leading_group_size"]}; m_runtime best = '
      F'{verdF["m_runtime"]["sample_best"]})')

# TSV: exactly the four statistics plus the MCB quantities; n = 20 materials
mcb_tsv = pd.read_csv(prefixF + '.tsv', sep='\t')
need_mcb = ['n_units', 'gap_to_best', 'se_gap', 'mcb_low', 'mcb_high',
            'pvalue_mcb_one_sided', 'rank_biserial_r_vs_best',
            'ci95_r_low', 'ci95_r_high', 'significant_inferior',
            'family_leading_group', 'family_leading_group_size',
            'metric_orientation']
missing = [c for c in need_mcb if c not in mcb_tsv.columns]
assert not missing, F'MCB TSV missing columns: {missing}'
assert mcb_tsv['n_units'].eq(20).all()
assert mcb_tsv[mcb_tsv['metric'] == 'm_runtime']['metric_orientation'].str.contains(
    'lower-is-better').all()
# the leading group of the m_top2 family must be flagged on its member rows
top2_rows = mcb_tsv[mcb_tsv['metric'] == 'm_top2']
assert top2_rows['in_leading_group'].sum() == 2
assert (top2_rows['in_leading_group'].astype(bool)
        == top2_rows['method'].isin({'infercnv', 'copykat'})).all()
print(F'OK F.1b: MCB TSV complete ({len(mcb_tsv)} rows, n = 20 materials, '
      F'lower-is-better orientation recorded)')

# JSON consensus
with open(prefixF + '.json') as fh:
    jsF = json.load(fh)
assert jsF['consensus']['families_with_unique_winner'] >= 1
assert 'infercnv' in jsF['consensus']['methods_never_significantly_inferior']
assert jsF['lower_is_better'] == ['m_runtime']
print(F'OK F.1c: MCB JSON consensus (never inferior: '
      F'{jsF["consensus"]["methods_never_significantly_inferior"]})')

# F.2 - CLI round trip on the unified long TSV (loader -> MCB -> LaTeX)
cliF = os.path.join(OUT, 'figFcli')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'mcb.py'),
                      '-i', long_tsv, '-o', cliF, '--boot', '300',
                      '--lower-is-better', 'm_runtime'],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, (ret.stdout[-1500:] + ret.stderr[-1500:])
for f in [cliF + '.tsv', cliF + '.json', cliF + '.tex']:
    assert os.path.isfile(f), F'MISSING {f}'
sd_cliF = pd.read_csv(cliF + '.tsv', sep='\t')
assert sd_cliF['n_units'].eq(20).all()
ret = subprocess.run([sys.executable, os.path.join(HERE, 'mcb.py'),
                      '-o', cliF, '--latex-table'],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0 and '$n$ & $p$ & $r$ & 95\\% CI' in ret.stdout
print('OK F.2: MCB CLI round trip (loader -> MCB -> LaTeX table)')

# F.3 - plot_cnv_heatmaps.py integration (Part D ran it with the current code)
for f in ['mcb.tsv', 'mcb.tex', 'mcb.json', 'fig5_source_data.tsv',
          'fig5_source_data.meta.json']:
    assert os.path.isfile(os.path.join(hm_out, f)), F'MISSING {f} (default run)'
# --no_stats skips the MCB too (it is part of the statistical layer) but the
# Fig. 5 source data is still written (it belongs to the figure, not the tests)
assert not os.path.exists(os.path.join(hm_out2, 'mcb.tsv'))
assert os.path.isfile(os.path.join(hm_out2, 'fig5_source_data.tsv'))
# a bogus reference skips the pairwise table but NOT the reference-free MCB
assert os.path.isfile(os.path.join(hm_out3, 'mcb.tsv'))
# --no_mcb_analysis skips only the MCB part
hm_out5 = os.path.join(OUT, 'hmF_nomcb')
ret = subprocess.run([sys.executable, os.path.join(HERE, 'plot_cnv_heatmaps.py'),
                      '--input_glob', os.path.join(TREE, '*solo-genefull_output/evaluation/*.tsv'),
                      '--outdir', hm_out5, '--dpi', '72', '--stats_boot', '300',
                      '--no_mcb_analysis', '--no_normal_cells_glob', ''],
                     capture_output=True, text=True, cwd=HERE)
assert ret.returncode == 0, ret.stderr[-2000:]
assert os.path.isfile(os.path.join(hm_out5, 'stats.pairwise.tsv'))
assert os.path.isfile(os.path.join(hm_out5, 'winner.stepdown.tsv'))
assert not os.path.exists(os.path.join(hm_out5, 'mcb.tsv'))
src5 = pd.read_csv(os.path.join(hm_out, 'fig5_source_data.tsv'), sep='\t')
assert {'dataset', 'method', 'metric', 'mean', 'purity_bin', 'technology'} \
    == set(src5.columns)
print(F'OK F.3: plot_cnv_heatmaps.py integration (MCB default-on, skipped by '
      F'--no_stats / --no_mcb_analysis, reference-free; Fig. 5 source data '
      F'{len(src5)} rows)')

# F.4 - LaTeX regression: exactly n, p, r, 95% CI per row (+ compile)
texF = open(prefixF + '.tex').read()
assert 'Method & $n$ & $p$ & $r$ & 95\\% CI' in texF
assert texF.count('multicolumn{5}{l}') == 4          # one section per metric
bodyF = [ln for ln in texF.splitlines()
         if ln.startswith('    ') and ln.rstrip().endswith('\\\\')
         and 'multicolumn' not in ln and '$n$ &' not in ln
         and 'toprule' not in ln and 'midrule' not in ln
         and 'bottomrule' not in ln]
assert bodyF and all(ln.count('&') == 4 for ln in bodyF), \
    'MCB rows must have exactly 5 cells (method, n, p, r, CI)'
if shutil.which('tectonic'):
    tex_dir = os.path.join(OUT, 'texcheckF')
    os.makedirs(tex_dir, exist_ok=True)
    shutil.copyfile(prefixF + '.tex', os.path.join(tex_dir, 'scrna_mcb.tex'))
    with open(os.path.join(tex_dir, 'wrap.tex'), 'w') as fh:
        fh.write('\\documentclass{article}\\usepackage{booktabs}\\begin{document}'
                 '\\input{scrna_mcb.tex}\\end{document}\n')
    ret = subprocess.run(['tectonic', 'wrap.tex'], cwd=tex_dir,
                         capture_output=True, text=True)
    assert ret.returncode == 0, ret.stderr[-1500:]
    print('OK F.4: MCB LaTeX table compiles with tectonic')
print(F'OK F.4: MCB LaTeX table = exactly n, p, r, CI ({len(bodyF)} rows)')

print('\nAll stat_tests + winner_analysis + mcb end-to-end tests PASSED')
