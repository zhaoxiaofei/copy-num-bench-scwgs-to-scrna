#!/usr/bin/env python3
"""mcb.py - Hsu's Multiple Comparison with the Best (MCB) for the scRNA-seq
CNV-caller benchmark (Fig. 5).

WHAT THIS MODULE ADDS
=====================
stat_tests.py answers, per metric, "does the REFERENCE method outperform
method b?" with exactly four statistics (n, p, r, 95% CI), and
winner_analysis.py answers the ordering / unique-winner / top-2 questions with
a Wilcoxon step-down procedure.  This module adds the classical interval view
of the same questions - Hsu's Multiple Comparison with the Best: for every
method configuration i it builds a SIMULTANEOUS confidence interval for

    theta_i - max_{j != i} theta_j        ("method i versus the best of the
                                           others"; larger = better)

so that, with family-wise confidence 1 - alpha over the whole metric family:

    * interval entirely BELOW 0   ->  i is significantly INFERIOR to the best
                                      (excluded from the leading group);
    * interval entirely ABOVE 0   ->  i is significantly BETTER than every
                                      other method: i is the UNIQUE WINNER
                                      (possible only for the sample-best
                                      method);
    * interval CONTAINS 0         ->  i is INDISTINGUISHABLE from the best;
                                      i belongs to the leading group (a group
                                      of size 2 is exactly the "top-2" of the
                                      manuscript's observations).

The MCB family is defined PER TASK (the Fig. 5 swarm-grid panel), PER METRIC
and PER CALLER (method configuration): one family per metric, rows = the
method configurations, units = the MATERIALS (patients / cell lines), exactly
the independent experimental unit of stat_tests.py and winner_analysis.py
(datasets sharing a material collapse onto one unit; reference-cell
configuration variants collapse onto their base material).

DESIGN (blocked / repeated-measures MCB, cluster-robust)
=======================================================
All k method configurations are evaluated on the same cells of the same
datasets, so performance is paired (blocked) by cell.  Inference runs at the
MATERIAL level, exactly as in the sibling modules: per-unit medians of the
per-cell values first, MCB on the resulting n x k complete-block matrix
afterwards (units with any missing method are dropped and counted; methods
missing everywhere are dropped from the family).  For the complete-block
matrix X (n units x k methods, higher = better - metrics named in
--lower-is-better are negated first):

1.  Point estimates: mu_i = mean over units; the "best competitor" of i is
    j*(i) = argmax_{j != i} mu_j and the gap is D_i = mu_i - mu_{j*(i)}.
2.  Per pair (i, j): paired per-unit differences d_ij with mean D_ij and
    standard error SE_ij = sd(d_ij) / sqrt(n).
3.  Simultaneous MCB intervals (PRIMARY, bootstrap): resample the n units
    with replacement B times (seeded, reproducible); T*_ij = (D*_ij -
    D_ij) / SE_ij; c_boot = (1 - alpha) quantile of max_{i<j} |T*_ij|; the
    MCB interval of method i is the Tukey-style projection
        L_i = min_{j != i} (D_ij - c_boot * SE_ij)
        U_i = min_{j != i} (D_ij + c_boot * SE_ij)
    For k = 2 this reduces exactly to the paired-t interval.
4.  MCB-adjusted one-sided P per method (the table's p): the single-step
    bootstrap max-|t| adjusted two-sided P of method i versus its best
    competitor, converted to the one-sided H1 "i is inferior to the best"
    (H0: theta_i >= max_{j != i} theta_j).  Small p = significantly
    inferior; p >= 0.5 for the sample-best method by construction.
5.  Effect size (the table's r): matched-pairs rank-biserial correlation of
    the per-unit differences d_{i,j*(i)} against the best competitor
    (positive = i tends to score higher), with a per-method percentile
    bootstrap 95% CI (descriptive; the simultaneous quantity of this
    analysis is the MCB interval of the gap, which is the CI shown in the
    tables).  The rank-biserial comes from stat_tests.rank_biserial_matched
    so this module can never disagree with the pairwise tables.
6.  Parametric cross-check: c_par = q_{k, n-1}(1 - alpha) / sqrt(2), the
    studentized-range critical value scaled so that k = 2 reproduces the
    paired-t critical value exactly.
7.  Omnibus: the Friedman test (stat_tests.friedman_test) on the same
    complete-block matrix - "does any ordering of the methods exist?" -
    with Kendall's W; the MCB verdicts (unique winner / leading group of
    size g / inferior set) then describe the ordering's separability, and
    they answer the manuscript's three observations directly: approximate
    ordering (Friedman supported + partial separability), no unique winner
    (g >= 2), top-2 exists (g == 2).

LATeX DISCIPLINE (same as every other table of this project)
============================================================
Every table row carries, after its identity columns, EXACTLY the four
statistics n (effective sample size = number of independent units =
materials), p (MCB-adjusted one-sided P), r (rank-biserial effect size
versus the best competitor) and the 95% CI (the simultaneous MCB interval
of the gap to the best, on the metric's own scale) - nothing else.

CALIBRATION NOTE (regression-tested in test_stat_tests.py, Part F.1b)
=====================================================================
The bootstrap max-|t| calibration is a TRUE studentized bootstrap (every
replicate recomputes its own per-pair standard error), so the family-wise
error of the MCB verdicts under a true null is: at or below the nominal
level for the benchmark's positively-correlated within-unit structure at
n = 9 donors (conservative), near-nominal at n ~ 40 materials, and mildly
liberal (up to ~10% at a nominal 5%) in the worst case of independent
within-unit values at moderate n - the price of the nonparametric,
selection-aware calibration.  For k = 2 the interval reduces exactly to the
paired-t interval (up to the bootstrap Monte Carlo error).

Outputs (prefix = -o/--output)
==============================
    <prefix>.tsv     one row per (metric, method)
    <prefix>.tex     booktabs table: one section per metric, rows = methods
    <prefix>.json    settings, per-metric verdicts, consensus across metrics

Usage
=====
Standalone, on the raw evaluation TSVs (dataset inferred from each file's
path exactly like plot_cnv_heatmaps.py / stat_tests.py):
    python mcb.py -i 'results/*/evaluation/*.tsv' -o heatmaps/mcb
On one unified long TSV:
    python mcb.py -i aggregated_per_cell_long.tsv -o heatmaps/mcb
Re-print the LaTeX table from an existing run:
    python mcb.py -o heatmaps/mcb --latex-table
plot_cnv_heatmaps.py runs this module by default next to the pairwise tests
and the winner analysis (--no_mcb_analysis skips it); it needs no reference
method, so it also runs when --stats_reference is bogus.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats as sps
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - the frozen env always has scipy
    _HAVE_SCIPY = False

# Everything numerical and every display helper is reused from the validated
# sibling module so the pairwise tables, the winner analysis and MCB can never
# disagree on a shared statistic or label.
import stat_tests as _ST

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s')

MAX_BOOTSTRAP = _ST.MAX_CLUSTER_BOOTSTRAP   # runtime cap for the unit bootstrap
DEFAULT_N_RESAMPLES = 2000                  # MCB bootstrap resamples (seeded)


# --------------------------------------------------------------------------- #
# Formatting helpers (delegate to the stat_tests table style)                 #
# --------------------------------------------------------------------------- #
def _tex_escape(s):
    return _ST._tex_escape(s)


def _fmt_effect(x, decimals=2):
    return _ST._fmt_effect(x, decimals=decimals)


def _fmt_ci(low, high, decimals=2):
    return _ST._fmt_ci(low, high, decimals=decimals)


def _fmt_pvalue_mcb(x, alpha=0.05):
    """p column: stat_tests' Holm-p style (scientific for tiny, bold if <= alpha)."""
    return _ST._fmt_pvalue_holm(x, alpha=alpha)


def _fmt_signed_int(x):
    """n column: plain integer; missing -> '--'."""
    if x is None or not np.isfinite(x):
        return '--'
    return F'{int(x)}'


# --------------------------------------------------------------------------- #
# Core: Hsu's MCB on a complete-block unit x method matrix                    #

# --------------------------------------------------------------------------- #
# Core: Hsu's MCB on a complete-block unit x method matrix                    #
# --------------------------------------------------------------------------- #

def mcb_analyse(mat, alpha=0.05, n_resamples=DEFAULT_N_RESAMPLES, seed=1):
    """MCB analysis of one family.

    mat: DataFrame, rows = independent units, columns = methods, values
    oriented so that LARGER IS BETTER.  Rows with any missing value are
    dropped (complete blocks); columns that are entirely missing are
    dropped.  Returns a dict:

        {'n_units', 'k_methods', 'units_dropped', 'methods_dropped',
         'friedman' (dict|None), 'c_boot', 'c_par',
         'rows': [per-method record dict], 'verdict': {...}, 'notes': [...]}

    or None when the family is not analysable (fewer than 2 units or
    fewer than 2 methods after the complete-block filter).
    """
    if not _HAVE_SCIPY:
        logging.warning('scipy is not available: MCB analysis skipped')
        return None
    X = mat.copy()
    methods = list(X.columns)
    notes = []
    all_missing = [m for m in methods if not X[m].notna().any()]
    if all_missing:
        X = X.drop(columns=all_missing)
        notes.append(F'methods with no values dropped: {", ".join(map(str, all_missing))}')
    methods = list(X.columns)
    before = len(X)
    X = X.dropna(axis=0, how='any')
    units_dropped = before - len(X)
    if units_dropped:
        notes.append(F'{units_dropped} of {before} units dropped (missing at least '
                     F'one method; complete blocks only)')
    if len(X) < 2 or len(methods) < 2:
        notes.append(F'not analysable: {len(X)} complete units x {len(methods)} methods')
        return None
    X = X.astype(float)
    n, k = X.shape
    vals = X.to_numpy(dtype=float)
    mu = vals.mean(axis=0)
    sd = vals.std(axis=0, ddof=1) if n > 1 else np.zeros(k)

    # ---- per-pair paired-difference statistics ------------------------------
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    D = np.zeros((k, k))       # D[i, j] = mean(x_i - x_j); D[j, i] = -D[i, j]
    SE = np.zeros((k, k))
    DD = {}
    for i, j in pairs:
        d = vals[:, i] - vals[:, j]
        D[i, j] = d.mean()
        D[j, i] = -D[i, j]
        se = (d.std(ddof=1) / np.sqrt(n)) if (n > 1 and d.std(ddof=1) > 0) else 0.0
        SE[i, j] = SE[j, i] = se
        DD[(i, j)] = d

    # ---- best competitor, gap ------------------------------------------------
    jstar = np.zeros(k, dtype=int)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        jstar[i] = max(others, key=lambda j: mu[j])
    gap = np.array([D[i, jstar[i]] for i in range(k)])

    # ---- bootstrap max-|t| over all pairs (simultaneous calibration) --------
    # TRUE bootstrap-t: every replicate recomputes its own per-pair standard
    # error S* from the resampled differences, so T* = (D* - D)/S* has the
    # same studentized form as the observed statistics and the (1-alpha)
    # quantile of max|T*| calibrates the family of all pairwise t's INCLUDING
    # the denominator's sampling variability and the cross-pair selection (a
    # plug-in fixed-SE bootstrap would under-estimate the critical value and
    # reject too often - the calibration is regression-tested in
    # test_stat_tests.py Part F.1b).
    # D*_ij is the difference of the replicate means (mean of paired
    # differences = difference of means).
    B = int(min(max(int(n_resamples), 200), MAX_BOOTSTRAP))
    # the per-method r CI is DESCRIPTIVE (not the simultaneous quantity), so
    # its resamples are capped tighter than the calibration bootstrap to keep
    # the runtime bounded on large families (k(k-1)/2 pairs x k methods).
    r_B = int(min(B, 1000))
    rng = np.random.default_rng(seed)
    max_abs_t = np.empty(B)
    r_boot = {i: np.empty(r_B) for i in range(k)}   # per-method r resamples
    pair_idx = np.array(pairs)                    # (n_pairs, 2)
    D_pairs = D[pair_idx[:, 0], pair_idx[:, 1]] if len(pairs) else np.zeros(0)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vb = vals[idx]
        if len(pairs):
            dij = vb[:, pair_idx[:, 0]] - vb[:, pair_idx[:, 1]]   # n x n_pairs
            D_b = dij.mean(axis=0)
            SE_b = dij.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(pairs))
            ok = SE_b > 0
            t = np.zeros(len(pairs))
            t[ok] = (D_b[ok] - D_pairs[ok]) / SE_b[ok]
            max_abs_t[b] = float(np.max(np.abs(t)))
        else:
            max_abs_t[b] = 0.0
        if b < r_B:
            for i in range(k):
                r_boot[i][b] = _ST.rank_biserial_matched(vb[:, i] - vb[:, jstar[i]])
    c_boot = float(np.quantile(max_abs_t, 1.0 - alpha))

    # ---- MCB intervals: Tukey-style projection of the pairwise bands --------
    L = np.array([min(D[i, j] - c_boot * SE[i, j] for j in range(k) if j != i)
                  for i in range(k)])
    U = np.array([min(D[i, j] + c_boot * SE[i, j] for j in range(k) if j != i)
                  for i in range(k)])

    # ---- MCB-adjusted one-sided P per method ---------------------------------
    # single-step max-|t| adjusted two-sided P of i vs its best competitor,
    # converted to the one-sided H1 "i is inferior to the best of the others".
    p_one = np.ones(k)
    for i in range(k):
        se_i = SE[i, jstar[i]]
        t_i = (gap[i] / se_i) if se_i > 0 else 0.0
        p_two = float(np.mean(max_abs_t >= abs(t_i)))
        p_one[i] = (p_two / 2.0) if gap[i] < 0 else 1.0 - p_two / 2.0

    # ---- parametric cross-check (classical Hsu/Tukey, pooled scale) ----------
    # The classical construction assumes one common standard deviation of the
    # paired differences; the pooled estimate removes the per-pair-SE selection
    # that would otherwise make the cross-check anticonservative (the per-pair
    # maximum t is stochastically larger than the studentized range).  With
    # c_par = q_{k, n-1}/sqrt(2) the k = 2 case reproduces the paired-t
    # critical value exactly.
    if pairs:
        s_pool = float(np.sqrt(np.mean([np.var(DD[(i, j)], ddof=1) if n > 1
                                        else 0.0 for i, j in pairs])))
    else:
        s_pool = 0.0
    try:
        c_par = float(sps.studentized_range.ppf(1.0 - alpha, k, df=n - 1) / np.sqrt(2.0))
    except Exception:                                   # pragma: no cover
        c_par = float('nan')
    if np.isfinite(c_par) and s_pool > 0:
        se_pool = s_pool / np.sqrt(n)
        Lp = np.array([min(D[i, j] - c_par * se_pool for j in range(k) if j != i)
                       for i in range(k)])
        Up = np.array([min(D[i, j] + c_par * se_pool for j in range(k) if j != i)
                       for i in range(k)])
    else:
        Lp = np.full(k, np.nan)
        Up = np.full(k, np.nan)

    # ---- Friedman omnibus on the same complete blocks ------------------------
    fr = _ST.friedman_test([vals[:, j] for j in range(k)])
    mean_ranks = fr['mean_ranks'] if fr else [float('nan')] * k

    # ---- verdicts -------------------------------------------------------------
    inferior = [i for i in range(k) if U[i] < 0]
    leading = [i for i in range(k) if U[i] >= 0]
    order = sorted(range(k), key=lambda i: (-mu[i], methods[i]))
    best = order[0]
    unique_winner = bool(L[best] > 0 and len(leading) == 1)

    rows = []
    for i in range(k):
        d_vs_best = vals[:, i] - vals[:, jstar[i]]
        r_i = _ST.rank_biserial_matched(d_vs_best)
        r_lo, r_hi = (float(np.quantile(r_boot[i], alpha / 2.0)),
                      float(np.quantile(r_boot[i], 1.0 - alpha / 2.0)))
        rows.append({
            'method': str(methods[i]),
            'n_units': int(n),
            'mean': float(mu[i]),
            'sd_units': float(sd[i]),
            'friedman_mean_rank': float(mean_ranks[i]),
            'rank_position': int(sum(1 for j in range(k) if mu[j] > mu[i]) + 1),
            'best_competitor': str(methods[jstar[i]]),
            'gap_to_best': float(gap[i]),
            'se_gap': float(SE[i, jstar[i]]),
            'mcb_low': float(L[i]),
            'mcb_high': float(U[i]),
            'mcb_low_param': float(Lp[i]),
            'mcb_high_param': float(Up[i]),
            'c_boot': float(c_boot),
            'c_par': float(c_par),
            's_pool': float(s_pool),
            'pvalue_mcb_one_sided': float(p_one[i]),
            'pvalue_mcb_two_sided_adj': float(2.0 * min(p_one[i], 1.0 - p_one[i])),
            'rank_biserial_r_vs_best': float(r_i),
            'ci95_r_low': r_lo,
            'ci95_r_high': r_hi,
            'cl_effect_vs_best': _ST.common_language_paired(d_vs_best),
            'significant_inferior': bool(U[i] < 0),
            'unique_best': bool(L[i] > 0),
            'in_leading_group': bool(U[i] >= 0),
            'note': '',
        })
    for r in rows:
        if r['se_gap'] == 0.0:
            r['note'] = 'zero within-unit variance of the gap; interval and P are degenerate'

    verdict = {
        'family': None,     # filled by the caller
        'n_units': int(n),
        'k_methods': int(k),
        'friedman_pvalue': float(fr['pvalue']) if fr else float('nan'),
        'friedman_chi2': float(fr['chi2']) if fr else float('nan'),
        'kendalls_w': float(fr['kendalls_w']) if fr else float('nan'),
        'ordering_exists': bool(fr is not None and fr['pvalue'] <= alpha),
        'order_by_mean': [str(methods[i]) for i in order],
        'sample_best': str(methods[best]),
        'unique_winner': unique_winner,
        'leading_group': [str(methods[i]) for i in leading],
        'leading_group_size': int(len(leading)),
        'inferior': [str(methods[i]) for i in inferior],
        'top2_exists': bool(len(leading) == 2),
    }
    return {'n_units': int(n), 'k_methods': int(k), 'units_dropped': int(units_dropped),
            'methods_dropped': all_missing, 'friedman': fr, 'c_boot': c_boot,
            'c_par': c_par, 'rows': rows, 'verdict': verdict, 'notes': notes}




# --------------------------------------------------------------------------- #
# Family builders: the SAME material x method matrices the sibling modules use #
# --------------------------------------------------------------------------- #
def _attach_cluster_ids(df, cluster_key_cols):
    """Attach the cluster_id (independent unit) column exactly like
    stat_tests.run_scrna_benchmark_stats.

    Returns (df_with_cluster_id, cluster_mode, cluster_key_list,
    cluster_key_source).  None -> 'material' derived from the dataset name
    (reference-cell configuration variants collapse onto their base material);
    [] -> naive per-unit mode; ['dataset'] -> per-dataset; a list -> custom."""
    if cluster_key_cols is None:
        cluster_mode = 'material'
        cluster_key_source = ('default: material derived from the dataset name '
                              '(patient / cell line; datasets sharing it are one unit)')
    elif list(cluster_key_cols) == []:
        cluster_mode = 'naive'
        cluster_key_source = 'clustering disabled (--cluster-key none): NAIVE per-unit level'
    elif list(cluster_key_cols) == ['dataset']:
        cluster_mode = 'dataset'
        cluster_key_source = 'user-specified: dataset (chips of one patient treated as independent)'
    else:
        cluster_mode = 'custom'
        cols = [c for c in cluster_key_cols if c in df.columns]
        dropped = [c for c in cluster_key_cols if c not in df.columns]
        if dropped:
            logging.warning('mcb: cluster-key columns %s not present in the input; '
                            'clustering uses the remaining %s',
                            dropped, cols or 'nothing')
        if not cols:
            cluster_mode = 'naive'
            cluster_key_source = 'no usable cluster columns: NAIVE per-unit level'
        else:
            cluster_key_cols = cols
            cluster_key_source = 'user-specified columns: ' + ', '.join(cols)
    df = df.copy()
    if cluster_mode == 'naive':
        df['cluster_id'] = ''
    elif cluster_mode == 'material':
        df['cluster_id'] = df['dataset'].map(_ST.material_from_dataset)
        _mat_map = _ST._collapse_materials(df['cluster_id'].unique())
        df['cluster_id'] = df['cluster_id'].map(_mat_map)
    elif cluster_mode == 'dataset':
        df['cluster_id'] = df['dataset'].map(
            lambda s: _ST._norm_missing(s) or '(missing)')
    else:
        df['cluster_id'] = df[cluster_key_cols].apply(
            lambda r: '|'.join(_ST._norm_missing(r[c]) or '(missing)'
                               for c in cluster_key_cols), axis=1)
    key_list = (['material'] if cluster_mode == 'material'
                else (['dataset'] if cluster_mode == 'dataset'
                      else (list(cluster_key_cols) if cluster_mode == 'custom' else [])))
    return df, cluster_mode, key_list, cluster_key_source


def _iter_metric_families(df, metric_list, cluster_key_cols=None,
                          cluster_agg='median', lower_is_better=()):
    """Yield the Fig. 5 MCB families: (metric, unit x method matrix), one per
    metric of the swarm grid.

    The matrix is built EXACTLY like the Friedman 'cluster' rows of
    stat_tests.run_scrna_benchmark_stats: per-cell values aggregated to
    (pair_id, method) medians, pivoted, then aggregated to the independent
    unit (material by default) with the same cluster_agg.  Metrics named in
    lower_is_better are negated so that LARGER always means BETTER; the
    orientation is reported with the family."""
    df, cluster_mode, key_list, _src = _attach_cluster_ids(df, cluster_key_cols)
    cell_col = 'ground_truth_cell' if 'ground_truth_cell' in df.columns else None
    if cell_col is None:
        df['pair_id'] = df['dataset'].map(_ST._norm_missing)
    else:
        df['pair_id'] = (df['dataset'].map(_ST._norm_missing) + '|'
                         + df[cell_col].map(_ST._norm_missing))
    for metric in metric_list:
        msub = df[df['metric'] == metric]
        if msub.empty:
            continue
        series = msub.groupby(['pair_id', 'cluster_id', 'method'])['value'].median()
        piv = series.unstack('method')
        if piv.empty:
            continue
        if cluster_mode != 'naive':
            mat = piv.groupby(level=1, sort=True).agg(cluster_agg)
        else:
            mat = piv.droplevel(1)
        orientation = 1.0
        if str(metric) in set(lower_is_better):
            mat = -mat
            orientation = -1.0
        yield {'metric': str(metric), 'matrix': mat,
               'cluster_key': '|'.join(key_list) if key_list else '',
               'cluster_mode': cluster_mode,
               'orientation': ('lower-is-better (negated so larger = better)'
                               if orientation < 0 else 'higher-is-better')}


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def _write_outputs(out_prefix, tsv_rows, settings, tex_lines=None):
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tsv_path = out_prefix + '.tsv'
    pd.DataFrame(tsv_rows).to_csv(tsv_path, sep='\t', index=False, na_rep='NA')
    with open(out_prefix + '.json', 'w') as fh:
        json.dump(settings, fh, indent=2)
    if tex_lines is not None:
        with open(out_prefix + '.tex', 'w') as fh:
            fh.write('\n'.join(tex_lines) + '\n')
        logging.info('mcb: wrote %s, .json, .tex', tsv_path)
    else:
        logging.info('mcb: wrote %s, .json', tsv_path)


def run_scrna_mcb(df, out_prefix, cluster_key_cols=None, metrics=None,
                  lower_is_better=(), cluster_agg='median',
                  n_resamples=DEFAULT_N_RESAMPLES, seed=1, alpha=0.05,
                  write_tex=True, table_label='tab:scrna-mcb'):
    """Fig. 5 (scRNA-seq benchmark) MCB: one family per metric.

    Writes <out_prefix>.tsv / .json and, by default, the booktabs table
    <out_prefix>.tex (one section per metric; rows = method configurations,
    each carrying exactly n, p, r and the simultaneous 95% MCB interval of
    the gap to the best).  Returns the settings dict, or None when nothing
    was analysable."""
    if not _HAVE_SCIPY:
        logging.warning('scipy is not available: MCB skipped')
        return None
    needed = {'dataset', 'method', 'metric', 'value'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(F'mcb: input lacks columns {sorted(missing)} '
                         '(need dataset, method, metric, value)')
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    metric_list = ([m for m in metrics if m in set(df['metric'].unique())]
                   if metrics is not None else
                   list(dict.fromkeys(df['metric'].tolist())))
    lower_is_better = set(lower_is_better or ())

    rows_out, verdicts, skipped = [], [], []
    n_families = 0
    for fam in _iter_metric_families(df, metric_list,
                                     cluster_key_cols=cluster_key_cols,
                                     cluster_agg=cluster_agg,
                                     lower_is_better=lower_is_better):
        n_families += 1
        res = mcb_analyse(fam['matrix'], alpha=alpha, n_resamples=n_resamples,
                          seed=seed)
        if res is None:
            skipped.append(fam['metric'])
            continue
        res['verdict']['family'] = fam['metric']
        verdicts.append(res['verdict'])
        for rec in res['rows']:
            rec = dict(rec)
            rec.update({
                'metric': fam['metric'],
                'task': 'Fig5 (scRNA-seq CNV-caller benchmark)',
                'cluster_key': fam['cluster_key'],
                'metric_orientation': fam['orientation'],
                'friedman_pvalue': res['verdict']['friedman_pvalue'],
                'kendalls_w': res['verdict']['kendalls_w'],
                'family_unique_winner': res['verdict']['unique_winner'],
                'family_leading_group': '|'.join(res['verdict']['leading_group']),
                'family_leading_group_size': res['verdict']['leading_group_size'],
                'family_notes': '; '.join(res['notes']),
            })
            rows_out.append(rec)
    if not verdicts:
        logging.warning('mcb: no analysable metric family%s',
                        F' (skipped: {", ".join(skipped)})' if skipped else '')
        return None

    # consensus across the metric families: the methods never significantly
    # inferior in any metric, and how often each method is the sample-best
    never_inferior = sorted(set.intersection(*[
        set(v['leading_group']) for v in verdicts])) if verdicts else []
    best_counts = {}
    for v in verdicts:
        best_counts[v['sample_best']] = best_counts.get(v['sample_best'], 0) + 1

    settings = {
        'analysis': "Hsu's MCB (comparison with the best) - scRNA-seq CNV-caller "
                    'benchmark (Fig. 5)',
        'design': 'per metric family: material x method complete-block matrix '
                  '(per-unit medians of the per-cell values, cluster key as in '
                  'stat_tests); MCB intervals for theta_i - max_{j != i} theta_j',
        'inference_level': 'cluster (independent experimental units; material by default)',
        'interval_method': 'studentized cluster bootstrap max-|t| (Tukey-style '
                           'projection; single-step, family = all method pairs '
                           'within the metric family)',
        'p_definition': 'MCB-adjusted one-sided P of H0: theta_i >= max_{j != i} '
                        'theta_j against H1: inferior to the best (single-step '
                        'bootstrap max-|t|, halved two-sided)',
        'effect_size': 'matched-pairs rank-biserial r versus the best competitor '
                       '(positive = the method scores higher), 95% percentile '
                       'bootstrap CI (descriptive, per method); the table CI is '
                       'the simultaneous MCB interval of the gap to the best',
        'parametric_cross_check': 'studentized range q(k, n-1)/sqrt(2) (k = 2 '
                                  'reproduces the paired-t critical value)',
        'omnibus': 'Friedman test on the same complete-block matrix',
        'metric_orientation': 'every metric oriented so larger = better '
                              '(--lower-is-better metrics negated first)',
        'tail': 'one-sided p (inferiority to the best); intervals two-sided',
        'verdict_rules': 'interval entirely below 0 = significantly inferior; '
                         'entirely above 0 = unique winner; contains 0 = member '
                         'of the leading group (g = group size; g = 2 is the '
                         'top-2 situation)',
        'n_families': n_families,
        'n_families_skipped': len(skipped),
        'skipped_families': skipped,
        'metrics': [v['family'] for v in verdicts],
        'lower_is_better': sorted(lower_is_better),
        'bootstrap_n_resamples': int(min(max(int(n_resamples), 200), MAX_BOOTSTRAP)),
        'bootstrap_seed': int(seed),
        'alpha': float(alpha),
        'per_family_verdicts': verdicts,
        'consensus': {
            'n_metric_families': len(verdicts),
            'methods_never_significantly_inferior': never_inferior,
            'sample_best_counts': best_counts,
            'families_with_unique_winner': sum(1 for v in verdicts if v['unique_winner']),
            'families_with_top2': sum(1 for v in verdicts if v['top2_exists']),
            'families_with_ordering': sum(1 for v in verdicts if v['ordering_exists']),
        },
        'software': _ST.software_versions(),
    }
    tex_lines = None
    if write_tex:
        tex_lines = scrna_mcb_latex_lines(pd.DataFrame(rows_out), alpha=alpha,
                                          table_label=table_label)
    _write_outputs(out_prefix, rows_out, settings, tex_lines)
    return settings


# --------------------------------------------------------------------------- #
# LaTeX table (exactly n, p, r, 95% CI after the identity columns)            #
# --------------------------------------------------------------------------- #
_MCB_CAPTION_CORE = (
    'Hsu\'s multiple comparison with the best (MCB): every row is one method '
    'configuration compared with the best of the others.  After the identity '
    'columns each row carries exactly $n$ (independent units = materials), '
    '$p$ (MCB-adjusted one-sided P; H$_0$: the method is at least as good as '
    'the best, small $P$ = significantly inferior), $r$ (matched-pairs '
    'rank-biserial effect size versus the best competitor; positive = better) '
    'and the simultaneous 95\\% MCB interval of the performance gap to the '
    'best ($\\theta_i - \\max_{j \\ne i} \\theta_j$; entirely below 0 = '
    'significantly inferior, entirely above 0 = unique best, brackets 0 = '
    'indistinguishable from the best, i.e.\\ member of the leading group).')


def _disp_method(method):
    """Method display label via stat_tests (e.g. 'copykat_autoInferRef' ->
    'CopyKAT 2021 (autoInferRef)')."""
    return _ST._method_display_tex(method)


def scrna_mcb_latex_lines(rows_df, alpha=0.05, table_label='tab:scrna-mcb'):
    """Booktabs table for the Fig. 5 scRNA MCB: one section per metric, rows =
    method configurations, columns = exactly n, p, r, 95% CI.  Returns the
    line list or None."""
    if rows_df is None or rows_df.empty:
        logging.error('mcb latex: no rows')
        return None
    metric_order = list(dict.fromkeys(rows_df['metric'].tolist()))
    by_metric = {}
    for rec in rows_df.to_dict('records'):
        by_metric.setdefault(rec['metric'], []).append(rec)
    caption = (F'Hsu\'s MCB (comparison with the best) for the scRNA-seq '
               F'CNV-caller benchmark (Fig.~5), one section per swarm-grid '
               F'metric: each row compares one method configuration with the '
               F'best of the others at the material level (per-material '
               F'medians; complete blocks).  ' + _MCB_CAPTION_CORE)
    lines = [
        F'% LaTeX table generated by mcb.py (requires \\usepackage{{booktabs}})',
        '\\begin{table}[htbp]',
        '  \\centering',
        '  \\small',
        F'  \\caption{{{caption}}}',
        F'  \\label{{{table_label}}}',
        '  \\begin{tabular}{lrrrr}',
        '    \\toprule',
        '    Method & $n$ & $p$ & $r$ & 95\\% CI \\\\',
        '    \\midrule',
    ]
    for metric in metric_order:
        rows = sorted(by_metric[metric],
                      key=lambda r: (r.get('rank_position', 99), str(r['method'])))
        lines.append(F'    \\multicolumn{{5}}{{l}}{{\\textit{{{_tex_escape(metric)}}}}} \\\\')
        for rec in rows:
            n_v = _fmt_signed_int(rec.get('n_units'))
            p_v = _fmt_pvalue_mcb(rec.get('pvalue_mcb_one_sided'), alpha=alpha)
            r_v = _fmt_effect(rec.get('rank_biserial_r_vs_best'))
            ci_v = _fmt_ci(rec.get('mcb_low'), rec.get('mcb_high'))
            lines.append(F'    {_tex_escape(_disp_method(rec["method"]))} '
                         F'& {n_v} & {p_v} & {r_v} & {ci_v} \\\\')
        lines.append('    \\midrule')
    if lines[-1] == '    \\midrule':
        lines[-1] = '    \\bottomrule'
    lines += ['  \\end{tabular}', '\\end{table}']
    return lines


# --------------------------------------------------------------------------- #
# Command-line interface                                                       #
# --------------------------------------------------------------------------- #
def _parse_cluster_key(raw):
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ('none', 'naive', 'off', 'no'):
        return []
    if s in ('', 'material', 'default'):
        return None
    if s in ('dataset', 'per-dataset'):
        return ['dataset']
    return [c.strip() for c in raw.split(',') if c.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Hsu's MCB (comparison with the best) for the scRNA-seq "
                    'CNV-caller benchmark (Fig. 5): one family per metric, '
                    'units = materials.')
    ap.add_argument('-i', '--input', nargs='*', default=[],
                    help="Input file(s)/glob(s): raw evaluation TSVs (dataset "
                         "inferred from each file's path, exactly like "
                         'plot_cnv_heatmaps.py) or unified long TSVs '
                         '(dataset/method/metric/value).')
    ap.add_argument('-o', '--output', default='mcb',
                    help='Output prefix (writes <prefix>.tsv/.json/.tex).')
    ap.add_argument('--metrics', default=None, metavar='IDS',
                    help='Comma-separated metric ids to analyse (default: every '
                         'metric present).')
    ap.add_argument('--lower-is-better', default='', metavar='IDS',
                    help='Comma-separated metric ids whose SMALLER values are '
                         'better (they are negated before the analysis).')
    ap.add_argument('--cluster-key', default=None, metavar='COLS',
                    help='Independent-unit key: "material" (default), "dataset", '
                         '"none" (naive per-unit) or comma-separated columns.')
    ap.add_argument('--boot', type=int, default=DEFAULT_N_RESAMPLES, metavar='N',
                    help=F'Bootstrap resamples (default: {DEFAULT_N_RESAMPLES}, '
                         F'capped at {MAX_BOOTSTRAP}).')
    ap.add_argument('--seed', type=int, default=1, metavar='SEED',
                    help='Seed of the bootstrap RNG (default: 1).')
    ap.add_argument('--alpha', type=float, default=0.05, metavar='ALPHA',
                    help='Family-wise alpha (default: 0.05).')
    ap.add_argument('--no-tex', dest='tex', action='store_false', default=True,
                    help='Do not write the booktabs LaTeX table.')
    ap.add_argument('--latex-table', action='store_true', default=False,
                    help='Rebuild and print the LaTeX table from an existing '
                         '<prefix>.tsv and exit.')
    args = ap.parse_args(argv)

    if args.latex_table:
        tsv = args.output + '.tsv'
        if not os.path.isfile(tsv):
            logging.error('mcb table file not found: %s', tsv)
            return 1
        rows = pd.read_csv(tsv, sep='\t')
        lines = scrna_mcb_latex_lines(rows, alpha=args.alpha)
        if lines is None:
            return 1
        print('\n'.join(lines))
        return 0

    if not args.input:
        logging.error('mcb: no input given (use -i <files/globs>)')
        return 1
    df = _ST._load_long_frames(args.input)
    if df is None or df.empty:
        logging.error('mcb: no usable data after loading %s', args.input)
        return 1
    metrics = ([m.strip() for m in args.metrics.split(',') if m.strip()]
               if args.metrics else None)
    lower_is_better = tuple(x.strip() for x in args.lower_is_better.split(',')
                            if x.strip())
    settings = run_scrna_mcb(
        df, args.output,
        cluster_key_cols=_parse_cluster_key(args.cluster_key),
        metrics=metrics, lower_is_better=lower_is_better,
        n_resamples=args.boot, seed=args.seed, alpha=args.alpha,
        write_tex=args.tex)
    return 0 if settings is not None else 1


if __name__ == '__main__':
    sys.exit(main())
