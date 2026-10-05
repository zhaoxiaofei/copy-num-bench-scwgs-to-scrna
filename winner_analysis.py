#!/usr/bin/env python3
"""winner_analysis.py - Ordering, winner and top-k group tests for the
scRNA-seq CNV-caller benchmark.

The companion module stat_tests.py answers, per metric, "does the reference
method outperform method b?" with exactly four statistics (n, p, r, 95% CI).
This module answers the three questions that reference-based pairwise
comparisons cannot:

1.  Does an APPROXIMATE ORDERING of the methods exist?  (the observed
    situation: the materials rank the methods concordantly, but adjacent
    methods in the ordering are not separable at the family-wise level)
2.  Does a UNIQUE WINNER exist?  (one method significantly outperforming the
    runner-up and, in the strict sense, every other method)
3.  Does a TOP-2 GROUP exist?  (two statistically indistinguishable methods
    that sit significantly ahead of the remaining methods)

Procedure (per metric; ALL inference at the level of the independent unit -
the MATERIAL, patient or cell line - exactly like stat_tests.py):

*   Orientation: for every metric the values are oriented so that LARGER
    always means BETTER (metrics named via --lower-is-better are negated
    first; every scRNA-seq benchmark metric is higher-is-better by default).
*   Ordering: methods are ordered by their Friedman MEAN RANK over the
    per-material method medians (complete blocks).
*   Omnibus "does an ordering exist": the Friedman test
    (H0: the methods are exchangeable across materials).  Kendall's W with a
    95% bootstrap CI measures the concordance strength, and the bootstrap
    rank-stability (mean Spearman rho between the observed ordering and the
    orderings of resampled materials, plus the reproduction rate of the
    leading group) quantifies how 'approximate' the ordering is.
    approximate_ordering_exists := Friedman P <= alpha AND at least one
    adjacent pair of the ordering is NOT separable (a fully separable chain
    would be a strict, not an approximate, ordering).
*   Separation: ALL k(k-1)/2 pairwise comparisons run as two-sided Wilcoxon
    signed-rank tests on the per-material differences of the two methods'
    medians (zero_method='zsplit', as in stat_tests.py), Holm-Bonferroni
    corrected within the metric family.  Method a of a comparison is always the
    BETTER-ranked method, so "a outperforms b" is confirmed iff
    p_holm <= alpha AND r > 0 (r = matched-pairs rank-biserial; two-sided P
    plus the sign of r, the same convention as the paper).
*   Step-down leading group: walk the ordered methods from the top and cut at
    the FIRST adjacent pair (M_g, M_{g+1}) that IS separable; the leading
    group is {M_1, ..., M_g}.  Hence

        unique winner exists  <=>  g = 1   (M1 vs M2 separable)
        top-2 group exists    <=>  g = 2   (M1 vs M2 NOT separable AND
                                            M2 vs M3 separable)
        no separation         <=>  no adjacent pair separable (leading group
                                    = all methods; the ordering, if any, is
                                    entirely approximate)

    As cross-checks: the homogeneity of the leading group (no member
    significantly beaten by another member) and its separation from below
    (every member vs every method below the boundary) are verified on ALL
    pairs, not only the adjacent ones; and a Nemenyi critical-difference
    grouping (q_alpha * sqrt(k(k+1)/(6N)) on the Friedman mean ranks) is
    reported next to the Wilcoxon/Holm result.

Outputs (prefix = -o/--output)
==============================
    <prefix>.order.tsv      per metric and method: rank position, Friedman
                            mean rank, raw median/mean, dominance counts,
                            leading-group / Nemenyi memberships
    <prefix>.stepdown.tsv   one row per adjacent step-down comparison (all
                            k-1; flags mark the inspected chain and the
                            boundary) with n, p, p_holm, r, 95% CI and the
                            per-metric verdict
    <prefix>.stepdown.tex   booktabs LaTeX table of the INSPECTED chain (the
                            comparisons from the top down to the boundary;
                            all of them when no boundary exists): every row
                            carries, after the identity columns, EXACTLY
                            n (effective sample size), p (Holm-adjusted
                            two-sided Wilcoxon P), r (matched-pairs
                            rank-biserial effect size) and the 95% bootstrap
                            CI of r - nothing else
    <prefix>.json           settings, per-metric verdicts, the consensus
                            across metrics, software versions, seed

Usage
=====
Standalone, on the raw evaluation TSVs (dataset inferred from each file's
path exactly like plot_cnv_heatmaps.py does):
    python winner_analysis.py -i 'results/*/evaluation/*.tsv' \
        -o heatmaps/winner
On one unified long TSV:
    python winner_analysis.py -i aggregated_per_cell_long.tsv -o heatmaps/winner
Re-print the LaTeX table from an existing run:
    python winner_analysis.py -o heatmaps/winner --latex-table
plot_cnv_heatmaps.py runs this module by default next to stat_tests.py
(--no_winner_analysis skips it); it also runs when the pairwise tests were
skipped because of a bogus --stats_reference, because no reference method is
needed here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats as sps
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - the frozen env always has scipy
    _HAVE_SCIPY = False

# Everything numerical is reused from the validated sibling module so that
# the two analyses can never disagree on a computed statistic.
import stat_tests as _ST

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s')

DEFAULT_STABILITY_RESAMPLES = 2000


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def nemenyi_cd(k, n, alpha=0.05):
    """Nemenyi critical difference on Friedman mean ranks.

    CD = q_{alpha}(k, inf) * sqrt(k(k+1)/(6N)); two methods whose mean ranks
    differ by less than CD are not separable by the Nemenyi post-hoc test
    (the standard all-pairs procedure after a Friedman omnibus).  Returns
    None when the studentized-range quantile is unavailable (old scipy).
    """
    if not _HAVE_SCIPY or k < 2 or n < 1:
        return None
    try:
        q = sps.studentized_range.ppf(1.0 - float(alpha), k, np.inf)
        if not np.isfinite(q):
            return None
        return float(q * np.sqrt(k * (k + 1.0) / (6.0 * n)))
    except Exception:
        return None


def _spearman(a, b):
    """Spearman correlation robust to scipy's result-object renaming."""
    if len(a) < 3:
        return float('nan')
    res = sps.spearmanr(a, b)
    val = getattr(res, 'statistic', None)
    if val is None:
        val = getattr(res, 'correlation', float('nan'))
    return float(val)


def _pair_stats(x, y, labels, cluster_agg='median', n_resamples=2000, seed=1):  # noqa: C901
    """One ordered pairwise comparison - the four reported statistics only.

    A light twin of stat_tests._pairwise_record: the two-sided Wilcoxon
    signed-rank test, the matched-pairs rank-biserial r and its 95%
    percentile-bootstrap CI all run on the paired differences of the two
    methods' per-cluster (material) aggregates - exactly the columns of the
    Friedman/MCB complete-block matrix - as in stat_tests.py (labels=None ->
    naive per-unit mode).  x is the BETTER-ranked method's values, so
    r > 0 means "x outperforms y".
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if labels is not None and len(labels) != len(x):
        raise ValueError('_pair_stats: labels length does not match x')
    ok = np.isfinite(x) & np.isfinite(y)
    if labels is not None:
        labels = [l for l, m in zip(labels, ok) if m]
    x, y = x[ok], y[ok]
    d = x - y
    if labels is not None:
        idx = pd.Index(labels, dtype=object)
        x_cl = pd.Series(x, index=idx).groupby(level=0).agg(cluster_agg).sort_index()
        y_cl = pd.Series(y, index=idx).groupby(level=0).agg(cluster_agg).sort_index()
        # Difference of the two methods' per-cluster aggregates (the
        # Friedman/MCB matrix columns), not the per-cluster median of the
        # per-cell differences: median(x-y) != median(x)-median(y).
        dv = (x_cl - y_cl).to_numpy(dtype=float)
    else:
        dv = d
    w = _ST.wilcoxon_signed_rank(dv, np.zeros_like(dv)) if len(dv) else None
    st = _ST.sign_test_two_sided(dv) if len(dv) else None
    r = _ST.rank_biserial_matched(dv)
    n_boot = int(min(n_resamples, _ST.MAX_CLUSTER_BOOTSTRAP))
    if len(dv) == 0:
        r_lo, r_hi, r_ci_method = float('nan'), float('nan'), ''
    else:
        r_lo, r_hi, r_ci_method = _ST.bootstrap_r_ci(
            dv, n_resamples=max(n_boot, 200), seed=seed)
    note = ''
    if len(dv) and 0 < len(dv) < 6:
        note = (F'only {len(dv)} independent units: the exact two-sided '
                'Wilcoxon cannot reach P < 0.05 below 6 units; see '
                'pvalue_sign_test')
    elif w is not None and w['note']:
        note = w['note']
    return {
        'n': int(len(dv)),                    # units the primary test runs on
        'n_units': int(len(d)),
        'pvalue_two_sided': w['pvalue'] if w else float('nan'),
        'pvalue_sign_test': st['pvalue'] if st else float('nan'),
        'rank_biserial_r': r,
        'ci95_r_low': r_lo, 'ci95_r_high': r_hi, 'ci_r_method': r_ci_method,
        'wilcoxon_W': w['statistic'] if w else float('nan'),
        'median_diff_a_minus_b': float(np.median(dv)) if len(dv) else float('nan'),
        'note': note,
    }


def bootstrap_order_stability(cmat, n_resamples=2000, seed=1,
                              method_names=None, leading_group=None):
    """Bootstrap the ordering by resampling the independent units.

    Resamples the complete-block matrix rows (materials) with replacement,
    recomputes the Friedman mean ranks and reports: the mean and 95%
    percentile interval of Spearman's rho between each bootstrap ordering and
    the observed ordering; the bootstrap distribution of Kendall's W; and,
    when `leading_group` is a proper non-empty subset of `method_names`, the
    rate at which the same leading-group SET is reproduced.  A high rho with
    a low reproduction rate means the ORDER is stable but its exact cut
    points are not - the statistical picture of an 'approximate ordering'.
    """
    mat = np.asarray(cmat, dtype=float)
    n, k = mat.shape
    if n < 2 or k < 3:
        return None
    names = list(method_names) if method_names is not None else None
    g = len(leading_group) if leading_group is not None else 0
    want = set(leading_group) if leading_group is not None else None
    rng = np.random.default_rng(seed)
    b = int(n_resamples)
    obs_mr = sps.rankdata(mat, axis=1).mean(axis=0)
    obs_c = sps.rankdata(obs_mr)
    obs_c = obs_c - obs_c.mean()
    obs_ss = float(np.sum(obs_c ** 2))
    want_idx = (np.sort([names.index(m) for m in leading_group])
                if 0 < g < k and names is not None and want is not None else None)
    rhos, ws = np.empty(b, dtype=float), np.empty(b, dtype=float)
    top_ok = np.zeros(b, dtype=bool) if want_idx is not None else None
    # Vectorised resampling in chunks of resamples (bounds the b x n x k memory).
    chunk = max(1, int(2_000_000 // max(n * k, 1)))
    for start in range(0, b, chunk):
        stop = min(start + chunk, b)
        sub = mat[rng.integers(0, n, size=(stop - start, n))]
        mr = sps.rankdata(sub, axis=2).mean(axis=1)          # (resamples, k)
        # Spearman rho = Pearson correlation of the mean-rank vectors
        # (scipy.stats.spearmanr on ranks with average ties).
        mrc = sps.rankdata(mr, axis=1)
        mrc = mrc - mrc.mean(axis=1, keepdims=True)
        den = np.sqrt(np.sum(mrc ** 2, axis=1) * obs_ss)
        with np.errstate(invalid='ignore', divide='ignore'):
            rhos[start:stop] = np.where(den > 0, (mrc @ obs_c) / den, np.nan)
        # Kendall's W from the (tie-uncorrected) Friedman chi-square, the
        # same statistic scipy.stats.friedmanchisquare computes
        s = np.sum((mr - (k + 1.0) / 2.0) ** 2, axis=1)
        ws[start:stop] = (12.0 * n * s / (k * (k + 1.0))) / (n * (k - 1.0))
        if want_idx is not None:
            top = np.sort(np.argsort(-mr, axis=1, kind='stable')[:, :g], axis=1)
            top_ok[start:stop] = np.all(top == want_idx, axis=1)
    rhos = rhos[np.isfinite(rhos)]
    out = {
        'n_resamples': int(n_resamples),
        'rho_mean': float(np.mean(rhos)) if len(rhos) else float('nan'),
        'rho_ci95': ([float(np.percentile(rhos, 2.5)),
                      float(np.percentile(rhos, 97.5))]
                     if len(rhos) else None),
        'kendalls_w_mean': float(np.mean(ws)) if len(ws) else float('nan'),
        'kendalls_w_ci95': ([float(np.percentile(ws, 2.5)),
                             float(np.percentile(ws, 97.5))]
                            if len(ws) else None),
    }
    if top_ok is not None:
        out['leading_group_reproduction'] = float(np.mean(top_ok))
    return out


# --------------------------------------------------------------------------- #
# Per-metric analysis                                                         #
# --------------------------------------------------------------------------- #
def _analyse_metric(msub, metric, raw_summaries, cluster_mode, cluster_agg,
                    n_resamples, stability_resamples, seed, alpha, direction):
    """Full winner/ordering analysis of one metric. Returns
    (verdict dict, order rows, stepdown rows) or (None, note, None)."""
    raw_med, raw_mean = raw_summaries
    if direction < 0:
        msub = msub.assign(value=-msub['value'].astype(float))

    series = msub.groupby(['pair_id', 'cluster_id', 'method'])['value'].median()
    piv = series.unstack('method')
    ok_methods = [m for m in piv.columns if piv[m].notna().any()]
    if len(ok_methods) < 2:
        return None, F'only {len(ok_methods)} method(s) with data', None

    # ---- complete-block cluster matrix (the Friedman input) ----
    if cluster_mode == 'naive':
        cmat = piv.loc[:, ok_methods]
    else:
        cmat = (piv.loc[:, ok_methods]
                .groupby(level=1, sort=True).agg(cluster_agg))
    cmat_complete = cmat.dropna(axis=0, how='any')
    n_blocks = int(len(cmat_complete))
    k = len(ok_methods)
    if n_blocks < 2:
        return None, F'only {n_blocks} complete block(s)', None

    mat = cmat_complete.to_numpy(dtype=float)
    mean_ranks = sps.rankdata(mat, axis=1).mean(axis=0)
    mr = {m: float(rk) for m, rk in zip(ok_methods, mean_ranks)}

    # ---- omnibus Friedman + concordance ----
    fr = _ST.friedman_test([cmat_complete[c].to_numpy() for c in ok_methods]) \
        if k >= 3 else None
    friedman_p = fr['pvalue'] if fr else float('nan')
    kendalls_w = fr['kendalls_w'] if fr else float('nan')

    # ---- ordering (higher mean rank = better; ties -> cluster median) ----
    cl_med = {m: float(cmat_complete[m].median()) for m in ok_methods}
    order = sorted(ok_methods,
                   key=lambda m: (-mr[m], -cl_med[m], str(m)))
    rank_of = {m: i + 1 for i, m in enumerate(order)}

    # ---- ALL pairwise comparisons (a = better ranked) ----
    pairs = [(order[i], order[j])
             for i in range(k) for j in range(i + 1, k)]
    recs = {}
    for a, b in pairs:
        both = piv[[a, b]].dropna(axis=0, how='any')
        if not len(both):
            continue
        labels = None
        if cluster_mode != 'naive':
            labels = both.index.get_level_values('cluster_id').tolist()
        recs[(a, b)] = _pair_stats(
            both[a].to_numpy(dtype=float), both[b].to_numpy(dtype=float),
            labels, cluster_agg=cluster_agg, n_resamples=n_resamples,
            seed=seed)
    if len(recs) < 1:
        return None, 'no overlapping method pair with data', None

    fam = [(a, b) for (a, b) in pairs if (a, b) in recs]
    adj = _ST.holm_bonferroni([recs[(a, b)]['pvalue_two_sided'] for (a, b) in fam])
    for (a, b), p_adj in zip(fam, adj):
        recs[(a, b)]['pvalue_holm'] = float(p_adj)
        recs[(a, b)]['a_beats_b'] = bool(
            np.isfinite(p_adj) and p_adj <= alpha
            and np.isfinite(recs[(a, b)]['rank_biserial_r'])
            and recs[(a, b)]['rank_biserial_r'] > 0)

    # ---- step-down leading group ----
    adjacent = [(order[i], order[i + 1]) for i in range(k - 1)]
    adjacent_sig = [recs.get(p, {}).get('a_beats_b', False) for p in adjacent]
    boundary = None
    for i, is_sig in enumerate(adjacent_sig):
        if is_sig:
            boundary = i + 1          # leading group = order[:i+1]
            break
    g = boundary if boundary is not None else k
    leading = order[:g]

    unique_winner = bool(boundary == 1)
    top2_exists = bool(boundary == 2)
    no_separation = boundary is None
    unique_winner_method = order[0] if unique_winner else None
    top2 = order[:2] if top2_exists else None

    # cross-checks on ALL pairs (not just adjacent ones)
    within_bad = [(order[i], order[j]) for i in range(g) for j in range(i + 1, g)
                  if recs.get((order[i], order[j]), {}).get('a_beats_b', False)]
    homogeneous = not within_bad
    cross = [(order[i], order[j]) for i in range(g) for j in range(g, k)]
    cross_sep = [recs[p]['a_beats_b'] for p in cross if p in recs]
    cross_min_r = (float(min(recs[p]['rank_biserial_r'] for p in cross
                             if p in recs)) if cross else float('nan'))

    # ---- Nemenyi cross-check ----
    cd = nemenyi_cd(k, n_blocks, alpha)
    best_mr = mr[order[0]]
    nemenyi_top = ([m for m in order if best_mr - mr[m] < cd]
                   if cd is not None else None)

    # ---- bootstrap stability of the ordering (rho, W CI, top-g set) ----
    stab = bootstrap_order_stability(
        mat, n_resamples=stability_resamples, seed=seed,
        method_names=ok_methods, leading_group=leading)

    ordering_supported = bool(np.isfinite(friedman_p) and friedman_p <= alpha)
    all_adjacent_sig = bool(k >= 2 and all(adjacent_sig)) if k >= 2 else False
    approximate = bool(ordering_supported and not all_adjacent_sig)

    if unique_winner:
        verdict = (F'unique winner: {order[0]} (g=1; M1 vs M2 separable)')
    elif top2_exists:
        verdict = (F'top-2 group: {order[0]} + {order[1]} '
                   '(g=2; M1 vs M2 not separable, M2 vs M3 separable)')
    elif no_separation:
        verdict = (F'no separation (g={k}; no adjacent pair separable)')
    else:
        verdict = (F'leading group of {g}: ' + ', '.join(leading))
    verdict += F'; ordering {"" if ordering_supported else "NOT "}supported' \
               F' (Friedman P={friedman_p:.3g})'
    if not homogeneous:
        verdict += '; WARNING: leading group not internally homogeneous'

    # ---- order.tsv rows ----
    dominance = {m: 0 for m in ok_methods}
    dominated_by = {m: 0 for m in ok_methods}
    for (a, b), rec in recs.items():
        if rec.get('a_beats_b'):
            dominance[a] += 1
            dominated_by[b] += 1
    order_rows = []
    for m in order:
        order_rows.append({
            'metric': metric,
            'direction': 'higher' if direction > 0 else 'lower',
            'method': m,
            'rank': rank_of[m],
            'mean_rank': mr[m],
            'median_value': raw_med.get(m, float('nan')),
            'mean_value': raw_mean.get(m, float('nan')),
            'n_blocks': n_blocks,
            'k_methods': k,
            'dominance_count': dominance[m],
            'dominated_by_count': dominated_by[m],
            'in_leading_group': m in leading,
            'in_nemenyi_top_group': (m in nemenyi_top
                                     if nemenyi_top is not None else ''),
            'unique_winner': unique_winner and m == order[0],
        })

    # ---- stepdown.tsv rows (all adjacent comparisons) ----
    stepdown_rows = []
    level_str = 'cluster' if cluster_mode != 'naive' else 'unit (naive)'
    for i, (a, b) in enumerate(adjacent):
        rec = recs.get((a, b))
        if rec is None:
            continue
        step = i + 1
        stepdown_rows.append({
            'metric': metric,
            'level': level_str,
            'step': step,
            'method_a': a,
            'method_b': b,
            'n': rec['n'],
            'n_units': rec['n_units'],
            'pvalue_two_sided': rec['pvalue_two_sided'],
            'pvalue_holm': rec['pvalue_holm'],
            'rank_biserial_r': rec['rank_biserial_r'],
            'ci95_r_low': rec['ci95_r_low'],
            'ci95_r_high': rec['ci95_r_high'],
            'ci_r_method': rec['ci_r_method'],
            'pvalue_sign_test': rec['pvalue_sign_test'],
            'wilcoxon_W': rec['wilcoxon_W'],
            'median_diff_a_minus_b': rec['median_diff_a_minus_b'],
            'a_beats_b': rec['a_beats_b'],
            'is_boundary': bool(boundary == step),
            'is_shown': bool(boundary is None or step <= boundary),
            'leading_group_size': g,
            'metric_verdict': verdict,
            'note': rec['note'],
        })

    verdict_dict = {
        'metric': metric,
        'direction': 'higher' if direction > 0 else 'lower',
        'n_blocks': n_blocks,
        'k_methods': k,
        'separable_pairs': [[a, b] for (a, b) in fam if recs[(a, b)]['a_beats_b']],
        'friedman_chi2': fr['chi2'] if fr else float('nan'),
        'friedman_df': fr['df'] if fr else None,
        'friedman_pvalue': friedman_p,
        'kendalls_w': kendalls_w,
        'ordering_supported': ordering_supported,
        'all_adjacent_separable': all_adjacent_sig,
        'approximate_ordering_exists': approximate,
        'ordering': order,
        'mean_ranks': mr,
        'rank_stability': stab,
        'leading_group_size': g,
        'leading_group': leading,
        'unique_winner': unique_winner,
        'unique_winner_method': unique_winner_method,
        'top2_exists': top2_exists,
        'top2': top2,
        'no_separation': no_separation,
        'leading_group_homogeneous': homogeneous,
        'cross_boundary_pairs_total': len(cross),
        'cross_boundary_pairs_separable': int(sum(1 for v in cross_sep if v)),
        'cross_boundary_min_r': cross_min_r,
        'nemenyi_cd': cd,
        'nemenyi_top_group': nemenyi_top,
        'verdict': verdict,
    }
    return verdict_dict, None, (order_rows, stepdown_rows)



# --------------------------------------------------------------------------- #
# The scRNA-seq winner / ordering analysis                                    #
# --------------------------------------------------------------------------- #
def run_scrna_winner_analysis(df, out_prefix, cluster_key_cols=None,
                              cluster_agg='median', n_resamples=10000,
                              stability_resamples=DEFAULT_STABILITY_RESAMPLES,
                              seed=1, alpha=0.05, metrics=None,
                              metric_directions=None):
    """Ordering / unique-winner / top-2 analysis of the scRNA-seq benchmark.

    df: long dataframe, one row per (dataset x cell x method x metric) - or
    per (dataset x method x metric) for dataset-level metrics - with at least
    the columns (dataset, method, metric, value).  NO reference method is
    needed: all methods are compared symmetrically.
    cluster_key_cols: None -> the derived 'material' (patient / cell line) is
    the independent unit (default); ['dataset'] -> per-dataset level;
    [] -> naive per-unit mode (discouraged).
    metric_directions: {metric: 'lower'} marks metrics where SMALLER values
    are better (every default benchmark metric is higher-is-better).
    Writes <out_prefix>.order.tsv, <out_prefix>.stepdown.tsv,
    <out_prefix>.stepdown.tex and <out_prefix>.json; returns the settings
    dict that is also written to .json.
    """
    if not _HAVE_SCIPY:
        logging.warning('winner_analysis: scipy is not available, analysis skipped')
        return None
    needed = {'dataset', 'method', 'metric', 'value'}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise SystemExit(F'winner_analysis: input lacks columns '
                         F'{sorted(missing_cols)} (need dataset, method, '
                         'metric, value)')
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # ---- resolve the cluster key (mirrors stat_tests.run_scrna_benchmark_stats)
    if cluster_key_cols is None:
        cluster_mode = 'material'
        cluster_key_source = ('default: material from dataset_materials.'
                              'MATERIAL_DATASETS (primary sample -> derived '
                              'dataset names; all datasets of one primary '
                              'sample are one unit)')
    elif list(cluster_key_cols) == []:
        cluster_mode = 'naive'
        cluster_key_source = 'clustering disabled: NAIVE per-unit level'
    elif list(cluster_key_cols) == ['dataset']:
        cluster_mode = 'dataset'
        cluster_key_source = 'user-specified: dataset level (sensitivity analysis)'
    else:
        cluster_mode = 'custom'
        dropped = [c for c in cluster_key_cols if c not in df.columns]
        cluster_key_cols = [c for c in cluster_key_cols if c in df.columns]
        cluster_key_source = 'user-specified columns: ' + (
            ', '.join(cluster_key_cols) or '(none)')
        if not cluster_key_cols:
            cluster_mode = 'naive'

    if cluster_mode == 'naive':
        df['cluster_id'] = ''
    elif cluster_mode == 'material':
        # exact lookup in dataset_materials.MATERIAL_DATASETS (primary sample
        # -> derived dataset names): all datasets of one primary sample are
        # ONE independent unit
        df['cluster_id'] = df['dataset'].map(_ST.material_from_dataset)
    elif cluster_mode == 'dataset':
        df['cluster_id'] = df['dataset'].map(
            lambda s: _ST._norm_missing(s) or '(missing)')
    else:
        df['cluster_id'] = df[cluster_key_cols].apply(
            lambda r: '|'.join(_ST._norm_missing(r[c]) or '(missing)'
                               for c in cluster_key_cols), axis=1)

    cell_col = 'ground_truth_cell' if 'ground_truth_cell' in df.columns else None
    if cell_col is None:
        df['pair_id'] = df['dataset'].map(_ST._norm_missing)
    else:
        df['pair_id'] = (df['dataset'].map(_ST._norm_missing) + '|'
                         + df[cell_col].map(_ST._norm_missing))

    metric_directions = metric_directions or {}
    metric_list = ([m for m in metrics if m in set(df['metric'].unique())]
                   if metrics is not None else
                   list(dict.fromkeys(df['metric'].tolist())))

    settings = {
        'analysis': 'scRNA-seq CNV-caller ordering / winner / top-k analysis',
        'questions': [
            'does an approximate ordering of the methods exist?',
            'does a unique winner exist?',
            'does a top-2 group exist?'],
        'design': ('randomized complete block; blocks = paired cells within '
                   'datasets; inference aggregated to the independent unit '
                   F'({cluster_mode}) before testing'),
        'cluster_key_source': cluster_key_source,
        'omnibus_test': 'Friedman per metric on per-material method medians '
                        '(complete blocks); Kendall W + bootstrap 95% CI',
        'posthoc_test': 'two-sided Wilcoxon signed-rank on the per-material '
                        'differences of the method medians for ALL k(k-1)/2 method pairs, '
                        'Holm-Bonferroni within each metric family',
        'decision_rule': ('method a (better Friedman mean rank) outperforms b '
                          'iff Holm-adjusted two-sided P <= alpha AND '
                          'rank-biserial r > 0 (the paper\'s two-sided '
                          'convention; the sign of r carries the direction)'),
        'step_down': 'leading group = methods above the first separable '
                     'adjacent pair; unique winner <=> group size 1; '
                     'top-2 <=> group size 2',
        'cross_checks': ['Nemenyi critical difference on Friedman mean ranks',
                         'leading-group homogeneity and separation-from-below '
                         'verified on all pairs',
                         'bootstrap rank stability (Spearman rho, leading-group '
                         'reproduction)'],
        'effect_size': 'matched-pairs rank-biserial r with a 95% '
                       'percentile-bootstrap CI over the independent units',
        'metric_directions': {m: ('lower' if str(metric_directions.get(m, ''))
                                  .strip().lower().startswith('l') else 'higher')
                              for m in metric_list},
        'metrics': list(metric_list),
        'bootstrap_n_resamples': int(n_resamples),
        'stability_n_resamples': int(stability_resamples),
        'bootstrap_seed': int(seed),
        'alpha': float(alpha),
        'software': _ST.software_versions(),
    }

    verdicts, order_rows, stepdown_rows, skipped = [], [], [], []
    for metric in metric_list:
        msub = df[df['metric'] == metric]
        if not len(msub):
            continue
        raw_med = msub.groupby('method')['value'].median().to_dict()
        raw_mean = msub.groupby('method')['value'].mean().to_dict()
        direction = -1 if str(metric_directions.get(metric, '')).strip().lower() \
            .startswith('l') else 1
        verdict, note, rows = _analyse_metric(
            msub, metric, (raw_med, raw_mean), cluster_mode, cluster_agg,
            n_resamples, stability_resamples, seed, alpha, direction)
        if verdict is None:
            skipped.append({'metric': metric, 'note': note})
            logging.info('winner_analysis: SKIP metric %r: %s', metric, note)
            continue
        verdicts.append(verdict)
        order_rows.extend(rows[0])
        stepdown_rows.extend(rows[1])
        logging.info('winner_analysis: %s -> %s', metric, verdict['verdict'])

    # ---- consensus across metrics ----
    consensus = {
        'n_metrics_analysed': len(verdicts),
        'metrics_skipped': skipped,
        'metrics_ordering_supported': [v['metric'] for v in verdicts
                                       if v['ordering_supported']],
        'metrics_approximate_ordering': [v['metric'] for v in verdicts
                                         if v['approximate_ordering_exists']],
        'metrics_unique_winner': [v['metric'] for v in verdicts
                                  if v['unique_winner']],
        'metrics_top2': [v['metric'] for v in verdicts if v['top2_exists']],
        'metrics_no_separation': [v['metric'] for v in verdicts
                                  if v['no_separation']],
    }
    per_method = {}
    for v in verdicts:
        sep_pairs = [tuple(p) for p in v.get('separable_pairs', [])]
        for i, m in enumerate(v['ordering']):
            slot = per_method.setdefault(m, {
                'n_metrics': 0, 'mean_rank_position': 0.0,
                'n_leading_group': 0, 'n_unique_winner': 0,
                'n_dominates_all': 0, 'n_nemenyi_top': 0})
            slot['n_metrics'] += 1
            slot['mean_rank_position'] += (i + 1)
            if m in v['leading_group']:
                slot['n_leading_group'] += 1
            if v['unique_winner'] and i == 0:
                slot['n_unique_winner'] += 1
            if v.get('nemenyi_top_group') and m in v['nemenyi_top_group']:
                slot['n_nemenyi_top'] += 1
            # dominates all: significantly beats EVERY other method here
            n_beaten = sum(1 for (a, _b) in sep_pairs if a == m)
            if n_beaten == v['k_methods'] - 1:
                slot['n_dominates_all'] += 1
    for slot in per_method.values():
        if slot['n_metrics']:
            slot['mean_rank_position'] /= slot['n_metrics']
    consensus['per_method'] = {
        m: dict(sorted(s.items())) for m, s in sorted(per_method.items())}
    n_an = len(verdicts)
    gs = [v['leading_group_size'] for v in verdicts]
    consensus['modal_leading_group_size'] = (
        int(pd.Series(gs).mode().iloc[0]) if gs else None)
    consensus['summary'] = (
        F'{len(consensus["metrics_ordering_supported"])}/{n_an} metrics '
        F'support an ordering (Friedman P <= {alpha}); '
        F'{len(consensus["metrics_unique_winner"])}/{n_an} have a unique '
        F'winner; {len(consensus["metrics_top2"])}/{n_an} support a top-2 '
        F'group; {len(consensus["metrics_no_separation"])}/{n_an} show no '
        'separation between the methods.')
    settings['verdicts'] = verdicts
    settings['consensus'] = consensus
    settings['n_datasets'] = int(df['dataset'].nunique())
    settings['n_units_per_method'] = {
        m: int(n) for m, n in df.groupby('method')['pair_id'].nunique().items()}

    # ---- write outputs ----
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(order_rows).to_csv(F'{out_prefix}.order.tsv', sep='\t',
                                    index=False, na_rep='NA')
    pd.DataFrame(stepdown_rows).to_csv(F'{out_prefix}.stepdown.tsv', sep='\t',
                                       index=False, na_rep='NA')
    with open(F'{out_prefix}.json', 'w') as fh:
        json.dump(settings, fh, indent=2)
    _write_stepdown_latex_table(F'{out_prefix}.stepdown.tsv',
                                F'{out_prefix}.stepdown.tex', alpha=alpha)
    logging.info('winner_analysis: wrote %s.order.tsv, %s.stepdown.tsv, '
                 '%s.stepdown.tex, %s.json',
                 out_prefix, out_prefix, out_prefix, out_prefix)
    logging.info('winner_analysis consensus: %s', consensus['summary'])
    return settings


# --------------------------------------------------------------------------- #
# LaTeX export of the step-down table                                         #
# --------------------------------------------------------------------------- #
def _fmt_pvalue_stepdown(p, r, alpha=0.05):
    """Holm-adjusted P cell of the step-down table.

    Bold only when the separation is CONFIRMED, i.e. P <= alpha and the
    effect points the right way (r > 0: the better-ranked method a really
    outperforms b).  A significant P with r <= 0 would contradict the mean-
    rank ordering and is printed unbolded (see the caption).
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return '--'
    if p != p or p in (float('inf'), float('-inf')):
        return '--'
    if p == 0.0:
        s = '$<2.2\\times10^{-16}$'
    elif p < 0.001:
        mant, exp = F'{p:.2e}'.split('e')
        s = F'${mant}\\times10^{{{int(exp)}}}$'
    else:
        s = F'{p:.3f}'
    try:
        r = float(r)
    except (TypeError, ValueError):
        r = float('nan')
    return F'\\textbf{{{s}}}' if (p <= alpha and r == r and r > 0) else s


def _prettify(text, methods):
    """Replace raw method ids in a verdict string by their display labels."""
    out = str(text)
    for m in sorted({str(m) for m in methods}, key=len, reverse=True):
        out = out.replace(str(m), _ST._method_display_tex(m))
    return out


def stepdown_caption(verdict_lines, alpha=0.05, unit_desc='materials'):
    unit = unit_desc[:-1] if unit_desc.endswith('s') else unit_desc
    cap = (
        'Step-down analysis of scRNA-seq CNV-caller performance per benchmark '
        'metric: methods are ordered by their Friedman mean rank over '
        F'per-{unit} method medians (method $a$ is always the better-ranked '
        'method), and the table lists the adjacent Wilcoxon signed-rank '
        'comparisons from the top down to the first separable pair (all '
        'pairs when none is separable). Each row reports, in this order: '
        '$n$, the effective sample size (number of independent '
        F'{unit_desc}); $p$, the Holm--Bonferroni-adjusted two-sided Wilcoxon '
        F'signed-rank P value of the per-{unit} differences of the method medians (bold when '
        F'the separation is confirmed at the {alpha:g} family-wise level, '
        'i.e.\\ $p \\le \\alpha$ and $r > 0$); $r$, the matched-pairs '
        'rank-biserial effect size; and the \\qty{95}{\\percent} percentile-bootstrap CI of '
        F'$r$ over the {unit_desc}. A unique winner exists iff the first '
        'row of a metric is separable; a top-2 group exists iff it is not '
        'separable and the second row is. CN, copy-number; CNV, copy-number '
        'variation.')
    if verdict_lines:
        cap += ' Verdicts: ' + ' '.join(verdict_lines)
    return cap


def stepdown_latex_table_lines(tsv_path, alpha=0.05,
                               table_label='tab:scrna-winner'):
    """Build the booktabs step-down table from a *.stepdown.tsv file.

    Every row (one adjacent step-down comparison) carries, after the
    identity columns (Metric, Method a, Method b), exactly these four
    statistics in this order and nothing else: n, p (Holm-adjusted), r
    (rank-biserial) and the 95% CI of r.  Returns the list of table lines,
    or None on any problem (a reason is logged).
    """
    if not os.path.isfile(tsv_path):
        logging.error('winner_analysis: stepdown file not found: %s', tsv_path)
        logging.error('run the winner analysis first (plot_cnv_heatmaps.py '
                      'runs it by default, or use winner_analysis.py -i/-o)')
        return None
    tab = pd.read_csv(tsv_path, sep='\t')
    need = ['metric', 'step', 'method_a', 'method_b', 'n', 'pvalue_holm',
            'rank_biserial_r', 'ci95_r_low', 'ci95_r_high', 'is_shown']
    missing = [c for c in need if c not in tab.columns]
    if missing:
        logging.error('winner_analysis: column(s) %s missing from %s',
                      missing, tsv_path)
        return None
    sub = tab[tab['is_shown'].astype(str).str.lower().isin(('true', '1'))].copy()
    sub = sub.dropna(subset=['metric', 'method_a', 'method_b'])
    if sub.empty:
        logging.error('winner_analysis: no step-down rows in %s', tsv_path)
        return None

    # inference level for the caption's unit description
    unit_desc = 'materials'
    if 'level' in tab.columns and len(tab) \
            and str(tab['level'].iloc[0]).startswith('unit (naive)'):
        unit_desc = 'paired units'

    metric_order = list(dict.fromkeys(sub['metric'].tolist()))
    sub['metric'] = pd.Categorical(sub['metric'], categories=metric_order,
                                   ordered=True)
    sub = sub.sort_values(['metric', 'step'])

    # per-metric verdict lines for the caption (first row of each metric)
    verdict_lines = []
    methods_seen = set(sub['method_a'].astype(str)) | set(sub['method_b'].astype(str))
    for m in metric_order:
        block = tab[tab['metric'] == m]
        if len(block) and 'metric_verdict' in block.columns:
            v = _prettify(str(block['metric_verdict'].iloc[0]), methods_seen)
            verdict_lines.append(F'\\emph{{{_ST._tex_escape(str(m))}}}: '
                                 F'{_ST._tex_escape(v)}.')

    lines = [
        '% LaTeX table generated by winner_analysis.py '
        '(requires \\usepackage{booktabs})',
        '\\begin{landscape}',
        F'\\captionof{{table}}{{{stepdown_caption(verdict_lines, alpha=alpha, unit_desc=unit_desc)}}}',
        F'\t\\label{{{table_label}}}',
        '\t\\scriptsize',
        '\t\\begin{longtable}{lllrrrr}',
        '\t\t\\toprule',
        '\t\tMetric & Method $a$ & Method $b$ & $n$ & $p$ & $r$ & '
        '\\qty{95}{\\percent} CI \\\\',
        '\t\t\\midrule',
        '\t\t\\endfirsthead',
        '\t\t\\toprule',
        '\t\tMetric & Method $a$ & Method $b$ & $n$ & $p$ & $r$ & '
        '\\qty{95}{\\percent} CI \\\\',
        '\t\t\\midrule',
        '\t\t\\endhead',
        '\t\t\\bottomrule',
        '\t\t\\endfoot',
    ]
    prev_metric = None
    for rec in sub.itertuples(index=False):
        if prev_metric is not None and rec.metric != prev_metric:
            lines.append('\t\t\\midrule')
        prev_metric = rec.metric
        cells = [
            _ST._tex_escape(str(rec.metric)),
            _ST._tex_escape(_ST._method_display_tex(rec.method_a)),
            _ST._tex_escape(_ST._method_display_tex(rec.method_b)),
            _ST._fmt_signed_int(rec.n),
            _fmt_pvalue_stepdown(rec.pvalue_holm, rec.rank_biserial_r, alpha),
            _ST._fmt_effect(rec.rank_biserial_r),
            _ST._fmt_ci(rec.ci95_r_low, rec.ci95_r_high),
        ]
        lines.append(F'\t\t{" & ".join(cells)} \\\\')
    lines += [
        '\t\t\\bottomrule',
        '\t\\end{longtable}',
        '% the longtable environment steps the table counter itself when the caption',
        '% package is loaded, in addition to the \\captionof above; undo that second step',
        '% so the table keeps its number',
        '\\addtocounter{table}{-1}',
        '\\end{landscape}',
    ]
    return lines


def emit_stepdown_latex_table(tsv_path, alpha=0.05,
                              table_label='tab:scrna-winner'):
    """Print the booktabs step-down table from a *.stepdown.tsv file.
    Returns 0 on success, 1 on any problem."""
    lines = stepdown_latex_table_lines(tsv_path, alpha=alpha,
                                       table_label=table_label)
    if lines is None:
        return 1
    print('\n'.join(lines))
    return 0


def _write_stepdown_latex_table(tsv_path, tex_path, alpha=0.05):
    """Write the step-down table to `tex_path` (default pipeline output).
    Never raises - a failure of the table must not break the benchmark run."""
    try:
        lines = stepdown_latex_table_lines(tsv_path, alpha=alpha)
        if lines is None:
            return 1
        with open(tex_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        logging.info('winner_analysis: LaTeX step-down table written to %s',
                     tex_path)
        return 0
    except Exception as exc:  # pragma: no cover
        logging.warning('winner_analysis: could not write the LaTeX table %s: %s',
                        tex_path, exc)
        return 1


# --------------------------------------------------------------------------- #
# Command-line interface                                                       #
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Ordering / unique-winner / top-2 tests for the '
                    'scRNA-seq CNV-caller benchmark (Friedman + Kendall W '
                    'omnibus; all-pairs two-sided Wilcoxon signed-rank with '
                    'Holm; step-down leading group; Nemenyi cross-check; '
                    'bootstrap rank stability). The LaTeX step-down table '
                    'carries exactly n, p, r and the 95% CI of r.')
    parser.add_argument('-i', '--input', nargs='*', default=None,
                        help='Input table(s): raw evaluation TSVs (dataset '
                             'inferred from the path) or one unified long TSV '
                             'with a dataset column. Globs allowed.')
    parser.add_argument('-o', '--output', default='scrna-winner',
                        help='Output prefix; writes <prefix>.order.tsv, '
                             '<prefix>.stepdown.tsv, <prefix>.stepdown.tex '
                             'and <prefix>.json.')
    parser.add_argument('--cluster-key', default=None,
                        help='Independent unit: material (default), dataset, '
                             'none (naive), or a comma-separated column list.')
    parser.add_argument('--cluster-agg', choices=['median', 'mean'],
                        default='median',
                        help='Aggregation within each cluster (default median).')
    parser.add_argument('--metrics', default=None,
                        help='Comma-separated metric names to analyse.')
    parser.add_argument('--lower-is-better', default=None, metavar='M1,M2',
                        help='Metrics for which SMALLER values are better '
                             '(default: every metric is higher-is-better).')
    parser.add_argument('--boot', type=int, default=10000,
                        help='Bootstrap resamples for the CIs of r '
                             '(default 10000; capped at 5000 on clusters).')
    parser.add_argument('--stability-boot', type=int,
                        default=DEFAULT_STABILITY_RESAMPLES,
                        help='Bootstrap resamples for the rank-stability '
                             'analysis (default 2000).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed of the bootstrap RNG (default 1).')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Family-wise significance level (default 0.05).')
    parser.add_argument('--no-latex-table', dest='latex_table_auto',
                        action='store_false', default=True,
                        help='Do not write <prefix>.stepdown.tex.')
    parser.add_argument('--latex-table', action='store_true', default=False,
                        help='Print the booktabs table built from the '
                             'existing <prefix>.stepdown.tsv and exit.')
    args = parser.parse_args(argv)

    if not _HAVE_SCIPY:
        raise SystemExit('winner_analysis: scipy is required (pip install scipy)')

    if args.latex_table:
        tsv = F'{args.output}.stepdown.tsv'
        return emit_stepdown_latex_table(tsv, alpha=args.alpha)

    if not args.input:
        parser.error('no input given: pass -i <files/globs>')
    df = _ST._load_long_frames(args.input, file_pattern=None)
    print(F'loaded {len(df)} rows, {df["dataset"].nunique()} datasets, '
          F'{df["method"].nunique()} methods, {df["metric"].nunique()} metrics')

    metrics = ([m.strip() for m in args.metrics.split(',')]
               if args.metrics else None)
    directions = {}
    if args.lower_is_better:
        for m in args.lower_is_better.split(','):
            if m.strip():
                directions[m.strip()] = 'lower'
    settings = run_scrna_winner_analysis(
        df, args.output,
        cluster_key_cols=_ST._parse_cluster_key(args.cluster_key),
        cluster_agg=args.cluster_agg,
        n_resamples=args.boot,
        stability_resamples=args.stability_boot,
        seed=args.seed, alpha=args.alpha, metrics=metrics,
        metric_directions=directions)
    if settings and not args.latex_table_auto:
        # keep the .json even when the LaTeX table is suppressed
        with open(F'{args.output}.json', 'w') as fh:
            json.dump(settings, fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
