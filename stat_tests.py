#!/usr/bin/env python3
"""stat_tests.py - Statistical tests for the scRNA-seq CNV-caller benchmark.

This module mirrors the statistical design of the companion scWGS repository
copy-num-bench-scwgs (bench_results/stat_tests.py, v3/v4) and adapts it to the
scRNA-seq co-sequencing benchmark evaluated by plot_cnv_heatmaps.py.

Design rationale (why these tests)
==================================
The scRNA-seq benchmark is a randomized-complete-block design: every CNV
caller (method) is evaluated on the SAME cells of the SAME datasets, so
per-cell performances are paired (blocked) by cell. The performance metrics
(Pearson/Spearman correlation, gain/loss ROC-AUC, F-scores, classification
accuracy) are bounded and non-normal, so nonparametric tests are used
throughout, and all comparisons are two-sided (two-tailed):

1.  Omnibus per metric: Friedman test (the repeated-measures rank ANOVA)
    across the k methods, with Kendall's W as concordance effect size and the
    per-method mean ranks, run on per-cluster medians ('cluster' rows) and,
    for transparency, on the raw per-unit rows ('unit (naive)' rows).
2.  Post-hoc pairwise: two-sided Wilcoxon signed-rank tests, reference method
    vs. every other method, paired per cell within each dataset
    (Pratt-style zero handling, zero_method='zsplit'), Holm-Bonferroni
    family-wise correction within each (metric) family of comparisons. All
    pairwise comparisons are available with --all-pairs.
3.  Effect sizes per comparison: matched-pairs rank-biserial correlation r
    (positive = reference performs better) with a 95% percentile-bootstrap
    confidence interval obtained by resampling the independent units (seeded,
    hence fully reproducible), the paired common-language effect size
    P(ref > other) + 0.5*P(ref == other), and the median difference with a
    95% CI.

Independence assumptions (READ THIS)
====================================
WHAT IS MODELLED: every test is paired/blocked. All k methods are run on the
same cells of the same datasets; the cross-method correlation within one cell
is exactly what the pairing (blocking) accounts for. No independent-samples
test is used anywhere.

WHAT IS ASSUMED: a randomized complete block design further requires the
BLOCKS to be mutually independent - i.e. that the many per-cell results
produced by the SAME method across blocks are independent draws. For this
benchmark that assumption is NOT credible at two nesting levels:

* WITHIN a dataset, all per-cell evaluations share the library, the reference
  set and the cell composition, so per-cell results of one method (and the
  paired differences between two methods) are positively correlated.
* ACROSS datasets, several datasets share the same biological material: the
  wellDR-seq chips of one patient (BCIS106T_chip1/chip2, ECIS44T_chip1-5,
  MDA231_chip1/2, ...), the same cell line sequenced by different
  technologies or with different reference-cell configurations (HCT116 in
  DNTR-seq and scONE-seq; scONE-seq_HCT116 with and without given reference
  cells). A method confused by that shared material repeats the error on
  every dataset carrying it.

WHAT HAPPENS IF THE ASSUMPTION FAILS: classic pseudoreplication. The null
variance of the Wilcoxon / Friedman statistics assumes independent blocks;
with intra-cluster correlation rho and average cluster size m the variance
is inflated by the design effect DE = 1 + (m - 1) * rho, so the effective
sample size is only n_eff = n / DE: test statistics are too large, P values
orders of magnitude too small, Holm no longer controls the family-wise
error, and the benchmark 'winner' can be decided by ONE patient whose
library favours a method.

THE FIX (default behaviour)
---------------------------
Inference is moved to the level of the independent experimental unit - the
biological MATERIAL (patient or cell line) behind the datasets - while
per-cell quantities are kept as DESCRIPTIVE statistics and as flagged naive
(non-inferential) comparisons:

* Default cluster key: `material`, derived from the dataset name (chips,
  run/SHA accessions and technology prefixes are stripped, so all datasets of
  one patient / cell line collapse into one cluster). Each material is one
  effective sample; n in the LaTeX table is the number of materials.
  `--cluster-key dataset` analyses at the per-dataset level (a sensitivity
  analysis that treats chips of one patient as independent), and
  `--cluster-key none` reverts to the naive per-cell tests (discouraged).
* Per comparison, the paired per-cell differences d = x - y are aggregated to
  per-cluster medians d_g (one value per cluster); the two-sided Wilcoxon
  signed-rank test and the exact two-sided sign test run on the d_g, and
  Holm-Bonferroni is applied to the CLUSTER-level P values.
* The Friedman omnibus likewise runs on per-cluster method medians.
* The 95% CI of the effect size r comes from a percentile bootstrap over the
  CLUSTER-level medians (the independent units), not from an i.i.d. cell
  bootstrap.
* Diagnostics per comparison: ICC(1,1) of the paired differences within
  clusters, the design effect, the effective sample size, and the
  naive-vs-cluster P-value ratio - the degree of dependence in the actual
  data is measured, not assumed away.

LaTeX table (the deliverable of run_scrna_benchmark_stats)
==========================================================
<prefix>.pairwise.tex is a copy-paste-ready booktabs table in which
every row (one metric x method_b comparison) carries, in this order and
NOTHING else:
    n   effective_sample_size (number of independent materials)
    p   Holm-adjusted two-sided Wilcoxon signed-rank P value
    r   effect size (matched-pairs rank-biserial correlation)
    CI  95% percentile-bootstrap confidence interval of r

Outputs (prefix = -o/--output)
==============================
<prefix>.pairwise.tsv          one row per comparison: inference level,
                               n_clusters / n_units_paired, medians, median
                               difference, W, two-sided P (cluster level by
                               default), sign-test P, Holm-adjusted P,
                               rank-biserial r + its 95% bootstrap CI
                               (ci95_r_low/high), CL effect size, 95% CI of
                               the median difference, naive P +
                               rank-biserial, ICC, design effect, effective
                               n, P-inflation ratio, notes
<prefix>.friedman.tsv          omnibus Friedman chi2, df, P, Kendall's W,
                               per-method mean ranks; rows at both levels:
                               'cluster' (primary) and 'unit (naive)'
<prefix>.json                  settings, exact n per analysis, the
                               independence/clustering record, versions, seed
<prefix>.pairwise.tex          the booktabs LaTeX table (see above)

Usage
=====
Standalone, on the raw evaluation TSVs (dataset inferred from each file's
path exactly like plot_cnv_heatmaps.py does):
    python stat_tests.py -i 'results/*/evaluation/*.tsv' \
        -o heatmaps/stats --reference infercnv
On one already-unified long TSV with an explicit dataset column:
    python stat_tests.py -i aggregated_per_cell_long.tsv -o heatmaps/stats
From plot_cnv_heatmaps.py (the default): the stats run on the SAME filtered
method set as the figures and the table is written next to them (default
prefix <outdir>/stats, i.e. heatmaps/stats.pairwise.{tsv,tex}).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import platform
import re
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats as sps
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - the frozen env always has scipy
    _HAVE_SCIPY = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s')

# Pair identity within one dataset: the eval TSVs carry the matched
# (ground_truth_cell, caller_cell) pair per row. Rows without a cell id (the
# per-dataset classification metrics folded in by plot_cnv_heatmaps.py) pair
# at the dataset level, which is their only level of replication.
DEFAULT_CELL_KEY = ['dataset', 'ground_truth_cell']

# Cluster bootstrap runtime cap (the pairwise loop can run >100 comparisons).
MAX_CLUSTER_BOOTSTRAP = 5000

TINY = float(np.finfo(float).tiny)

# Official capitalisation + publication year of the scRNA-seq CNV callers
# (kept in sync with plot_cnv_heatmaps.py; used for the LaTeX method labels
# and for the chronological column order of the table).
METHOD_PUBLICATION = {
    'infercnv' : ('inferCNV',  2014, '2014-06-12'),
    'conicsmat': ('CONICSmat', 2018, '2018-09-01'),
    'infercna' : ('inferCNA',  2019, '2019-07-18'),
    'casper'   : ('CaSpER',    2020, '2020-01-03'),
    'copykat'  : ('CopyKAT',   2021, '2021-01-18'),
    # Label year = the reference-list (issue) year 2023; the exact date stays
    # the online date so the chronological column order is unchanged.
    'numbat'   : ('Numbat',    2023, '2022-09-26'),
    'scevan'   : ('SCEVAN',    2023, '2023-02-25'),
}

# Display names of the per-configuration tokens that may follow the caller name
# in a raw `method` value (e.g. 'copykat_cellline' -> 'CopyKAT 2021' plus the
# 'cell-line mode' label below).  Internal pipeline tokens must never reach a
# printed table.
METHOD_SUFFIX_DISPLAY_NAMES = {
    'cellline': 'cell-line mode',
}


# --------------------------------------------------------------------------- #
# Core, dependency-light statistics helpers                                   #
# (identical semantics to copy-num-bench-scwgs/bench_results/stat_tests.py)   #
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals):
    """Holm-Bonferroni family-wise adjusted p-values (step-down)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m, dtype=float)
    order = np.argsort(p, kind='stable')
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running_max = max(running_max, val)
        adj[i] = min(1.0, running_max)
    return adj


def rank_biserial_matched(d):
    """Matched-pairs rank-biserial correlation for paired differences d.

    r = (R+ - R-) / (n(n+1)/2), Pratt-consistent: zero differences contribute
    half of their rank to each side (matching zero_method='zsplit').
    r > 0 means x tends to be larger than y (reference better when d = x - y).
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float('nan')
    ranks = sps.rankdata(np.abs(d))
    zero = d == 0
    r_pos = float(np.sum(ranks[d > 0]) + 0.5 * np.sum(ranks[zero]))
    r_neg = float(np.sum(ranks[d < 0]) + 0.5 * np.sum(ranks[zero]))
    total = n * (n + 1.0) / 2.0
    return (r_pos - r_neg) / total


def common_language_paired(d):
    """P(d > 0) + 0.5*P(d == 0): probability that the reference scores higher
    than the competitor on a random paired unit (ties count half)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float('nan')
    return float(np.mean(d > 0) + 0.5 * np.mean(d == 0))


def wilcoxon_signed_rank(x, y, alternative='two-sided', zero_method='zsplit'):
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns a dict with n_pairs, n_zero, n_nonzero, statistic, pvalue and an
    optional note. Complete pairs only (both values finite).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(F'wilcoxon_signed_rank: shape mismatch {x.shape} vs {y.shape}')
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    d = x - y
    out = {
        'n_pairs': int(len(d)),
        'n_zero': int(np.sum(d == 0)),
        'n_nonzero': int(np.sum(d != 0)),
        'statistic': float('nan'),
        'pvalue': float('nan'),
        'note': '',
    }
    if len(d) == 0:
        out['note'] = 'no complete pairs'
        return out
    if np.all(d == 0):
        out['statistic'] = 0.0
        out['pvalue'] = 1.0
        out['note'] = 'all paired differences are zero'
        return out
    try:
        with warnings.catch_warnings():
            # small cluster counts make scipy warn about the normal
            # approximation while still returning a valid exact/asymptotic P
            warnings.simplefilter('ignore', UserWarning)
            res = sps.wilcoxon(x, y, zero_method=zero_method, alternative=alternative,
                               correction=True, method='auto')
        out['statistic'] = float(res.statistic)
        out['pvalue'] = float(res.pvalue)
    except (ValueError, RuntimeWarning) as exc:  # degenerate tiny samples
        out['note'] = F'Wilcoxon undefined on this sample ({exc}); P set to 1'
        out['pvalue'] = 1.0
        return out
    if out['pvalue'] <= TINY:
        out['note'] = F'asymptotic two-sided P underflows double precision (P < {TINY:.1e})'
    return out


def sign_test_two_sided(d):
    """Exact two-sided sign test on paired differences (zeros dropped).

    Complements the Wilcoxon signed-rank test when the number of independent
    units is too small for the exact signed-rank null to reach P < 0.05.
    Returns dict(n, k_positive, pvalue)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return {'n': 0, 'k_positive': pos, 'pvalue': 1.0}
    p = min(1.0, 2.0 * sps.binom.cdf(min(pos, neg), n, 0.5))
    return {'n': n, 'k_positive': pos, 'pvalue': float(p)}


def bca_bootstrap_ci(d, statistic=np.median, n_resamples=10000,
                     confidence_level=0.95, seed=1):
    """BCa bootstrap CI of a statistic of paired differences (seeded).

    NAIVE-MODE ONLY (assumes i.i.d. observations): used when clustering is
    disabled (--cluster-key none). Falls back to the percentile method for
    degenerate samples, then to the point estimate itself.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float('nan'), float('nan'), 'no data'
    point = float(statistic(d))
    if np.all(d == d[0]):
        return point, point, 'degenerate sample (all differences equal)'
    rng = np.random.default_rng(seed)
    for method in ('BCa', 'percentile'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = sps.bootstrap((d,), statistic, n_resamples=int(n_resamples),
                                    confidence_level=confidence_level, method=method,
                                    random_state=rng)
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi, method
            logging.debug('bootstrap (%s) returned a non-finite interval', method)
        except Exception as exc:  # degenerate jackknife / too few distinct values
            logging.debug('bootstrap (%s) failed: %s', method, exc)
    return point, point, 'bootstrap degenerate; point estimate returned'


def bootstrap_r_ci(units, n_resamples=2000, confidence_level=0.95, seed=1):
    """Percentile-bootstrap CI of the matched-pairs rank-biserial effect size r.

    `units` are the i.i.d. paired differences AT THE INFERENCE LEVEL: the
    per-cluster (material) medians in cluster mode, or the raw per-unit
    differences in naive mode. The independent units are resampled with
    replacement, r is recomputed on every resample, and the percentile
    interval of the r distribution is returned. This targets the same estimand
    as the reported rank_biserial_r, so (r, CI) describe ONE effect size.
    Degenerate samples fall back to the point estimate with an explanatory
    method string. Returns (low, high, method).
    """
    u = np.asarray(units, dtype=float)
    u = u[np.isfinite(u)]
    if len(u) == 0:
        return float('nan'), float('nan'), 'no data'
    point = rank_biserial_matched(u)
    if len(u) < 2:
        return point, point, 'single unit: CI undefined'
    if np.all(u == u[0]):
        return point, point, 'degenerate sample (all differences equal)'
    rng = np.random.default_rng(seed)
    b = int(n_resamples)
    rs = np.empty(b, dtype=float)
    for i in range(b):
        pick = rng.integers(0, len(u), len(u))
        rs[i] = rank_biserial_matched(u[pick])
    a = 1.0 - float(confidence_level)
    lo, hi = np.quantile(rs, [a / 2.0, 1.0 - a / 2.0])
    return float(lo), float(hi), 'unit-percentile'


def icc_design_effect(d, cluster_ids):
    """ICC(1,1) of clustered observations + design effect + effective n.

    One-way random-effects ANOVA, method of moments:
        ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW),
    with the unbalanced-size adjustment
        m0 = (N - sum(n_g^2)/N) / (G - 1).
    Design effect DE = 1 + (m0 - 1) * max(ICC, 0) (a negative ICC estimate is
    reported as-is but treated as 0 for DE); effective sample size
    n_eff = N / DE. This quantifies, on the actual data, how badly a per-cell
    (naive) test would overstate significance. Returns a dict."""
    d = np.asarray(d, dtype=float)
    cl = np.asarray(list(cluster_ids), dtype=object)
    if len(d) != len(cl):
        raise ValueError('icc_design_effect: d and cluster_ids length mismatch')
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    n = int(len(d))
    out = {'n_obs': n, 'n_clusters': 0, 'icc': float('nan'), 'm0': float('nan'),
           'design_effect': float('nan'), 'n_effective': float('nan')}
    uniq, inv = np.unique(cl, return_inverse=True)
    g = int(len(uniq))
    out['n_clusters'] = g
    if g < 2 or n <= g or n < 3:
        return out
    if np.all(d == d[0]):
        return out  # zero variance: ICC undefined
    cnt = np.bincount(inv, minlength=g).astype(float)
    sums = np.bincount(inv, weights=d, minlength=g)
    means = sums / cnt
    grand = float(d.mean())
    ssb = float(np.sum(cnt * (means - grand) ** 2))
    ssw = float(np.sum(d ** 2) - np.sum(cnt * means ** 2))
    msb = ssb / (g - 1.0)
    msw = ssw / (n - g)
    m0 = (n - float(np.sum(cnt ** 2)) / n) / (g - 1.0)
    denom = msb + (m0 - 1.0) * msw
    icc = float('nan')
    if denom > 0:
        icc = float((msb - msw) / denom)
    de = float(1.0 + (m0 - 1.0) * max(icc, 0.0)) if np.isfinite(icc) else float('nan')
    out.update({'icc': icc, 'm0': float(m0), 'design_effect': de,
                'n_effective': float(n / de) if np.isfinite(de) and de > 0 else float('nan')})
    return out


def friedman_test(samples):
    """Friedman test on complete blocks. samples: list of k equal-length arrays.

    Returns dict(chi2, df, pvalue, n_blocks, k, kendalls_w, mean_ranks).
    NOTE: the blocks are assumed mutually independent - with clustered blocks
    run this on per-cluster aggregated values, not on raw per-cell values.
    """
    mat = np.column_stack([np.asarray(s, dtype=float) for s in samples])
    n, k = mat.shape
    if k < 3 or n < 2:
        return None
    res = sps.friedmanchisquare(*[mat[:, j] for j in range(k)])
    mean_ranks = sps.rankdata(mat, axis=1).mean(axis=0)
    return {
        'chi2': float(res.statistic),
        'df': int(k - 1),
        'pvalue': float(res.pvalue),
        'n_blocks': int(n),
        'k': int(k),
        'kendalls_w': float(res.statistic / (n * (k - 1))),
        'mean_ranks': [float(r) for r in mean_ranks],
    }


# --------------------------------------------------------------------------- #
# Cluster bookkeeping                                                         #
# --------------------------------------------------------------------------- #
def _norm_missing(v):
    """Normalise a metadata value to '' when missing, else to its str form."""
    if v is None:
        return ''
    if isinstance(v, float) and np.isnan(v):
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v).strip()


# Accession / run / study id fragments removed when deriving the shared
# biological material from a dataset name.
_MAT_ACCESSION_RE = re.compile(r'_(?:SAMN|SRR|ERR|DRR|SRP|PRJNA|GSM|HCA)[0-9A-Za-z]*')
# Technology prefixes of the co-sequencing studies.
_MAT_TECH_RE = re.compile(r'^(?:wellDR-seq|scONE-seq|DNTR-seq|wellDR)_')
# Chip / replicate suffixes of one patient's sample.
_MAT_CHIP_RE = re.compile(r'_chip[0-9]+')


def material_from_dataset(dataset):
    """Independent biological material (patient / cell line) behind a dataset.

    Datasets of the benchmark are named like
    'BCIS106T_chip1_SAMN48409192_SRR33511671' or 'scONE-seq_HCT116_HUVEC_H9_as_T_N':
    the shared material is recovered by stripping the chip/replicate suffix,
    the run/sample/study accessions and the technology prefix. Datasets that
    share the resulting token (chips of one patient; one cell line sequenced
    by different technologies) form ONE cluster, because the per-method
    results on them are correlated through the shared material. A token that
    is empty or purely numeric falls back to the full dataset name (each
    dataset its own unit).

    Reference-cell configuration variants of one dataset (e.g. scONE-seq HCT116
    run with and without given HUVEC/H9 reference cells) collapse onto their
    base material as well - see _collapse_materials, applied by
    run_scrna_benchmark_stats in the default mode.
    """
    s = _norm_missing(dataset)
    if not s:
        return '(missing)'
    s = _MAT_CHIP_RE.sub('', s)
    s = _MAT_ACCESSION_RE.sub('', s)
    s = _MAT_TECH_RE.sub('', s)
    s = s.strip('_ ')
    if not s or s.isdigit():
        return str(dataset)
    return s


def _collapse_materials(ids):
    """Map reference-cell configuration variants onto their base material.

    A material whose underscore-token sequence strictly EXTENDS another
    occurring material is the same biological sample analysed with a
    different reference-cell configuration (e.g. 'HCT116_HUVEC_H9_as_T_N'
    extends 'HCT116'): the evaluated tumour cells are the same, so the two
    configurations are ONE independent unit and the variant collapses onto
    the shortest present token-prefix. Materials that are not prefix-related
    (e.g. 'Cellline_mixing_experiment_1_1' vs '..._1_2' vs '..._2', which are
    different mixtures) stay separate. Returns {material: base_material}.
    """
    id_list = [str(i) for i in ids]
    id_set = set(id_list)
    mapping = {}
    for m in id_list:
        toks = m.split('_')
        base = m
        for k in range(1, len(toks)):
            cand = '_'.join(toks[:k])
            if cand in id_set:
                base = cand
                break          # shortest present token-prefix wins
        mapping[m] = base
    return mapping


def _pairwise_record(x, y, labels, base, cluster_agg='median', n_resamples=10000,
                     seed=1, cluster_key_str=''):
    """One pairwise-comparison record.

    Primary inference on the CLUSTER level (per-cluster/material aggregation
    of the per-unit differences), naive per-unit comparison kept for
    transparency.
    x, y: paired per-unit values, equal length; labels: cluster id per
    element (None -> naive mode). base: dict with the identity columns
    (metric/method_a/method_b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if labels is not None and len(labels) != len(x):
        raise ValueError('_pairwise_record: labels length does not match x')
    ok = np.isfinite(x) & np.isfinite(y)
    if labels is not None:
        labels = [l for l, m in zip(labels, ok) if m]
    x, y = x[ok], y[ok]
    d = x - y
    rec = dict(base)
    w_cell = wilcoxon_signed_rank(x, y)
    notes = [n for n in (w_cell['note'],) if n]
    rec.update({
        'inference_level': 'cluster' if labels is not None else 'unit (naive)',
        'cluster_key': cluster_key_str if labels is not None else '',
        'n_units_paired': w_cell['n_pairs'],
        'median_a': float(np.median(x)) if len(x) else float('nan'),
        'median_b': float(np.median(y)) if len(y) else float('nan'),
        'median_diff_a_minus_b': float(np.median(d)) if len(d) else float('nan'),
        'mean_diff_a_minus_b': float(np.mean(d)) if len(d) else float('nan'),
        'cl_effect_paired': common_language_paired(d),
        'rank_biserial_r_unit_naive': rank_biserial_matched(d),
        'wilcoxon_W_unit_naive': w_cell['statistic'],
        'pvalue_unit_naive': w_cell['pvalue'],
        'n_zero_diffs': w_cell['n_zero'],
    })
    if labels is not None:
        cl_ser = pd.Series(d, index=pd.Index(labels, dtype=object))
        cm = cl_ser.groupby(level=0).agg(cluster_agg).sort_index()
        cm_v = cm.to_numpy(dtype=float)
        w_cl = wilcoxon_signed_rank(cm_v, np.zeros_like(cm_v))
        st = sign_test_two_sided(cm_v)
        n_boot = int(min(n_resamples, MAX_CLUSTER_BOOTSTRAP))
        ci_lo, ci_hi, ci_method = bca_bootstrap_ci(
            d, np.median, n_resamples=max(n_boot, 200), seed=seed)
        r_lo, r_hi, r_ci_method = bootstrap_r_ci(
            cm_v, n_resamples=max(n_boot, 200), seed=seed)
        icc = icc_design_effect(d, labels)
        rec.update({
            'n_pairs': int(len(cm_v)),            # units the primary test runs on
            'n_clusters': int(len(cm_v)),
            'wilcoxon_W': w_cl['statistic'],
            'pvalue_two_sided': w_cl['pvalue'],
            'pvalue_sign_test': st['pvalue'],
            'rank_biserial_r': rank_biserial_matched(cm_v),
            'ci95_r_low': r_lo, 'ci95_r_high': r_hi, 'ci_r_method': r_ci_method,
            'ci95_median_diff_low': ci_lo, 'ci95_median_diff_high': ci_hi,
            'ci_method': ci_method,
            'icc_within_cluster_d': icc['icc'],
            'design_effect': icc['design_effect'],
            'n_effective_units': icc['n_effective'],
            'mean_cluster_size_m0': icc['m0'],
        })
        if w_cl['note']:
            notes.append(w_cl['note'])
        if 0 < len(cm_v) < 6:
            notes.append(F'only {len(cm_v)} independent clusters: the exact '
                         'two-sided Wilcoxon cannot reach P < 0.05 below 6 '
                         'units; see pvalue_sign_test')
        if np.isfinite(w_cl['pvalue']) and np.isfinite(w_cell['pvalue']):
            if w_cl['pvalue'] < 1e-12 and w_cell['pvalue'] < 1e-12:
                rec['p_inflation_ratio'] = float('nan')  # both underflow
                notes.append('both naive and cluster P underflow; ratio undefined')
            else:
                # > 1 means the naive per-unit P overstates significance
                rec['p_inflation_ratio'] = float(
                    w_cl['pvalue'] / max(w_cell['pvalue'], TINY))
        else:
            rec['p_inflation_ratio'] = float('nan')
    else:
        st = sign_test_two_sided(d)
        ci_lo, ci_hi, ci_method = bca_bootstrap_ci(
            d, np.median, n_resamples=n_resamples, seed=seed)
        r_lo, r_hi, r_ci_method = bootstrap_r_ci(
            d, n_resamples=min(n_resamples, MAX_CLUSTER_BOOTSTRAP), seed=seed)
        rec.update({
            'n_pairs': w_cell['n_pairs'],
            'n_clusters': float('nan'),
            'wilcoxon_W': w_cell['statistic'],
            'pvalue_two_sided': w_cell['pvalue'],
            'pvalue_sign_test': st['pvalue'],
            'rank_biserial_r': rank_biserial_matched(d),
            'ci95_r_low': r_lo, 'ci95_r_high': r_hi, 'ci_r_method': r_ci_method,
            'ci95_median_diff_low': ci_lo, 'ci95_median_diff_high': ci_hi,
            'ci_method': ci_method,
            'icc_within_cluster_d': float('nan'),
            'design_effect': float('nan'),
            'n_effective_units': float('nan'),
            'mean_cluster_size_m0': float('nan'),
            'p_inflation_ratio': 1.0,
        })
        notes.append('NAIVE per-unit level: per-cell results treated as '
                     'independent (pseudoreplication risk); rerun with a '
                     '--cluster-key for valid inference')
    rec['note'] = '; '.join(notes)
    return rec


def software_versions():
    info = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }
    if _HAVE_SCIPY:
        info['scipy'] = scipy.__version__
    return info


# --------------------------------------------------------------------------- #
# The scRNA-seq benchmark statistics                                          #
# --------------------------------------------------------------------------- #
def run_scrna_benchmark_stats(df, out_prefix, reference='infercnv',
                              all_pairs=False, cluster_key_cols=None,
                              cluster_agg='median', n_resamples=10000,
                              seed=1, alpha=0.05, metrics=None):
    """Statistical tests for the scRNA-seq CNV-caller benchmark.

    df: long dataframe, one row per (dataset x cell x method x metric) - or
    per (dataset x method x metric) for the dataset-level classification
    metrics - with at least the columns (dataset, method, metric, value).
    Extra columns may be used through cluster_key_cols.
    cluster_key_cols: None -> the derived 'material' (patient / cell line)
    is the independent unit; ['dataset'] -> per-dataset level (sensitivity
    analysis); [] -> naive per-unit mode; a list -> custom columns.
    Writes <out_prefix>.pairwise.tsv, <out_prefix>.friedman.tsv,
    <out_prefix>.json and the booktabs LaTeX table
    <out_prefix>.pairwise.tex with exactly n, p, r, CI.
    Returns the dict that is also written to .json.
    """
    if not _HAVE_SCIPY:
        logging.warning('scipy is not available: statistical tests skipped')
        return None
    needed = {'dataset', 'method', 'metric', 'value'}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise SystemExit(F'stat_tests: input lacks columns {sorted(missing_cols)} '
                         '(need dataset, method, metric, value)')
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    methods_present = sorted(df['method'].astype(str).unique())
    if reference not in methods_present:
        raise SystemExit(F'stat_tests: reference method {reference!r} not among the '
                         F'available methods {methods_present}')

    # ---- resolve the cluster key (the independent experimental unit) ----
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
        dropped = [c for c in cluster_key_cols if c not in df.columns]
        cluster_key_cols = [c for c in cluster_key_cols if c in df.columns]
        if dropped:
            logging.warning('cluster-key columns %s not present in the input; '
                            'clustering uses the remaining %s',
                            dropped, cluster_key_cols or 'nothing')
        cluster_key_source = 'user-specified columns: ' + (', '.join(cluster_key_cols) or '(none)')
        if not cluster_key_cols:
            cluster_mode = 'naive'
            logging.warning('no usable cluster columns: falling back to the NAIVE '
                            'per-unit level (pseudoreplication risk)')

    if cluster_mode == 'naive':
        df['cluster_id'] = ''
    elif cluster_mode == 'material':
        df['cluster_id'] = df['dataset'].map(material_from_dataset)
        # reference-cell configuration variants of one base dataset collapse
        # onto their base material (one independent unit per sample)
        _mat_map = _collapse_materials(df['cluster_id'].unique())
        df['cluster_id'] = df['cluster_id'].map(_mat_map)
    elif cluster_mode == 'dataset':
        df['cluster_id'] = df['dataset'].map(lambda s: _norm_missing(s) or '(missing)')
    else:
        df['cluster_id'] = df[cluster_key_cols].apply(
            lambda r: '|'.join(_norm_missing(r[c]) or '(missing)'
                               for c in cluster_key_cols), axis=1)

    # ---- pair identity: dataset + cell (when the metric is per-cell) ----
    cell_col = 'ground_truth_cell' if 'ground_truth_cell' in df.columns else None
    if cell_col is None:
        df['pair_id'] = df['dataset'].map(_norm_missing)
    else:
        df['pair_id'] = (df['dataset'].map(_norm_missing) + '|'
                         + df[cell_col].map(_norm_missing))

    independence = {
        'pairing': 'all methods evaluated on the same cells of the same datasets; '
                   'per-cell metrics paired (blocked) by cell within dataset',
        'remaining_assumption_of_blocks': 'blocks (cells / datasets) mutually '
                   'independent - i.e. per-cell results of the SAME method treated '
                   'as independent',
        'assumption_violation': 'per-cell results of one method share the dataset '
                   'library/reference set/composition, and datasets sharing a '
                   'patient or cell line (wellDR-seq chips of one patient; one '
                   'cell line across technologies or reference-cell '
                   'configurations) share that material, so same-method results '
                   'are positively correlated within materials',
        'cluster_key': (['material'] if cluster_mode == 'material'
                        else (['dataset'] if cluster_mode == 'dataset' else
                              (list(cluster_key_cols) if cluster_mode == 'custom' else []))),
        'cluster_key_source': cluster_key_source,
        'aggregation': F'per-cluster {cluster_agg} of the per-unit paired differences',
        'inference_level': ('cluster (independent experimental units)'
                            if cluster_mode != 'naive' else
                            'unit (NAIVE - independence assumed; pseudoreplication risk)'),
        'primary_tests': ('two-sided Wilcoxon signed-rank + exact sign test on '
                          'per-material medians; Friedman on the same per-material '
                          'method medians; Holm-Bonferroni on cluster-level P; '
                          'unit-level bootstrap 95% CI of the effect size'
                          if cluster_mode != 'naive' else
                          'two-sided Wilcoxon signed-rank per unit; Friedman per '
                          'unit; BCa 95% CI (all NAIVE)'),
        'naive_columns_kept_for_comparison': ['pvalue_unit_naive',
                                              'rank_biserial_r_unit_naive',
                                              'wilcoxon_W_unit_naive'],
        'diagnostics': 'ICC(1,1) of the paired differences within clusters (one-way '
                       'ANOVA, method of moments); design effect DE = 1 + (m0-1)*ICC; '
                       'n_effective = n_units / DE; p_inflation_ratio = cluster P / naive P',
    }
    if cluster_mode != 'naive':
        cid_counts = df.drop_duplicates(subset=['pair_id'])['cluster_id'].value_counts()
        independence['n_clusters'] = int(len(cid_counts))
        independence['cluster_sizes'] = {
            'min': int(cid_counts.min()), 'median': float(np.median(cid_counts)),
            'max': int(cid_counts.max()), 'total_units': int(cid_counts.sum())}
        if len(cid_counts) < 10:
            logging.warning('only %d independent clusters detected: statistical power '
                            'is limited and conclusions should be phrased cautiously',
                            len(cid_counts))

    metric_list = ([m for m in metrics if m in set(df['metric'].unique())]
                   if metrics is not None else
                   list(dict.fromkeys(df['metric'].tolist())))

    settings = {
        'analysis': 'scRNA-seq CNV-caller benchmark (co-sequencing datasets)',
        'design': ('randomized complete block; blocks = paired cells within '
                   'datasets; inference aggregated to the independent unit '
                   F'({cluster_mode}) before testing'),
        'independence': independence,
        'omnibus_test': 'Friedman per metric at the cluster level '
                        "('cluster' rows) and at the per-unit level "
                        "('unit (naive)' rows)",
        'posthoc_test': ('two-sided Wilcoxon signed-rank + exact sign test on '
                         'per-cluster (material by default) medians of the paired '
                         'per-cell differences '
                         "(zero_method='zsplit', continuity correction)"
                         if cluster_mode != 'naive' else
                         "two-sided Wilcoxon signed-rank paired per unit "
                         "(zero_method='zsplit', continuity correction)"),
        'multiple_comparison_correction': 'Holm-Bonferroni within each (metric) '
                        'family on the cluster-level P values',
        'effect_sizes': ['matched-pairs rank-biserial r (cluster/material level) with '
                         'a 95% percentile-bootstrap CI over the independent units',
                         'paired common-language effect size (unit population)',
                         'median difference with bootstrap 95% CI'],
        'tail': 'two-sided (two-tailed) for all tests; the SIGN of r carries the '
                'direction (positive = the reference method performs better)',
        'reference_method': reference,
        'all_pairs': bool(all_pairs),
        'metrics': list(metric_list),
        'n_methods': int(len(methods_present)),
        'n_datasets': int(df['dataset'].nunique()),
        'bootstrap_n_resamples': int(n_resamples),
        'cluster_bootstrap_n_resamples': int(min(n_resamples, MAX_CLUSTER_BOOTSTRAP)),
        'bootstrap_seed': int(seed),
        'alpha': float(alpha),
        'n_units_per_method': {m: int(n) for m, n in
                               df.groupby('method')['pair_id'].nunique().items()},
        'software': software_versions(),
    }

    pairwise_rows, friedman_rows = [], []

    def _fr_row(level, mat, n_units, metric):
        complete = mat.dropna(axis=0, how='any')
        cols = list(mat.columns)
        row = {'level': level, 'metric': metric,
               'n_blocks': int(len(complete)),
               'n_units_aggregated': int(n_units),
               'k_methods': int(len(cols))}
        fr = friedman_test([complete[c].to_numpy() for c in cols]) if len(complete) else None
        if fr is not None:
            row.update({'friedman_chi2': fr['chi2'], 'df': fr['df'],
                        'pvalue': fr['pvalue'], 'kendalls_w': fr['kendalls_w'],
                        **{F'mean_rank_{c}': rk for c, rk in zip(cols, fr['mean_ranks'])}})
        else:
            row['note'] = 'omnibus undefined (needs >= 3 methods and >= 2 complete blocks)'
        return row

    for metric in metric_list:
        msub = df[df['metric'] == metric]
        # ---- pivot: rows = paired units, columns = methods ----
        series = msub.groupby(['pair_id', 'cluster_id', 'method'])['value'].median()
        piv = series.unstack('method')
        piv = piv.reindex(columns=methods_present)
        piv = piv.loc[:, piv.notna().any(axis=0)]
        ok_methods = [m for m in piv.columns if piv[m].notna().any()]
        if len(ok_methods) < 2:
            continue

        # ---- omnibus Friedman: cluster level (primary) + unit level (naive) ----
        if cluster_mode != 'naive':
            cmat = piv.groupby(level=1, sort=True).agg(cluster_agg)
            cmat = cmat.dropna(axis=0, how='any').loc[:, ok_methods]
            friedman_rows.append(_fr_row('cluster', cmat,
                                         int(piv.notna().any(axis=1).sum()), metric))
        friedman_rows.append(_fr_row('unit (naive)', piv.loc[:, ok_methods],
                                     int(piv.notna().any(axis=1).sum()), metric))

        # ---- pairwise post-hoc ----
        others = [m for m in ok_methods if m != reference]
        pairs = ([(a, b) for a in ok_methods for b in ok_methods if a < b]
                 if all_pairs else [(reference, b) for b in others])
        records = []
        for a, b in pairs:
            both = piv[[a, b]].dropna(axis=0, how='any')
            if cluster_mode == 'naive':
                labels = None
            else:
                labels = both.index.get_level_values('cluster_id').tolist()
            records.append(_pairwise_record(
                both[a].to_numpy(dtype=float), both[b].to_numpy(dtype=float),
                labels, {'metric': metric, 'method_a': a, 'method_b': b},
                cluster_agg=cluster_agg, n_resamples=n_resamples, seed=seed,
                cluster_key_str=('|'.join(independence['cluster_key'])
                                 if cluster_mode != 'naive' else '')))
        # Holm family = all comparisons within (metric)
        if records:
            adj = holm_bonferroni([r['pvalue_two_sided'] for r in records])
            for r, p_adj in zip(records, adj):
                r['pvalue_holm'] = float(p_adj)
                r['reject_holm'] = bool(p_adj <= alpha)
            pairwise_rows.extend(records)

    for r in pairwise_rows:
        r.setdefault('pvalue_holm', float('nan'))
        r.setdefault('reject_holm', '')

    _write_tables(out_prefix, pairwise_rows, friedman_rows, settings)
    _write_latex_table(F'{out_prefix}.pairwise.tsv',
                       F'{out_prefix}.pairwise.tex',
                       reference=reference, alpha=alpha,
                       caption_note=None)
    return settings


def _write_tables(out_prefix, pairwise_rows, friedman_rows, settings):
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(pairwise_rows).sort_values(['metric', 'method_a', 'method_b']).to_csv(
        F'{out_prefix}.pairwise.tsv', sep='\t', index=False, na_rep='NA')
    pd.DataFrame(friedman_rows).to_csv(
        F'{out_prefix}.friedman.tsv', sep='\t', index=False, na_rep='NA')
    with open(F'{out_prefix}.json', 'w') as fh:
        json.dump(settings, fh, indent=2)
    logging.info('stat_tests: wrote %s.pairwise.tsv, %s.friedman.tsv, '
                 '%s.json, %s.pairwise.tex',
                 out_prefix, out_prefix, out_prefix, out_prefix)


# --------------------------------------------------------------------------- #
# LaTeX export of the pairwise statistical table                              #
# --------------------------------------------------------------------------- #
def _method_display_tex(method):
    """One-line LaTeX label of a raw method value, e.g. 'copykat_autoInferRef'
    -> 'CopyKAT 2021 (autoInferRef)'."""
    name = str(method)
    tool, _, suffix = name.partition('_')
    pretty, year, _date = METHOD_PUBLICATION.get(
        tool, (tool, None, '9999-99-99'))
    label = F'{pretty} {year}' if year else pretty
    if not suffix:
        return label
    suffix = ', '.join(METHOD_SUFFIX_DISPLAY_NAMES.get(part, part)
                       for part in suffix.split('_'))
    return F'{label} ({suffix})'


def _method_sort_key(method):
    """Chronological order (exact publication date); unknown callers last."""
    tool = str(method).partition('_')[0]
    _name, _year, date = METHOD_PUBLICATION.get(tool, (tool, None, '9999-99-99'))
    return (date, str(method))


def _tex_escape(s):
    """Escape LaTeX special characters in a table text cell."""
    return (str(s)
            .replace('\\', r'\textbackslash{}')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('_', r'\_')
            .replace('#', r'\#')
            .replace('$', r'\$')
            .replace('^', r'\^{}'))


def _fmt_pvalue_holm(x, alpha=0.05):
    """Format one Holm-adjusted P value for a LaTeX table cell.

    Missing values become '--'; underflow (P = 0) becomes '<2.2e-16'; values
    >= 1e-3 use three decimals, smaller ones scientific notation.  Values
    significant at level `alpha` are wrapped in \\textbf.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    if x != x or x in (float('inf'), float('-inf')):  # NaN / inf
        return '--'
    if x == 0.0:
        s = '$<2.2\\times10^{-16}$'
    elif x < 0.001:
        mant, exp = F'{x:.2e}'.split('e')
        s = F'${mant}\\times10^{{{int(exp)}}}$'
    else:
        s = F'{x:.3f}'
    return F'\\textbf{{{s}}}' if x < alpha else s


def _fmt_effect(x, decimals=2):
    """Format a rank-biserial effect-size cell; missing values become '--'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    if x != x or x in (float('inf'), float('-inf')):
        return '--'
    return F'{x:.{decimals}f}'


def _fmt_ci(low, high, decimals=2):
    """Format a 95% confidence-interval cell as '[low, high]'; missing -> '--'."""
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError):
        return '--'
    if lo != lo or hi != hi or lo in (float('inf'), float('-inf')) \
            or hi in (float('inf'), float('-inf')):
        return '--'
    return F'[{lo:.{decimals}f}, {hi:.{decimals}f}]'


def _fmt_signed_int(x):
    """Format an effective-sample-size cell; missing values become '--'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    if x != x or x in (float('inf'), float('-inf')):
        return '--'
    return F'{int(round(x)):d}'


def _legend(ref_desc, show_ref_col=False, unit_desc='materials'):
    """Table legend: describes exactly the four statistics (n, p, r, CI)."""
    ref_clause = (F'({ref_desc})'
                  if not show_ref_col else
                  F'($a$; see the Method $a$ column, {ref_desc} by default)')
    return (
        'Pairwise comparison of scRNA-seq CNV-caller performance between the '
        F'reference method {ref_clause} and each other method ($b$), per benchmark '
        'metric. Each row reports, in this order: $n$, the effective sample size '
        F'(number of independent {unit_desc} - patients or cell lines whose paired '
        'per-cell differences enter the test); $p$, the two-sided Wilcoxon '
        F'signed-rank test on per-{unit_desc[:-1]} medians of the paired per-cell '
        'differences (reference vs.\\ method $b$), Holm--Bonferroni-adjusted within '
        'each metric family, bold at the 0.05 family-wise level; $r$, the '
        'matched-pairs rank-biserial correlation (positive means the reference '
        'outperforms method $b$); and the 95\\% percentile-bootstrap CI of $r$ '
        F'obtained by resampling the {unit_desc[:-1]}. '
        'CN, copy number.'
    )


def latex_table_lines(tsv_path, reference='infercnv', table_label='tab:scrna-pairwise',
                      alpha=0.05, caption_note=None):
    """Build the booktabs LaTeX table from a *.pairwise.tsv file.

    Every row (one metric x method_b comparison) carries, after the identity
    columns, exactly these four statistics in this order and nothing else:
    n (effective sample size), p (Holm-adjusted two-sided Wilcoxon P),
    r (matched-pairs rank-biserial effect size) and the 95% bootstrap CI of r.
    With --all-pairs a Method $a$ column is added to identify the pairs.
    Returns the list of table lines, or None on any problem (a reason is
    logged).
    """
    if not os.path.isfile(tsv_path):
        logging.error('pairwise stats file not found: %s', tsv_path)
        logging.error('run the statistical tests first (plot_cnv_heatmaps.py '
                      'runs them by default, or use this script\'s -i/-o)')
        return None
    tab = pd.read_csv(tsv_path, sep='\t')
    # n column: prefer the cluster count, fall back to n_pairs (naive mode)
    n_col = None
    for c in ('n_clusters', 'n_pairs'):
        if c in tab.columns and pd.to_numeric(tab[c], errors='coerce').notna().any():
            n_col = c
            break
    value_cols = ('pvalue_holm', 'rank_biserial_r', 'ci95_r_low', 'ci95_r_high')
    missing = [c for c in (('metric', 'method_a', 'method_b') + value_cols)
               if c not in tab.columns]
    if missing:
        logging.error('column(s) %s missing from %s (available: %s)',
                      missing, tsv_path, ', '.join(map(str, tab.columns)))
        return None
    if n_col is None:
        tab['n_effective'] = float('nan')
        n_col = 'n_effective'
    sub = tab.dropna(subset=['metric', 'method_b']).copy()
    if sub.empty:
        logging.error('no pairwise rows in %s', tsv_path)
        return None
    sub[n_col] = pd.to_numeric(sub[n_col], errors='coerce')

    # detect the inference level for the caption's unit description
    if 'inference_level' in sub.columns and len(sub) \
            and str(sub['inference_level'].iloc[0]).startswith('unit (naive)'):
        unit_desc = 'paired units'
    else:
        unit_desc = 'materials'

    ref_callers = list(dict.fromkeys(sub['method_a'].tolist())) \
        if 'method_a' in sub.columns else []
    show_ref = len(ref_callers) > 1

    # row order: metrics keep their order of first appearance; methods are
    # ordered chronologically (exact publication date) like the figures
    metric_order = list(dict.fromkeys(sub['metric'].tolist()))
    method_order = sorted(set(sub['method_b'].tolist()), key=_method_sort_key)
    sub['metric'] = pd.Categorical(sub['metric'], categories=metric_order, ordered=True)
    sub['method_b'] = pd.Categorical(sub['method_b'], categories=method_order, ordered=True)
    sort_cols = ['metric'] + (['method_a'] if show_ref else []) + ['method_b']
    if show_ref:
        sub['method_a'] = pd.Categorical(sub['method_a'], categories=ref_callers, ordered=True)
    sub = sub.sort_values(sort_cols)

    ref_desc = _method_display_tex(reference)
    legend = _legend(ref_desc, show_ref_col=show_ref, unit_desc=unit_desc)
    if caption_note:
        legend = F'{caption_note} {legend}'

    header_cells = (['Metric', 'Method $a$', 'Method $b$'] if show_ref
                    else ['Metric', 'Method $b$'])
    n_id = 3 if show_ref else 2
    header_cells += ['$n$', '$p$', '$r$', '95\\% CI']
    colspec = 'l' * n_id + 'rrrr'

    lines = [
        '% LaTeX table generated by stat_tests.py '
        '(requires \\usepackage{booktabs})',
        '\\begin{table}[htbp]',
        '  \\centering',
        F'  \\caption{{{legend}}}',
        F'  \\label{{{table_label}}}',
        F'  \\begin{{tabular}}{{{colspec}}}',
        '    \\toprule',
        F'    {" & ".join(header_cells)} \\\\',
        '    \\midrule',
    ]
    for rec in sub.itertuples(index=False):
        cells = [_tex_escape(rec.metric)]
        if show_ref:
            cells.append(_tex_escape(_method_display_tex(rec.method_a)))
        cells.append(_tex_escape(_method_display_tex(rec.method_b)))
        cells.append(_fmt_signed_int(getattr(rec, n_col)))
        cells.append(_fmt_pvalue_holm(rec.pvalue_holm, alpha=alpha))
        cells.append(_fmt_effect(rec.rank_biserial_r))
        cells.append(_fmt_ci(rec.ci95_r_low, rec.ci95_r_high))
        lines.append(F'    {" & ".join(cells)} \\\\')
    lines += [
        '    \\bottomrule',
        '  \\end{tabular}',
        '\\end{table}',
    ]
    return lines


def emit_latex_stats_table(tsv_path, reference='infercnv',
                           table_label='tab:scrna-pairwise', alpha=0.05,
                           caption_note=None):
    """Print a copy-paste-ready booktabs LaTeX table from a
    *.pairwise.tsv file. Returns 0 on success, 1 on any problem."""
    lines = latex_table_lines(tsv_path, reference=reference,
                              table_label=table_label, alpha=alpha,
                              caption_note=caption_note)
    if lines is None:
        return 1
    print('\n'.join(lines))
    return 0


def _write_latex_table(tsv_path, tex_path, reference='infercnv', alpha=0.05,
                       caption_note=None):
    """Write the booktabs LaTeX table to `tex_path` (default pipeline output).

    Same table as emit_latex_stats_table, but into a file instead of stdout.
    Returns 0 on success, 1 on any problem (logged; never raises - a failure
    of the table must not break the benchmark run).
    """
    try:
        lines = latex_table_lines(tsv_path, reference=reference,
                                  table_label='tab:scrna-pairwise', alpha=alpha,
                                  caption_note=caption_note)
        if lines is None:
            return 1
        with open(tex_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        logging.info('LaTeX pairwise table written to %s', tex_path)
        return 0
    except Exception as exc:  # pragma: no cover
        logging.warning('could not write the LaTeX table %s: %s', tex_path, exc)
        return 1


# --------------------------------------------------------------------------- #
# Command-line interface                                                       #
# --------------------------------------------------------------------------- #
def _expand_inputs(patterns):
    paths, seen = [], set()
    for pat in patterns:
        matched = sorted(glob.glob(pat, recursive=True)) if any(c in pat for c in '*?[') else [pat]
        for p in matched:
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap):
                seen.add(ap)
                paths.append(p)
    return paths


def _parse_cluster_key(raw):
    """None | 'none' | 'dataset' | 'a,b,c' -> None | [] | ['dataset'] | list."""
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ('none', 'naive', 'off', 'no'):
        return []
    if s in ('dataset', 'per-dataset'):
        return ['dataset']
    if s in ('material', 'default'):
        return None
    return [c.strip() for c in raw.split(',') if c.strip()]


def _load_long_frames(input_patterns, file_pattern=None):
    """Load the benchmark tables named by -i into ONE unified long frame.

    Two input kinds are recognised per pattern, BEFORE any loader runs:

    * a path/glob that expands only to ALREADY-UNIFIED long TSVs carrying the
      (dataset, method, metric, value) columns: taken as-is (the method
      column still gets the _predict -> _autoInferRef normalisation), so a
      pre-aggregated table is analysed exactly as given - the figures'
      method filtering is NOT silently re-applied to it;
    * anything else: the ORIGINAL (unexpanded) glob patterns are fed through
      plot_cnv_heatmaps.py's own loaders (load_all + load_classification +
      filter_benchmark_methods), so the standalone run sees EXACTLY the same
      datasets/methods/metrics as the figures (lazy import; per-pattern guard
      against empty globs). Without plot_cnv_heatmaps.py next to this script,
      a light fallback loader reads each raw evaluation TSV
      (caller/metric/value): the dataset comes from the file path and the
      method from the caller column.
    """
    long_cols = {'dataset', 'method', 'metric', 'value'}

    def _split_pattern(pat):
        matched = (sorted(glob.glob(pat, recursive=True))
                   if any(c in pat for c in '*?[') else [pat])
        matched = [m for m in matched if os.path.isfile(m)]
        if not matched:
            return None, None
        try:
            heads = [pd.read_csv(m, sep='\t', nrows=0) for m in matched]
        except Exception:
            return None, matched
        if all(long_cols.issubset(set(h.columns)) for h in heads):
            return matched, []
        return None, matched

    frames, raw_patterns = [], []
    for pat in input_patterns:
        long_files, raw_files = _split_pattern(pat)
        if long_files is None and raw_files is None:
            print(F'  SKIP {pat}: no files match')
            continue
        if long_files:
            for m in long_files:
                df = pd.read_csv(m, sep='\t')
                df['method'] = (df['method'].astype(str).str.strip()
                                .str.replace(r'_predict', r'_autoInferRef',
                                             regex=False))
                frames.append(df)
            print(F'  {len(long_files)} unified long TSV(s) taken as-is '
                  F'(dataset/method/metric/value columns detected)')
        if raw_files:
            raw_patterns.append(pat)

    if raw_patterns:
        loaded_raw = False
        try:
            from plot_cnv_heatmaps import (load_all, load_classification,
                                           filter_benchmark_methods)
            for pat in raw_patterns:
                if not glob.glob(pat, recursive=True):
                    continue
                unified = load_all(pat, file_pattern)
                clf = load_classification(pat, file_pattern)
                if not clf.empty:
                    unified = pd.concat([unified, clf], ignore_index=True)
                unified = filter_benchmark_methods(unified)
                frames.append(unified)
            loaded_raw = True
        except ImportError:
            pass
        if not loaded_raw:
            # ---- fallback without plot_cnv_heatmaps.py -------------------
            for fp in _expand_inputs(raw_patterns):
                try:
                    df = pd.read_csv(fp, sep='\t')
                except Exception as exc:
                    print(F'  SKIP {fp}: {exc}')
                    continue
                if long_cols.issubset(set(df.columns)):
                    frames.append(df)
                    continue
                if not {'metric', 'value', 'caller'}.issubset(set(df.columns)):
                    print(F'  SKIP {fp}: not an evaluation TSV (need '
                          'caller/metric/value or dataset/method/metric/value '
                          'columns)')
                    continue
                m = re.search(r'(\S+?)_solo-genefull_output', fp)
                dataset = m.group(1) if m else os.path.basename(
                    os.path.dirname(os.path.dirname(fp)))
                df = df.copy()
                df['dataset'] = dataset
                df['method'] = (df['caller'].astype(str).str.strip()
                                .str.replace(r'_predict', r'_autoInferRef',
                                             regex=False))
                frames.append(df)

    if not frames:
        raise SystemExit('stat_tests: no usable input tables')
    out = pd.concat(frames, ignore_index=True)
    keep = [c for c in ('dataset', 'method', 'metric', 'value',
                        'ground_truth_cell') if c in out.columns]
    return out[keep]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Statistical tests for the scRNA-seq CNV-caller benchmark '
                    '(cluster-level Friedman omnibus; two-sided Wilcoxon '
                    'signed-rank + exact sign test post-hoc with Holm '
                    'correction; rank-biserial effect size with 95% bootstrap '
                    'CI; ICC / design-effect diagnostics). The LaTeX table '
                    'carries exactly n, p, r and the 95% CI of r.')
    parser.add_argument('-i', '--input', nargs='*', default=None,
                        help='Input table(s): raw evaluation TSVs (caller/'
                             'metric/value; dataset inferred from the path like '
                             'plot_cnv_heatmaps.py) or one unified long TSV with '
                             'a dataset column. Globs allowed.')
    parser.add_argument('-o', '--output', default='scrna-stats',
                        help='Output prefix; writes <prefix>.pairwise.tsv, '
                             '<prefix>.friedman.tsv, <prefix>.json and '
                             'the booktabs table <prefix>.pairwise.tex.')
    parser.add_argument('--reference', default='infercnv',
                        help="Reference method (scenario A), default 'infercnv'. "
                             'The comparisons answer: does the reference '
                             'outperform method b?')
    parser.add_argument('--all-pairs', action='store_true',
                        help='Test all method pairs, not only reference vs rest.')
    parser.add_argument('--cluster-key', default=None,
                        help='Columns defining the independent experimental unit '
                             '(cluster). Default: material (the patient / cell '
                             'line derived from the dataset name; datasets '
                             'sharing it are one unit). "dataset" analyses at the '
                             'per-dataset level; "none" disables clustering '
                             '(naive per-unit tests, discouraged).')
    parser.add_argument('--cluster-agg', choices=['median', 'mean'], default='median',
                        help='Aggregation of the per-unit differences within each '
                             'cluster (default median).')
    parser.add_argument('--metrics', default=None,
                        help='Comma-separated metric names to analyse (default: '
                             'all metrics found in the input).')
    parser.add_argument('--boot', type=int, default=10000,
                        help='Bootstrap resamples for the CIs (default 10000; the '
                             'cluster bootstrap is capped at 5000).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed of the bootstrap RNG (default 1).')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Family-wise significance level for Holm rejection (default 0.05).')
    parser.add_argument('--no-latex-table', dest='latex_table_auto',
                        action='store_false', default=True,
                        help='Do not write <prefix>.pairwise.tex.')
    parser.add_argument('--latex-table', action='store_true', default=False,
                        help='Print the booktabs LaTeX table built from the '
                             'existing <prefix>.pairwise.tsv and exit.')
    args = parser.parse_args(argv)

    if not _HAVE_SCIPY:
        raise SystemExit('stat_tests: scipy is required (pip install scipy)')

    if args.latex_table:
        tsv = F'{args.output}.pairwise.tsv'
        return emit_latex_stats_table(tsv, reference=args.reference,
                                      alpha=args.alpha)

    if not args.input:
        parser.error('no input given: pass -i <files/globs> (or pipe a unified '
                     'long TSV through plot_cnv_heatmaps.py)')
    df = _load_long_frames(args.input, file_pattern=None)
    print(F'loaded {len(df)} rows, {df["dataset"].nunique()} datasets, '
          F'{df["method"].nunique()} methods, {df["metric"].nunique()} metrics')

    metrics = ([m.strip() for m in args.metrics.split(',')]
               if args.metrics else None)
    settings = run_scrna_benchmark_stats(
        df, args.output, reference=args.reference, all_pairs=args.all_pairs,
        cluster_key_cols=_parse_cluster_key(args.cluster_key),
        cluster_agg=args.cluster_agg, n_resamples=args.boot,
        seed=args.seed, alpha=args.alpha, metrics=metrics)
    if settings:
        settings['input'] = F'{len(args.input)} pattern(s): ' + ', '.join(args.input)
        with open(F'{args.output}.json', 'w') as fh:
            json.dump(settings, fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
