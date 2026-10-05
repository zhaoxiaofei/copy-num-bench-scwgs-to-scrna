#!/usr/bin/env python3
"""Dataset-name tables of the scRNA-seq co-sequencing benchmark.

The benchmark uses two name layers:

* the RAW folder name of a dataset (as it appears under ``results/``, e.g.
  ``scONE-seq_HCT116_HUVEC_H9_as_T_N``), and
* the DISPLAY dataset name used by the figures, tables and statistics (e.g.
  ``scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef``).

``DATASET_NAME_ALIASES`` maps raw -> display.  It is the same table
``plot_cnv_heatmaps.py`` applies when it labels/loads datasets; it lives here
so the statistical modules (``stat_tests.py``, ``winner_analysis.py``,
``mcb.py``) share ONE table instead of re-deriving names from their format.

``MATERIAL_DATASETS`` is the authoritative mapping
``primary sample -> derived dataset names``.  A primary sample is the
independent biological material (patient or cell line) behind the datasets.
One entry lists every dataset derived from the same material: chips/captures
of one patient, one cell line sequenced by different technologies, and the
reference-cell / tumour-normal configuration variants of one cell line.
The statistics aggregate all datasets of a primary sample into ONE effective
sample before any test or effect-size computation, so THIS TABLE -- not a
name-parsing heuristic -- defines the units.

To add a dataset: append its display name (and, if it is loaded through the
raw folder path, add the raw name to ``DATASET_NAME_ALIASES``) under the
primary sample it belongs to.  A dataset that is not listed is reported once
by :func:`material_from_dataset` and then treated as its own material (the
conservative choice: never silently merge unrelated samples).
"""
from __future__ import annotations

import logging
import re

# --------------------------------------------------------------------------- #
# Raw folder name -> display dataset name                                     #
# --------------------------------------------------------------------------- #
DATASET_NAME_ALIASES = {
    'A375_HCA00102_PRJNA603321':
        'DNTR-seq_PRJNA603321_A375_noRefCells',
    'BCIS106T_chip1_SAMN48409192_SRR33511671':
        'wellDR-seq_BCIS106T_chip1_SRR33511671_tumorNormalMix',
    'BCIS106T_chip2_SAMN48409193_SRR33511670':
        'wellDR-seq_BCIS106T_chip2_SRR33511670_tumorNormalMix',
    'BCIS13T_chip1_SAMN40389282_SRR28357490':
        'wellDR-seq_BCIS13T_chip1_SRR28357490_tumorNormalMix',
    'BCIS13T_chip2_SAMN40389283_SRR28357489':
        'wellDR-seq_BCIS13T_chip2_SRR28357489_tumorNormalMix',
    'BCIS28T_chip1_SAMN40389273_SRR28357499':
        'wellDR-seq_BCIS28T_chip1_SRR28357499_tumorNormalMix',
    'BCIS28T_chip2_SAMN40389274_SRR28357498':
        'wellDR-seq_BCIS28T_chip2_SRR28357498_tumorNormalMix',
    'BCIS51T_chip1_SAMN48409194_SRR33511669':
        'wellDR-seq_BCIS51T_chip1_SRR33511669_tumorNormalMix',
    'BCIS51T_chip2_SAMN48409195_SRR33511668':
        'wellDR-seq_BCIS51T_chip2_SRR33511668_tumorNormalMix',
    'BCIS66T_chip1_SAMN48409190_SRR33511673':
        'wellDR-seq_BCIS66T_chip1_SRR33511673_tumorNormalMix',
    'BCIS66T_chip2_SAMN48409191_SRR33511672':
        'wellDR-seq_BCIS66T_chip2_SRR33511672_tumorNormalMix',
    'BCIS70T_chip1_SAMN48409188_SRR33511675':
        'wellDR-seq_BCIS70T_chip1_SRR33511675_tumorNormalMix',
    'BCIS70T_chip2_SAMN48409189_SRR33511674':
        'wellDR-seq_BCIS70T_chip2_SRR33511674_tumorNormalMix',
    'BCIS74T_chip1_SAMN48409186_SRR33511677':
        'wellDR-seq_BCIS74T_chip1_SRR33511677_tumorNormalMix',
    'BCIS74T_chip2_SAMN48409187_SRR33511676':
        'wellDR-seq_BCIS74T_chip2_SRR33511676_tumorNormalMix',
    'Cellline_mixing_experiment_1_1_SAMN48409183_SRR33511680':
        'wellDR-seq_Cellline_mixing_experiment_1_1_SRR33511680_noRefCells',
    'Cellline_mixing_experiment_1_2_SAMN48409184_SRR33511679':
        'wellDR-seq_Cellline_mixing_experiment_1_2_SRR33511679_noRefCells',
    'Cellline_mixing_experiment_2_SAMN48409185_SRR33511678':
        'wellDR-seq_Cellline_mixing_experiment_2_SRR33511678_noRefCells',
    'ECIS25T_chip1_SAMN40389270_SRR28357502':
        'wellDR-seq_ECIS25T_chip1_SRR28357502_tumorNormalMix',
    'ECIS25T_chip2_SAMN40389271_SRR28357501':
        'wellDR-seq_ECIS25T_chip2_SRR28357501_tumorNormalMix',
    'ECIS25T_chip3_SAMN40389272_SRR28357500':
        'wellDR-seq_ECIS25T_chip3_SRR28357500_tumorNormalMix',
    'ECIS36T_chip1_SAMN40389284_SRR28357488':
        'wellDR-seq_ECIS36T_chip1_SRR28357488_tumorNormalMix',
    'ECIS36T_chip2_SAMN40389285_SRR28357487':
        'wellDR-seq_ECIS36T_chip2_SRR28357487_tumorNormalMix',
    'ECIS44T_chip1_SAMN40389277_SRR28357495':
        'wellDR-seq_ECIS44T_chip1_SRR28357495_tumorNormalMix',
    'ECIS44T_chip2_SAMN40389278_SRR28357494':
        'wellDR-seq_ECIS44T_chip2_SRR28357494_tumorNormalMix',
    'ECIS44T_chip3_SAMN40389279_SRR28357493':
        'wellDR-seq_ECIS44T_chip3_SRR28357493_tumorNormalMix',
    'ECIS44T_chip4_SAMN40389280_SRR28357492':
        'wellDR-seq_ECIS44T_chip4_SRR28357492_tumorNormalMix',
    'ECIS44T_chip5_SAMN40389281_SRR28357491':
        'wellDR-seq_ECIS44T_chip5_SRR28357491_tumorNormalMix',
    'ECIS48T_chip1_SAMN40389275_SRR28357497':
        'wellDR-seq_ECIS48T_chip1_SRR28357497_tumorNormalMix',
    'ECIS48T_chip2_SAMN40389276_SRR28357496':
        'wellDR-seq_ECIS48T_chip2_SRR28357496_tumorNormalMix',
    'ECIS57T_chip1_SAMN48409196_SRR33511667':
        'wellDR-seq_ECIS57T_chip1_SRR33511667_tumorNormalMix',
    'ECIS57T_chip2_SAMN48409197_SRR33511666':
        'wellDR-seq_ECIS57T_chip2_SRR33511666_tumorNormalMix',
    'HCT116_PRJNA603321':
        'DNTR-seq_PRJNA603321_HCT116_noRefCells',
    'MDA231_chip1_SAMN40389268_SRR33482482':
        'wellDR-seq_MDA231_chip1_SRR33482482_noRefCells',
    'MDA231_chip2_SAMN40389269_SRR33482483':
        'wellDR-seq_MDA231_chip2_SRR33482483_noRefCells',
    'scONE-seq_HCT116':
        'scONE-seq_PRJNA768428_HCT116_noRefCells',
    'scONE-seq_HCT116_HUVEC_H9_as_T_N':
        'scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef',
    'scONE-seq_NPC43':
        'scONE-seq_PRJNA768428_NPC43_noRefCells',
    'scONE-seq_NPC43_HUVEC_H9_as_T_N':
        'scONE-seq_PRJNA768428_NPC43_asTumor_HUVEC-H9_asRef',
    'wellDR3_SAMN48409182_SRR33511681':
        'wellDR-seq_wellDR3_SRR33511681',
}

# --------------------------------------------------------------------------- #
# Primary sample -> derived dataset names (display spelling)                  #
# --------------------------------------------------------------------------- #
MATERIAL_DATASETS = {
    'A375': (
        'DNTR-seq_PRJNA603321_A375_noRefCells',
    ),
    'BCIS106T': (
        'wellDR-seq_BCIS106T_chip1_SRR33511671_tumorNormalMix',
        'wellDR-seq_BCIS106T_chip2_SRR33511670_tumorNormalMix',
    ),
    'BCIS13T': (
        'wellDR-seq_BCIS13T_chip1_SRR28357490_tumorNormalMix',
        'wellDR-seq_BCIS13T_chip2_SRR28357489_tumorNormalMix',
    ),
    'BCIS28T': (
        'wellDR-seq_BCIS28T_chip1_SRR28357499_tumorNormalMix',
        'wellDR-seq_BCIS28T_chip2_SRR28357498_tumorNormalMix',
    ),
    'BCIS51T': (
        'wellDR-seq_BCIS51T_chip1_SRR33511669_tumorNormalMix',
        'wellDR-seq_BCIS51T_chip2_SRR33511668_tumorNormalMix',
    ),
    'BCIS66T': (
        'wellDR-seq_BCIS66T_chip1_SRR33511673_tumorNormalMix',
        'wellDR-seq_BCIS66T_chip2_SRR33511672_tumorNormalMix',
    ),
    'BCIS70T': (
        'wellDR-seq_BCIS70T_chip1_SRR33511675_tumorNormalMix',
        'wellDR-seq_BCIS70T_chip2_SRR33511674_tumorNormalMix',
    ),
    'BCIS74T': (
        'wellDR-seq_BCIS74T_chip1_SRR33511677_tumorNormalMix',
        'wellDR-seq_BCIS74T_chip2_SRR33511676_tumorNormalMix',
    ),
    'Cellline_mixing_experiment_1_1': (
        'wellDR-seq_Cellline_mixing_experiment_1_1_SRR33511680_noRefCells',
    ),
    'Cellline_mixing_experiment_1_2': (
        'wellDR-seq_Cellline_mixing_experiment_1_2_SRR33511679_noRefCells',
    ),
    'Cellline_mixing_experiment_2': (
        'wellDR-seq_Cellline_mixing_experiment_2_SRR33511678_noRefCells',
    ),
    'ECIS25T': (
        'wellDR-seq_ECIS25T_chip1_SRR28357502_tumorNormalMix',
        'wellDR-seq_ECIS25T_chip2_SRR28357501_tumorNormalMix',
        'wellDR-seq_ECIS25T_chip3_SRR28357500_tumorNormalMix',
    ),
    'ECIS36T': (
        'wellDR-seq_ECIS36T_chip1_SRR28357488_tumorNormalMix',
        'wellDR-seq_ECIS36T_chip2_SRR28357487_tumorNormalMix',
    ),
    'ECIS44T': (
        'wellDR-seq_ECIS44T_chip1_SRR28357495_tumorNormalMix',
        'wellDR-seq_ECIS44T_chip2_SRR28357494_tumorNormalMix',
        'wellDR-seq_ECIS44T_chip3_SRR28357493_tumorNormalMix',
        'wellDR-seq_ECIS44T_chip4_SRR28357492_tumorNormalMix',
        'wellDR-seq_ECIS44T_chip5_SRR28357491_tumorNormalMix',
    ),
    'ECIS48T': (
        'wellDR-seq_ECIS48T_chip1_SRR28357497_tumorNormalMix',
        'wellDR-seq_ECIS48T_chip2_SRR28357496_tumorNormalMix',
    ),
    'ECIS57T': (
        'wellDR-seq_ECIS57T_chip1_SRR33511667_tumorNormalMix',
        'wellDR-seq_ECIS57T_chip2_SRR33511666_tumorNormalMix',
    ),
    'HCT116': (
        'DNTR-seq_PRJNA603321_HCT116_noRefCells',
        'scONE-seq_PRJNA768428_HCT116_asTumor_HUVEC-H9_asRef',
        'scONE-seq_PRJNA768428_HCT116_noRefCells',
    ),
    'HUVEC_H9': (
        'scONE-seq_HUVEC_H9',
    ),
    'MDA231': (
        'wellDR-seq_MDA231_chip1_SRR33482482_noRefCells',
        'wellDR-seq_MDA231_chip2_SRR33482483_noRefCells',
    ),
    'NPC43': (
        'scONE-seq_PRJNA768428_NPC43_asTumor_HUVEC-H9_asRef',
        'scONE-seq_PRJNA768428_NPC43_noRefCells',
    ),
    'wellDR3': (
        'wellDR-seq_wellDR3_SRR33511681',
    ),
}

# Reverse index: display dataset name -> primary sample.  A dataset listed
# under two primary samples would be ambiguous and silently keep the last one,
# so that is an error in the table itself.
DATASET_MATERIAL = {}
for _primary, _datasets in MATERIAL_DATASETS.items():
    for _ds in _datasets:
        if _ds in DATASET_MATERIAL:
            raise ValueError(
                F'dataset_materials: dataset {_ds!r} is listed under both '
                F'{DATASET_MATERIAL[_ds]!r} and {_primary!r}')
        DATASET_MATERIAL[_ds] = _primary

# Display-only footnote markers appended by plot_cnv_heatmaps.py (`$^A$` low
# tumor purity, `$^B$` no normal-cell cluster).  They carry no material
# information and are stripped before the lookup.
_DISPLAY_MARKER_RE = re.compile(r'\$\^[AB]\$$')
_MISSING = {'', 'nan', 'nat', 'none', 'null'}
_WARNED_UNKNOWN = set()


def normalise_dataset(name) -> str:
    """Display dataset name of any accepted spelling (raw folder or display)."""
    s = str(name).strip()
    s = _DISPLAY_MARKER_RE.sub('', s).strip()
    return DATASET_NAME_ALIASES.get(s, s)


def material_from_dataset(dataset) -> str:
    """Primary sample (independent material) of one dataset name.

    The name may be a raw folder name, a display dataset name, either with
    the display footnote markers.  Datasets not listed in MATERIAL_DATASETS
    are reported once and returned unchanged (their own material - the
    conservative choice, never a silent merge).
    """
    s = normalise_dataset(dataset)
    if s.lower() in _MISSING:
        return '(missing)'
    mat = DATASET_MATERIAL.get(s)
    if mat is not None:
        return mat
    if s not in _WARNED_UNKNOWN:
        _WARNED_UNKNOWN.add(s)
        logging.warning('dataset %r is not listed in MATERIAL_DATASETS; treating it '
                        'as its own material.  Add it under its primary sample in '
                        'dataset_materials.py', s)
    return s
