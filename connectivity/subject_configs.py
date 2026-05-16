"""subject_configs.py
Per-subject configuration dicts consumed by psilo_session_comparison.py.

To add a new subject:
  1. Copy an existing config dict and update the keys.
  2. Add it to the CONFIGS registry at the bottom.
  3. Run:  python psilo_session_comparison.py --subject sub-XXX

Keys
----
subject_label   : string used in figure titles
ts_dir          : directory containing extracted timeseries CSVs
out_dir         : directory where figures are written
session_labels  : {ses-key -> display label}
session_colors  : {ses-key -> hex colour}
conditions      : {condition-name -> {ses-key -> file prefix (no atlas/ext)}}
                  Use None for a session where the condition was not collected.
rest_conditions : list of condition keys shown in fig1 (rest + checktr scans)
music_conditions: list of condition keys shown in fig2 / fig4 / fig7
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from config import TS_OUTPUT_DIR as _TS_OUTPUT_DIR  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# sub-002
# ─────────────────────────────────────────────────────────────────────────────
SUB002 = {
    "subject_label": "sub-002",
    "ts_dir":  os.path.join(_REPO, "outputs", "psilo_sub002_GSR_off_v2"),
    "out_dir": os.path.join(_REPO, "outputs", "psilo_sub002_analysis"),
    "session_labels": {
        "ses-1": "Baseline",
        "ses-2": "Psilocybin",
        "ses-3": "Follow-up",
    },
    "session_colors": {
        "ses-1": "#2196F3",
        "ses-2": "#F44336",
        "ses-3": "#4CAF50",
    },
    "conditions": {
        "rest": {
            "ses-1": "sub-002_ses-1_task-rest_run-1",
            "ses-2": "sub-002_ses-2_task-rest_run-1",
            "ses-3": "sub-002_ses-3_task-rest_run-1",
        },
        "agami": {
            "ses-1": "sub-002_ses-1_task-music_acq-agami_run-3",
            "ses-2": "sub-002_ses-2_task-music_acq-agami_run-3",
            "ses-3": "sub-002_ses-3_task-music_acq-agami_run-3",
        },
        "bailero": {
            "ses-1": "sub-002_ses-1_task-music_acq-baliero_run-2",   # typo in ses-1 filename
            "ses-2": "sub-002_ses-2_task-music_acq-bailero_run-2",
            "ses-3": "sub-002_ses-3_task-music_acq-bailero_run-2",
        },
        "glass": {
            "ses-1": None,   # glass was not collected in sub-002 baseline session
            "ses-2": "sub-002_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-002_ses-3_task-music_acq-glass_run-1",
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["agami", "bailero", "glass"],
}

# ─────────────────────────────────────────────────────────────────────────────
# sub-004
# ses-1: rest, restchecktr2, glass(run-1), tundra(run-2), bailero(run-3), personal(run-4)
# ses-2: rest, restchecktr2, glass(run-2), bailero(run-3), personal(run-4)  [no tundra]
# ses-3: rest, restchecktr2, glass(run-2), bailero(run-3), tundra(run-4), personal(run-5)
# ─────────────────────────────────────────────────────────────────────────────
SUB004 = {
    "subject_label": "sub-004",
    "ts_dir":  os.path.join(_REPO, "outputs", "psilo_sub004_GSR_off"),
    "out_dir": os.path.join(_REPO, "outputs", "psilo_sub004_analysis"),
    "session_labels": {
        "ses-1": "Baseline",
        "ses-2": "Psilocybin",
        "ses-3": "Follow-up",
    },
    "session_colors": {
        "ses-1": "#2196F3",
        "ses-2": "#F44336",
        "ses-3": "#4CAF50",
    },
    "conditions": {
        "rest": {
            "ses-1": "sub-004_ses-1_task-rest_run-1",
            "ses-2": "sub-004_ses-2_task-rest_run-1",
            "ses-3": "sub-004_ses-3_task-rest_run-1",
        },
        "restchecktr2": {
            "ses-1": "sub-004_ses-1_task-restchecktr2_run-1",
            "ses-2": "sub-004_ses-2_task-restchecktr2_run-1",
            "ses-3": "sub-004_ses-3_task-restchecktr2_run-1",
        },
        "glass": {
            "ses-1": "sub-004_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-004_ses-2_task-music_acq-glass_run-2",
            "ses-3": "sub-004_ses-3_task-music_acq-glass_run-2",
        },
        "bailero": {
            "ses-1": "sub-004_ses-1_task-music_acq-bailero_run-3",
            "ses-2": "sub-004_ses-2_task-music_acq-bailero_run-3",
            "ses-3": "sub-004_ses-3_task-music_acq-bailero_run-3",
        },
        "tundra": {
            "ses-1": "sub-004_ses-1_task-music_acq-tundra_run-2",
            "ses-2": None,   # not collected in psilocybin session
            "ses-3": "sub-004_ses-3_task-music_acq-tundra_run-4",
        },
        "personal": {
            "ses-1": "sub-004_ses-1_task-music_acq-personal_run-4",
            "ses-2": "sub-004_ses-2_task-music_acq-personal_run-4",
            "ses-3": "sub-004_ses-3_task-music_acq-personal_run-5",
        },
    },
    "rest_conditions":  ["rest", "restchecktr2"],
    "music_conditions": ["glass", "bailero", "personal", "tundra"],
}

# ─────────────────────────────────────────────────────────────────────────────
# sub-003  (Nimrod)
# ses-1: rest, glass(run-1), bailero(run-2), tundra(run-3), personal1(run-4), personal2(run-5)
# ses-2: rest, glass(run-1), tundra(run-2), personal2(run-3)
# ses-3: rest, glass(run-1), bailero(run-2), tundra(run-3), personal2(run-4)
# ─────────────────────────────────────────────────────────────────────────────
SUB003 = {
    "subject_label": "sub-003",
    "ts_dir":  os.path.join(_REPO, "outputs", "psilo_sub003_GSR_off"),
    "out_dir": os.path.join(_REPO, "outputs", "psilo_sub003_analysis"),
    "session_labels": {
        "ses-1": "Baseline",
        "ses-2": "Psilocybin",
        "ses-3": "Follow-up",
    },
    "session_colors": {
        "ses-1": "#2196F3",
        "ses-2": "#F44336",
        "ses-3": "#4CAF50",
    },
    "conditions": {
        "rest": {
            "ses-1": "sub-003_ses-1_task-rest_run-1",
            "ses-2": "sub-003_ses-2_task-rest_run-1",
            "ses-3": "sub-003_ses-3_task-rest_run-1",
        },
        "glass": {
            "ses-1": "sub-003_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-003_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-003_ses-3_task-music_acq-glass_run-1",
        },
        "bailero": {
            "ses-1": "sub-003_ses-1_task-music_acq-bailero_run-2",
            "ses-2": None,  # not collected in psilocybin session
            "ses-3": "sub-003_ses-3_task-music_acq-bailero_run-2",
        },
        "tundra": {
            "ses-1": "sub-003_ses-1_task-music_acq-tundra_run-3",
            "ses-2": "sub-003_ses-2_task-music_acq-tundra_run-2",
            "ses-3": "sub-003_ses-3_task-music_acq-tundra_run-3",
        },
        "personal1": {
            "ses-1": "sub-003_ses-1_task-music_acq-personal1_run-4",
            "ses-2": None,  # not collected in psilocybin session
            "ses-3": None,  # not collected in follow-up
        },
        "personal2": {
            "ses-1": "sub-003_ses-1_task-music_acq-personal2_run-5",
            "ses-2": "sub-003_ses-2_task-music_acq-personal2_run-3",
            "ses-3": "sub-003_ses-3_task-music_acq-personal2_run-4",
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["glass", "bailero", "tundra", "personal2"],
}

# ─────────────────────────────────────────────────────────────────────────────
# sub-001
# ses-1: rest, glass(run-1), bailero(run-2), tundra(run-3), personal1(run-4), personal2(run-5)
# ses-2: rest, glass(run-1), bailero(run-2), personal1(run-3)   [no tundra/personal2]
# ses-3: rest, glass(run-2), tundra(run-4), bailero(run-5), personal2(run-6), personal1(run-7)
#        [run-1 & task-glass were 149-vol check-TR scans — deleted]
# ─────────────────────────────────────────────────────────────────────────────
SUB001 = {
    "subject_label": "sub-001",
    "ts_dir":  os.path.join(_REPO, "outputs", "psilo_sub001_GSR_off"),
    "out_dir": os.path.join(_REPO, "outputs", "psilo_sub001_analysis"),
    "session_labels": {
        "ses-1": "Baseline",
        "ses-2": "Psilocybin",
        "ses-3": "Follow-up",
    },
    "session_colors": {
        "ses-1": "#2196F3",
        "ses-2": "#F44336",
        "ses-3": "#4CAF50",
    },
    "conditions": {
        "rest": {
            "ses-1": "sub-001_ses-1_task-rest_run-1",
            "ses-2": "sub-001_ses-2_task-rest_run-1",
            "ses-3": "sub-001_ses-3_task-rest_run-1",
        },
        "glass": {
            "ses-1": "sub-001_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-001_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-001_ses-3_task-music_acq-glass_run-2",
        },
        "bailero": {
            "ses-1": "sub-001_ses-1_task-music_acq-bailero_run-2",
            "ses-2": "sub-001_ses-2_task-music_acq-bailero_run-2",
            "ses-3": "sub-001_ses-3_task-music_acq-bailero_run-5",
        },
        "tundra": {
            "ses-1": "sub-001_ses-1_task-music_acq-tundra_run-3",
            "ses-2": None,  # not collected in psilocybin session
            "ses-3": "sub-001_ses-3_task-music_acq-tundra_run-4",
        },
        "personal1": {
            "ses-1": "sub-001_ses-1_task-music_acq-personal1_run-4",
            "ses-2": "sub-001_ses-2_task-music_acq-personal1_run-3",
            "ses-3": "sub-001_ses-3_task-music_acq-personal1_run-7",
        },
        "personal2": {
            "ses-1": "sub-001_ses-1_task-music_acq-personal2_run-5",
            "ses-2": None,  # not collected in psilocybin session
            "ses-3": "sub-001_ses-3_task-music_acq-personal2_run-6",
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["glass", "bailero", "tundra", "personal1", "personal2"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper — generate a GSR-on variant of any config without duplication
# ─────────────────────────────────────────────────────────────────────────────
def make_gsr_config(base: dict) -> dict:
    """Return a copy of *base* with ts_dir and out_dir pointing to GSR-on folders.

    Convention: replaces the trailing folder name
      psilo_subXXX_GSR_off  →  psilo_subXXX_GSR_on
      psilo_subXXX_analysis →  psilo_subXXX_analysis_GSR_on
    """
    import copy
    cfg = copy.deepcopy(base)
    cfg["ts_dir"]  = base["ts_dir"].replace("_GSR_off_v2", "_GSR_on").replace("_GSR_off", "_GSR_on")
    cfg["out_dir"] = base["out_dir"].replace("_analysis", "_analysis_GSR_on")
    # subject_label stays clean — GSR status is read from extraction_config.json
    return cfg


SUB001_GSR_ON = make_gsr_config(SUB001)
SUB002_GSR_ON = make_gsr_config(SUB002)
SUB003_GSR_ON = make_gsr_config(SUB003)
SUB004_GSR_ON = make_gsr_config(SUB004)

# ─────────────────────────────────────────────────────────────────────────────
# synFmap dataset  (all subjects share one output directory)
# ─────────────────────────────────────────────────────────────────────────────
_SYNFMAP_TS  = _TS_OUTPUT_DIR   # from config.py — single source of truth
_SYNFMAP_FIG = os.path.join(_SYNFMAP_TS, "figures")

SUB001_SYNFMAP = {
    "subject_label": "sub-001",
    "ts_dir":  _SYNFMAP_TS,
    "out_dir": os.path.join(_SYNFMAP_FIG, "sub-001"),
    "session_labels": {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"},
    "session_colors": {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"},
    "conditions": {
        "rest": {
            "ses-1": "sub-001_ses-1_task-rest_run-1",
            "ses-2": "sub-001_ses-2_task-rest_run-1",
            "ses-3": "sub-001_ses-3_task-rest_run-1",
        },
        "glass": {
            "ses-1": "sub-001_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-001_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-001_ses-3_task-music_acq-glass_run-2",
        },
        "bailero": {
            "ses-1": "sub-001_ses-1_task-music_acq-bailero_run-2",
            "ses-2": "sub-001_ses-2_task-music_acq-bailero_run-2",
            "ses-3": "sub-001_ses-3_task-music_acq-bailero_run-5",
        },
        "tundra": {
            "ses-1": "sub-001_ses-1_task-music_acq-tundra_run-3",
            "ses-2": None,
            "ses-3": "sub-001_ses-3_task-music_acq-tundra_run-4",
        },
        "personal1": {
            "ses-1": "sub-001_ses-1_task-music_acq-personal1_run-4",
            "ses-2": "sub-001_ses-2_task-music_acq-personal1_run-3",
            "ses-3": "sub-001_ses-3_task-music_acq-personal1_run-7",
        },
        "personal2": {
            "ses-1": "sub-001_ses-1_task-music_acq-personal2_run-5",
            "ses-2": None,
            "ses-3": "sub-001_ses-3_task-music_acq-personal2_run-6",
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["glass", "bailero", "tundra", "personal1", "personal2"],
}

SUB002_SYNFMAP = {
    "subject_label": "sub-002",
    "ts_dir":  _SYNFMAP_TS,
    "out_dir": os.path.join(_SYNFMAP_FIG, "sub-002"),
    "session_labels": {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"},
    "session_colors": {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"},
    "conditions": {
        "rest": {
            "ses-1": "sub-002_ses-1_task-rest_run-1",
            "ses-2": "sub-002_ses-2_task-rest_run-1",
            "ses-3": "sub-002_ses-3_task-rest_run-1",
        },
        "agami": {
            "ses-1": "sub-002_ses-1_task-music_acq-agami_run-3",
            "ses-2": "sub-002_ses-2_task-music_acq-agami_run-3",
            "ses-3": "sub-002_ses-3_task-music_acq-agami_run-3",
        },
        "bailero": {
            "ses-1": "sub-002_ses-1_task-music_acq-baliero_run-2",  # note: typo in raw filename
            "ses-2": "sub-002_ses-2_task-music_acq-bailero_run-2",
            "ses-3": "sub-002_ses-3_task-music_acq-bailero_run-2",
        },
        "glass": {
            "ses-1": None,   # glass not collected in baseline for sub-002
            "ses-2": "sub-002_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-002_ses-3_task-music_acq-glass_run-1",
        },
        "salome": {
            "ses-1": None,
            "ses-2": "sub-002_ses-2_task-music_acq-salome_run-5",
            "ses-3": None,
        },
        "tundra": {
            "ses-1": None,
            "ses-2": "sub-002_ses-2_task-music_acq-tundra_run-4",
            "ses-3": None,
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["agami", "bailero", "glass"],
}

SUB003_SYNFMAP = {
    "subject_label": "sub-003",
    "ts_dir":  _SYNFMAP_TS,
    "out_dir": os.path.join(_SYNFMAP_FIG, "sub-003"),
    "session_labels": {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"},
    "session_colors": {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"},
    "conditions": {
        "rest": {
            "ses-1": "sub-003_ses-1_task-rest_run-1",
            "ses-2": "sub-003_ses-2_task-rest_run-1",
            "ses-3": "sub-003_ses-3_task-rest_run-1",
        },
        "glass": {
            "ses-1": "sub-003_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-003_ses-2_task-music_acq-glass_run-1",
            "ses-3": "sub-003_ses-3_task-music_acq-glass_run-1",
        },
        "bailero": {
            "ses-1": "sub-003_ses-1_task-music_acq-bailero_run-2",
            "ses-2": None,
            "ses-3": "sub-003_ses-3_task-music_acq-bailero_run-2",
        },
        "tundra": {
            "ses-1": "sub-003_ses-1_task-music_acq-tundra_run-3",
            "ses-2": "sub-003_ses-2_task-music_acq-tundra_run-2",
            "ses-3": "sub-003_ses-3_task-music_acq-tundra_run-3",
        },
        "personal1": {
            "ses-1": "sub-003_ses-1_task-music_acq-personal1_run-4",
            "ses-2": None,
            "ses-3": None,
        },
        "personal2": {
            "ses-1": "sub-003_ses-1_task-music_acq-personal2_run-5",
            "ses-2": "sub-003_ses-2_task-music_acq-personal2_run-3",
            "ses-3": "sub-003_ses-3_task-music_acq-personal2_run-4",
        },
    },
    "rest_conditions":  ["rest"],
    "music_conditions": ["glass", "bailero", "tundra", "personal2"],
}

SUB004_SYNFMAP = {
    "subject_label": "sub-004",
    "ts_dir":  _SYNFMAP_TS,
    "out_dir": os.path.join(_SYNFMAP_FIG, "sub-004"),
    "session_labels": {"ses-1": "Baseline", "ses-2": "Psilocybin", "ses-3": "Follow-up"},
    "session_colors": {"ses-1": "#2196F3", "ses-2": "#F44336", "ses-3": "#4CAF50"},
    "conditions": {
        "rest": {
            "ses-1": "sub-004_ses-1_task-rest_run-1",
            "ses-2": "sub-004_ses-2_task-rest_run-1",
            "ses-3": "sub-004_ses-3_task-rest_run-1",
        },
        "restchecktr2": {
            "ses-1": "sub-004_ses-1_task-restchecktr2_run-1",
            "ses-2": None,   # not collected in psilocybin session
            "ses-3": "sub-004_ses-3_task-restchecktr2_run-1",
        },
        "glass": {
            "ses-1": "sub-004_ses-1_task-music_acq-glass_run-1",
            "ses-2": "sub-004_ses-2_task-music_acq-glass_run-2",
            "ses-3": "sub-004_ses-3_task-music_acq-glass_run-2",
        },
        "bailero": {
            "ses-1": "sub-004_ses-1_task-music_acq-bailero_run-3",
            "ses-2": "sub-004_ses-2_task-music_acq-bailero_run-3",
            "ses-3": "sub-004_ses-3_task-music_acq-bailero_run-3",
        },
        "tundra": {
            "ses-1": "sub-004_ses-1_task-music_acq-tundra_run-2",
            "ses-2": None,
            "ses-3": "sub-004_ses-3_task-music_acq-tundra_run-4",
        },
        "personal": {
            "ses-1": "sub-004_ses-1_task-music_acq-personal_run-4",
            "ses-2": "sub-004_ses-2_task-music_acq-personal_run-4",
            "ses-3": "sub-004_ses-3_task-music_acq-personal_run-5",
        },
    },
    "rest_conditions":  ["rest", "restchecktr2"],
    "music_conditions": ["glass", "bailero", "personal", "tundra"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Registry — used by  psilo_session_comparison.py  --subject  argument
#
# Default keys (sub-001 … sub-004) now point to the synFmap pipeline output
# (single shared output dir driven by config.TS_OUTPUT_DIR / TS_CONFIG_DIRS).
# Legacy per-subject GSR-off folders are kept under the -legacy suffix.
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = {
    # Current pipeline output (Schaefer-400 + Tian S2)
    "sub-001": SUB001_SYNFMAP,
    "sub-002": SUB002_SYNFMAP,
    "sub-003": SUB003_SYNFMAP,
    "sub-004": SUB004_SYNFMAP,
    # Legacy per-subject extractions (old paths, kept for reference)
    "sub-001-legacy": SUB001,
    "sub-002-legacy": SUB002,
    "sub-003-legacy": SUB003,
    "sub-004-legacy": SUB004,
    "sub-001-gsr":    SUB001_GSR_ON,
    "sub-002-gsr":    SUB002_GSR_ON,
    "sub-003-gsr":    SUB003_GSR_ON,
    "sub-004-gsr":    SUB004_GSR_ON,
}
