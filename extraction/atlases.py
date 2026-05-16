"""Atlas definitions for fMRI timeseries extraction.

Each atlas is represented by an AtlasSpec (immutable, picklable).
Workers load atlases lazily via factory functions so imports stay fast.

Usage
-----
    from atlases import DEFAULT_ATLASES, get_schaefer, get_tian_s2

    # Use all default atlases:
    atlas_factories = DEFAULT_ATLASES

    # Or build a custom list:
    atlas_factories = [get_aal, lambda: get_schaefer(n_rois=200)]
"""

import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from nilearn import datasets

# Pull nilearn cache path from central config (falls back gracefully if missing)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
try:
    from config import NILEARN_CACHE
    os.environ.setdefault("NILEARN_DATA", NILEARN_CACHE)
    _ATLAS_CACHE = os.path.join(NILEARN_CACHE, "tian_s2")
except ImportError:
    _ATLAS_CACHE = os.path.join(os.path.dirname(__file__), "..", "atlas_cache")

os.makedirs(_ATLAS_CACHE, exist_ok=True)


# -------------------------------------------------------------------------
# AtlasSpec — fully picklable, safe to pass to multiprocessing workers
# -------------------------------------------------------------------------

@dataclass
class AtlasSpec:
    name: str                       # short label used in logs
    maps: str                       # path to NIfTI atlas file
    index_to_name: Dict[int, str]   # integer voxel value -> region name
    suffix: str                     # output filename suffix, e.g. "_schaefer400_ts.csv"


# -------------------------------------------------------------------------
# Label helper (used by workers)
# -------------------------------------------------------------------------

def map_labels_to_names(masker_labels: list, index_to_name: Dict[int, str]) -> List[str]:
    """Convert NiftiLabelsMasker.labels_ (numeric) to region name strings."""
    return [index_to_name.get(int(lbl), f"region_{lbl}") for lbl in masker_labels]


# -------------------------------------------------------------------------
# Atlas factory functions — each returns one AtlasSpec
# -------------------------------------------------------------------------

def get_aal() -> AtlasSpec:
    """AAL SPM12 — 116 regions (90 cortical + 26 cerebellar)."""
    aal = datasets.fetch_atlas_aal()
    index_to_name = {int(idx): name for idx, name in zip(aal.indices, aal.labels)}
    return AtlasSpec(
        name="aal",
        maps=aal.maps,
        index_to_name=index_to_name,
        suffix="_aal_ts.csv",
    )


def get_harvard_oxford_cortical() -> AtlasSpec:
    """Harvard-Oxford cortical — 48 cortical regions."""
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    index_to_name = {i: name for i, name in enumerate(ho.labels)}
    return AtlasSpec(
        name="ho_cortical",
        maps=ho.maps,
        index_to_name=index_to_name,
        suffix="_ho_cortical_ts.csv",
    )


def get_harvard_oxford_subcortical() -> AtlasSpec:
    """Harvard-Oxford subcortical — 21 subcortical regions."""
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    index_to_name = {i: name for i, name in enumerate(ho.labels)}
    return AtlasSpec(
        name="ho_subcortical",
        maps=ho.maps,
        index_to_name=index_to_name,
        suffix="_ho_subcortical_ts.csv",
    )


def get_schaefer(n_rois: int = 400) -> AtlasSpec:
    """Schaefer 2018 parcellation (default 400 ROIs, 7 networks).

    Voxel values 1..n_rois map to labels[0]..labels[n_rois-1].
    Use n_rois=100 for the coarser parcellation.
    """
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in sch.labels]
    index_to_name = {i + 1: name for i, name in enumerate(labels)}
    return AtlasSpec(
        name=f"schaefer{n_rois}",
        maps=sch.maps,
        index_to_name=index_to_name,
        suffix=f"_schaefer{n_rois}_ts.csv",
    )


_TIAN_BASE    = ("https://raw.githubusercontent.com/yetianmed/subcortex/master/"
                 "Group-Parcellation/3T/Subcortex-Only/")
_TIAN_URL_NII = _TIAN_BASE + "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
_TIAN_URL_TXT = _TIAN_BASE + "Tian_Subcortex_S2_3T_label.txt"


def get_tian_s2() -> Optional[AtlasSpec]:
    """Tian 2020 subcortical parcellation — Scale II, 32 bilateral ROIs.

    Downloaded once and cached at NILEARN_CACHE/tian_s2/.
    Returns None if download fails (so extraction can continue without it).
    """
    nii_path = os.path.join(_ATLAS_CACHE, "Tian_S2_3T_2009cAsym.nii.gz")
    txt_path = os.path.join(_ATLAS_CACHE, "Tian_S2_3T_labels.txt")

    if not os.path.isfile(nii_path):
        try:
            print("  Downloading Tian S2 atlas... ", end="", flush=True)
            urllib.request.urlretrieve(_TIAN_URL_NII, nii_path)
            print("done.", flush=True)
        except Exception as exc:
            print(f"FAILED ({exc}). Skipping Tian S2.", flush=True)
            return None

    if not os.path.isfile(txt_path):
        try:
            urllib.request.urlretrieve(_TIAN_URL_TXT, txt_path)
        except Exception:
            pass

    labels: List[str] = []
    if os.path.isfile(txt_path):
        with open(txt_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    labels.append(line.split("\t")[0] if "\t" in line else line)

    # Tian S2 NIfTI has voxel values 1..32
    index_to_name = {i + 1: name for i, name in enumerate(labels)} if labels \
                    else {i: str(i) for i in range(1, 33)}

    return AtlasSpec(
        name="tian_s2",
        maps=nii_path,
        index_to_name=index_to_name,
        suffix="_tian_s2_ts.csv",
    )


# -------------------------------------------------------------------------
# Default atlas list — passed as factory callables so loading stays lazy
# -------------------------------------------------------------------------
# Schaefer-400 (cortical, 7 networks) + Tian S2 (subcortical, 32 ROIs)
# matches the atlas combination used in QA benchmarks and connectivity analysis.

DEFAULT_ATLASES: List[Callable[[], AtlasSpec]] = [
    get_schaefer,      # Schaefer-400, 7 networks (default n_rois=400)
    get_tian_s2,       # Tian subcortical S2, 32 bilateral ROIs
]
