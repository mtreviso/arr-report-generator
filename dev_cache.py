"""
dev_cache.py — Shared developer cache utilities for fast iteration.

Provides:
  - Pickle cache (save/load) for all report generators

The cache stores per-generator:
    submissions.pkl     — raw Note objects (the slowest fetch)
    group_index.pkl     — pre-fetched groups dict
    processed.pkl       — all computed data attributes

Which attributes are saved/restored is controlled by CACHE_ATTRS below.
Any attr that is missing on a given generator is silently skipped.
"""

import os
import pickle
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Attributes persisted by the cache, per generator type.
# Attributes missing on the generator are silently skipped.
# ---------------------------------------------------------------------------

# Common to ARRReportGenerator and all subclasses
_COMMON_ATTRS = [
    "papers_data",
    "ac_meta_data",
    "comments_data",
    "comments_level",
    "reply_details",
    "correlation_data",
    "score_distributions",
    "profile_cache",
    "ac_email_cache",
    "group_index_complete",
    "missing_group_ids",
    "reviewer_load",
    "reviewer_confidence_data",
]

# PCReportGenerator extras
_PC_EXTRA_ATTRS = [
    "sac_meta_data",
    "track_data",
    "attention_papers",
    "reviewer_load",
]

# CommitmentReportGenerator extras
_COMMITMENT_EXTRA_ATTRS = [
    "linked_note_cache",
    "linked_replies_cache",
]

_ALL_PROCESSED_ATTRS = _COMMON_ATTRS + _PC_EXTRA_ATTRS + _COMMITMENT_EXTRA_ATTRS

_CACHE_FILES = {
    "submissions": "submissions.pkl",
    "group_index": "group_index.pkl",
    "processed":   "processed.pkl",
}


# ---------------------------------------------------------------------------
# Public: cache existence check
# ---------------------------------------------------------------------------

def cache_exists(cache_dir: str) -> bool:
    """Return True if all expected cache files are present."""
    p = Path(cache_dir)
    return all((p / fn).exists() for fn in _CACHE_FILES.values())


# ---------------------------------------------------------------------------
# Public: save / load
# ---------------------------------------------------------------------------

def save_cache(generator, cache_dir: str = ".dev_cache") -> None:
    """
    Persist the generator's raw fetch data AND all processed results to disk.
    Call AFTER generate_report() (or process_data()) has completed.
    Unknown/missing attributes on the generator are silently skipped.
    """
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)

    _dump(generator.submissions, p / _CACHE_FILES["submissions"])
    _dump(generator.group_index, p / _CACHE_FILES["group_index"])

    processed = {
        attr: getattr(generator, attr)
        for attr in _ALL_PROCESSED_ATTRS
        if hasattr(generator, attr)
    }
    _dump(processed, p / _CACHE_FILES["processed"])

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[cache] Saved to '{cache_dir}/' at {ts}")
    print(f"        submissions : {len(generator.submissions)} notes")
    print(f"        group_index : {len(generator.group_index)} groups")
    print(f"        papers_data : {len(generator.papers_data)} papers")


def load_cache(generator, cache_dir: str = ".dev_cache") -> None:
    """
    Restore the generator's state from a previously saved cache,
    bypassing all OpenReview API calls.
    Call right after constructing the generator, before process_data().
    """
    p = Path(cache_dir)
    if not cache_exists(cache_dir):
        missing = [fn for fn in _CACHE_FILES.values() if not (p / fn).exists()]
        raise FileNotFoundError(
            f"Cache incomplete in '{cache_dir}/'. Missing: {missing}\n"
            "Run once with --save-cache to build the cache."
        )

    generator.submissions = _load(p / _CACHE_FILES["submissions"])
    generator.group_index = _load(p / _CACHE_FILES["group_index"])

    processed = _load(p / _CACHE_FILES["processed"])
    for attr, value in processed.items():
        if hasattr(generator, attr):
            setattr(generator, attr, value)

    print(f"[cache] Loaded from '{cache_dir}/'")
    print(f"        submissions : {len(generator.submissions)} notes")
    print(f"        group_index : {len(generator.group_index)} groups")
    print(f"        papers_data : {len(generator.papers_data)} papers")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dump(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"[cache]   wrote {path.name} ({size_mb:.1f} MB)")


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
