"""
dev_cache.py — Shared developer utilities for fast iteration.

Provides:
  - Pickle cache (save/load) for all report generators
  - Impersonation helper (for review & commitment reports)
  - Shared argparse argument groups

Usage in any generate_*.py script:
    from dev_cache import add_cache_args, add_impersonate_arg, \
                          handle_cache_and_impersonate, save_cache, load_cache

The cache stores per-generator:
    submissions.pkl     — raw Note objects (the slowest fetch)
    group_index.pkl     — pre-fetched groups dict
    processed.pkl       — all computed data attributes

Which attributes are saved/restored is controlled by CACHE_ATTRS below.
Any attr that is missing on a given generator is silently skipped.
"""

import os
import pickle
import requests
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
    "correlation_data",
    "score_distributions",
    "profile_cache",
    "ac_email_cache",
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
# Public: impersonation
# ---------------------------------------------------------------------------

def impersonate_user(client, group_id: str) -> None:
    """
    Obtain a token that acts as `group_id` (e.g. the Program_Chairs group).

    The OpenReview /impersonate endpoint accepts a groupId — the ID of a group
    whose token you want to assume.  You must already be authenticated as a
    user who has permission to impersonate that group (typically a superuser or
    a user listed in the group's `impersonators` field).

    Typical usage: pass ``venue_id + "/Program_Chairs"`` to get a PC-level
    token that can read all submissions regardless of your normal role.

    Patches client.token in-place so all subsequent API calls run under
    that group's identity.

    See: https://docs.openreview.net/reference/api-v2/openapi-definition#post-impersonate
    """
    url = f"{client.baseurl}/impersonate"
    headers = {
        "Authorization": f"Bearer {client.token}",
        "Content-Type":  "application/json",
    }

    resp = requests.post(url, json={"groupId": group_id}, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Impersonation failed (HTTP {resp.status_code}): {resp.text}\n"
            f"Make sure you have permission to impersonate '{group_id}'.\n"
            f"Typical value: <venue_id>/Program_Chairs"
        )

    data = resp.json()
    new_token = data.get("token") or data.get("access_token")
    if not new_token:
        raise RuntimeError(
            f"Impersonation response did not include a token. Full response: {data}"
        )

    client.token = new_token
    if hasattr(client, "session") and hasattr(client.session, "headers"):
        client.session.headers["Authorization"] = f"Bearer {new_token}"

    print(f"[impersonate] Now acting as group: {group_id}")


# ---------------------------------------------------------------------------
# Public: shared argparse helpers
# ---------------------------------------------------------------------------

def add_cache_args(parser) -> None:
    """Add --save-cache / --use-cache / --cache-dir to an argparse parser."""
    g = parser.add_argument_group("developer cache (skip slow API calls)")
    g.add_argument(
        "--save-cache", action="store_true",
        help="Fetch from OpenReview AND save a pickle cache for future --use-cache runs.",
    )
    g.add_argument(
        "--use-cache", action="store_true",
        help="Load data from a previous --save-cache run; skips all OpenReview API calls.",
    )
    g.add_argument(
        "--cache-dir", default=".dev_cache",
        help="Directory for pickle cache files.",
    )


def add_impersonate_arg(parser) -> None:
    """Add --impersonate to an argparse parser."""
    parser.add_argument(
        "--impersonate",
        nargs="?",
        const="__DEFAULT_PROGRAM_CHAIRS__",
        default="",
        metavar="GROUP_ID",
        help=(
            "Impersonate an OpenReview group to fetch data under that group's identity. "
            "Use '--impersonate GROUP_ID' to pick a specific group, or just '--impersonate' "
            "to default to '<venue_id>/Program_Chairs'."
        ),
    )


# ---------------------------------------------------------------------------
# Public: filename builder
# ---------------------------------------------------------------------------

def make_filename(venue_id: str, base: str, append_date: bool = False) -> str:
    """
    Build a report filename from the venue ID and an optional date suffix.

    Examples
    --------
    venue_id = "aclweb.org/ACL/ARR/2026/January", base = "review_report"
      → "ARR_2026_January_review_report.html"          (append_date=False)
      → "ARR_2026_January_review_report_2026-03-11.html" (append_date=True)

    The venue slug is built from the last 3 path segments (skipping the
    organisation prefix), joined with underscores.  If the venue_id has
    fewer than 3 segments the whole path is used.
    """
    import re
    from datetime import date

    # Strip trailing slashes, split on "/"
    parts = [p for p in venue_id.strip("/").split("/") if p]
    # Drop the org prefix (e.g. "aclweb.org", "ACL") — keep last 3 at most
    slug_parts = parts[-3:] if len(parts) >= 3 else parts
    slug = "_".join(slug_parts)
    # Sanitise: keep only alphanumerics, dashes, underscores
    slug = re.sub(r"[^A-Za-z0-9_\-]", "_", slug)

    stem = f"{slug}_{base}"
    if append_date:
        stem += f"_{date.today().isoformat()}"
    return stem + ".html"


def add_append_date_arg(parser) -> None:
    """Add --append-date flag to an argparse parser."""
    parser.add_argument(
        "--append-date", action="store_true",
        help="Append today's date (YYYY-MM-DD) to the output filename to avoid overwriting previous reports.",
    )


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
