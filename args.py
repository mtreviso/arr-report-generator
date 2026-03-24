"""Shared argparse helpers for report generation scripts."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from dev_cache import cache_exists


def add_args(
    parser,
    *,
    include_impersonate: bool = False,
    require_venue: bool = True,
    default_comments: str = "full",
    default_role: str | None = None,
) -> None:
    """Add the common CLI arguments used by report-generation scripts."""
    parser.add_argument("--username", default=os.environ.get("OPENREVIEW_USERNAME", ""))
    parser.add_argument("--password", default=os.environ.get("OPENREVIEW_PASSWORD", ""))

    parser.add_argument(
        "--phase",
        default="review",
        choices=["review", "commitment"],
        help="review=ARR review phase, commitment=ACL commitment phase",
    )
    parser.add_argument(
        "--linked-venue-id",
        default="",
        help=(
            "ARR venue ID that commitment papers link back to, "
            "e.g. aclweb.org/ACL/ARR/2026/February. "
            "When provided for --phase commitment, all linked ARR reviews and notes are "
            "bulk-fetched upfront instead of one per paper — significantly faster. "
            "If omitted, the venue is auto-detected from the first paper link."
        ),
    )

    parser.add_argument(
        "--venue-id",
        required=require_venue,
        help="OpenReview venue ID, e.g. aclweb.org/ACL/ARR/2025/May",
    )
    parser.add_argument(
        "--me",
        default=os.environ.get("OPENREVIEW_ID", ""),
        help="Your OpenReview tilde ID, e.g. ~Your_Name1",
    )

    if default_role is not None:
        parser.add_argument("--role", default=default_role, help="Role filter.")

    parser.add_argument("--output-dir", default="./reports")

    g = parser.add_argument_group("developer cache (skip slow API calls)")
    g.add_argument(
        "--save-cache",
        action="store_true",
        help="Fetch from OpenReview AND save a pickle cache for future --use-cache runs.",
    )
    g.add_argument(
        "--use-cache",
        action="store_true",
        help="Load data from a previous --save-cache run; skips all OpenReview API calls.",
    )
    g.add_argument(
        "--cache-dir",
        default="",
        help="Directory for pickle cache files. Defaults to an auto-generated path under .dev_cache/.",
    )

    if include_impersonate:
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

    parser.add_argument(
        "--comments-level",
        choices=["none", "basic", "full"],
        default=default_comments,
        help=(
            "Comment detail level. "
            "none = omit comments tab and skip comment processing; "
            "basic = direct replies only; "
            "full = full reply threads (slowest / largest output)."
        ),
    )
    parser.add_argument(
        "--append-date",
        action="store_true",
        help="Append today's date (YYYY-MM-DD) to the output filename to avoid overwriting previous reports.",
    )


def prompt_for_missing_credentials(args) -> None:
    if not args.username:
        args.username = input("OpenReview username: ")
    if not args.password:
        args.password = getpass.getpass("OpenReview password: ")
    if not args.me:
        args.me = input("Your OpenReview tilde ID (e.g. ~Your_Name1): ")


def validate_cache_args(args) -> None:
    if args.use_cache and args.save_cache:
        print("Error: --use-cache and --save-cache are mutually exclusive.")
        sys.exit(1)
    if args.use_cache and not cache_exists(args.cache_dir):
        print(f"Error: No cache found in '{args.cache_dir}/'. Run once with --save-cache first.")
        sys.exit(1)


def _slugify_cache_part(value: str) -> str:
    value = (value or "").strip().replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in value) or "unknown"


def resolve_cache_dir(args, report_kind: str) -> str:
    """Resolve the effective cache dir.

    If --cache-dir is omitted, build a phase- and report-aware path like:
      .dev_cache/<report_kind>/<phase>/<venue_id>[/role-<role>]
    This avoids collisions between review vs commitment and review vs PC runs.
    """
    if getattr(args, "cache_dir", ""):
        return args.cache_dir

    parts = [".dev_cache", _slugify_cache_part(report_kind)]
    phase = getattr(args, "phase", "")
    if phase:
        parts.append(_slugify_cache_part(phase))
    venue_id = getattr(args, "venue_id", "")
    if venue_id:
        parts.append(_slugify_cache_part(venue_id))
    role = getattr(args, "role", "")
    if role and report_kind != "pc_report":
        parts.append(f"role-{_slugify_cache_part(role)}")
    impersonate = getattr(args, "impersonate", "")
    if impersonate:
        parts.append(f"impersonate-{_slugify_cache_part(impersonate)}")
    return str(Path(*parts))
