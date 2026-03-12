"""Shared argparse helpers for all generate_* scripts."""


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


def add_comments_level_arg(parser, default="basic") -> None:
    parser.add_argument(
        "--comments-level",
        choices=["none", "basic", "full"],
        default=default,
        help=(
            "Comment detail level. "
            "none = omit comments tab and skip comment processing; "
            "basic = direct replies only; "
            "full = full reply threads (slowest / largest output)."
        ),
    )


def add_append_date_arg(parser) -> None:
    """Add --append-date flag to an argparse parser."""
    parser.add_argument(
        "--append-date", action="store_true",
        help="Append today's date (YYYY-MM-DD) to the output filename to avoid overwriting previous reports.",
    )
