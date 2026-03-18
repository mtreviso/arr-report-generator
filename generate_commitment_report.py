#!/usr/bin/env python3
"""
Generate Commitment Phase Report for ACL Conferences.

--role sac (default): papers where you are Senior Area Chair
--role ac:            papers where you are Area Chair
--role pc:            all papers (Program Chair / testing mode)

Usage:
  python generate_commitment_report.py \
    --username your@email.com --password yourpassword \
    --venue_id aclweb.org/ACL/2025/Conference --me ~Your_Name1

  # Impersonate a SAC to see their view (requires PC permission):
  python generate_commitment_report.py ... --impersonate ~SAC_Name1

  # Save a cache after a full run:
  python generate_commitment_report.py ... --save-cache

  # Use cached data for fast iteration:
  python generate_commitment_report.py ... --use-cache

  # Combine: impersonate a SAC, cache their data, then iterate quickly:
  python generate_commitment_report.py ... --impersonate ~SAC_Name1 --save-cache --cache-dir .cache_sac1
  python generate_commitment_report.py ... --use-cache --cache-dir .cache_sac1
"""
import os, sys, argparse
from pathlib import Path
from datetime import datetime
import jinja2

from arr_commitment_generator import CommitmentReportGenerator
from args import add_cache_args, add_impersonate_arg, add_append_date_arg, add_comments_level_arg
from dev_cache import save_cache, load_cache, cache_exists
from utils import make_filename


def _build_template_data(gen):
    return {
        "title":                   f"Commitment Phase Report: {gen.venue_id}",
        "generated_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue_id":                gen.venue_id,
        "role":                    gen.role,
        "attention_papers":        gen.attention_papers,
        **gen.attention_template_flags(),
        "papers":                  gen.papers_data,
        "comments_count":          len(gen.comments_data),
        "comments":                gen.comments_data,
        "comments_level":          gen.comments_level,
        "comments_enabled":        gen.comments_level != "none",
        "histogram_data":          gen.generate_histogram_data(),
        "correlation_data":        gen.correlation_data,
        "paper_type_distribution": gen.generate_paper_type_distribution(),
        "contribution_type_distribution": gen.generate_contribution_type_distribution(),
        "score_scatter_data":      gen.generate_score_scatter_data(),
    }


def _render_report(gen, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    if not gen.papers_data:
        return gen._write_error_report(output_dir, filename, "No papers in cache",
            "Cache loaded but papers_data is empty. Re-run with --save-cache.")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(gen._resolve_template_dir())),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    html = env.get_template("commitment_report.html").render(**_build_template_data(gen))
    output_path = Path(output_dir) / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Commitment Phase Report for ACL Conferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--username",   default=os.environ.get("OPENREVIEW_USERNAME", ""))
    parser.add_argument("--password",   default=os.environ.get("OPENREVIEW_PASSWORD", ""))
    parser.add_argument("--venue_id",   default="aclweb.org/ACL/2025/Conference")
    parser.add_argument("--me",         default=os.environ.get("OPENREVIEW_ID", ""),
                        help="Your OpenReview tilde ID, e.g. ~Your_Name1")
    parser.add_argument("--role",       default="sac", choices=["sac", "ac", "pc"],
                        help="sac=Senior AC (default), ac=Area Chair, pc=all papers (PC/test mode)")
    parser.add_argument("--output_dir", default="./reports")
    add_cache_args(parser)
    add_impersonate_arg(parser)
    add_comments_level_arg(parser, default="basic")
    add_append_date_arg(parser)
    args = parser.parse_args()

    if not args.use_cache and (not args.username or not args.password or not args.me):
        parser.print_help()
        sys.exit(1)

    if args.use_cache and args.save_cache:
        print("Error: --use-cache and --save-cache are mutually exclusive.")
        sys.exit(1)
    if args.use_cache and not cache_exists(args.cache_dir):
        print(f"Error: No cache found in '{args.cache_dir}/'. Run once with --save-cache first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating commitment report | venue: {args.venue_id} | role: {args.role} | comments: {args.comments_level} | me: {args.me}")

    try:
        gen = CommitmentReportGenerator(
            username=args.username,
            password=args.password,
            venue_id=args.venue_id,
            me=args.me,
            role=args.role,
            impersonate_group=args.impersonate or None,
            comments_level=args.comments_level,
            skip_api_init=args.use_cache,
        )

        if args.impersonate:
            print(f"[impersonate] Filtering papers for: {args.me}")

        filename = make_filename(args.venue_id, "commitment_report", args.append_date)
        print(f"Output filename: {filename}")

        if args.use_cache:
            print(f"[cache] Loading from '{args.cache_dir}/' (skipping API calls)...")
            load_cache(gen, cache_dir=args.cache_dir)
            path = _render_report(gen, args.output_dir, filename)
        elif args.save_cache:
            print("NOTE: Full fetch -- will save cache afterwards.")
            path = gen.generate_report(output_dir=args.output_dir, filename=filename)
            save_cache(gen, cache_dir=args.cache_dir)
        else:
            path = gen.generate_report(output_dir=args.output_dir, filename=filename)

        print(f"Report generated: {path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
