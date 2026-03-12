#!/usr/bin/env python3
"""
Generate ARR Review Phase Report.

--role sac (default): only papers in your SAC batch
--role pc:            all papers in the venue (Program Chair / testing mode)

Usage:
  python generate_review_report.py \
    --username your@email.com --password yourpassword \
    --venue_id aclweb.org/ACL/ARR/2025/February --me ~Your_Name1

  # Impersonate the PC group to see a specific SAC's papers:
  # --impersonate = group whose token to assume (must have permission)
  # --me          = the SAC whose paper batch you want to view
  python generate_review_report.py ... \
    --impersonate aclweb.org/ACL/ARR/2026/January/Program_Chairs \
    --me ~Target_SAC_Name1

  # Save a cache after a full run:
  python generate_review_report.py ... --save-cache

  # Use cached data for fast iteration:
  python generate_review_report.py ... --use-cache

  # Combine: impersonate PC, view a SAC's batch, cache for fast iteration:
  python generate_review_report.py ... \
    --impersonate aclweb.org/ACL/ARR/2026/January/Program_Chairs \
    --me ~Target_SAC_Name1 --save-cache --cache-dir .cache_sac1
  python generate_review_report.py ... --use-cache --cache-dir .cache_sac1
"""
import os, sys, argparse, getpass
from pathlib import Path
from datetime import datetime
import jinja2

from arr_report_generator import ARRReportGenerator
from args import add_cache_args, add_impersonate_arg, add_append_date_arg, add_comments_level_arg
from dev_cache import save_cache, load_cache, cache_exists
from utils import make_filename


def _build_template_data(gen):
    return {
        "title":                   f"ARR Review Report: {gen.venue_id}",
        "generated_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue_id":                gen.venue_id,
        "papers":                  gen.papers_data,
        "ac_meta":                 gen.ac_meta_data,
        "attention_papers":        gen.attention_papers,
        **gen.attention_template_flags(),
        "comments_count":          len(gen.comments_data),
        "comments":                gen.comments_data,
        "comments_level":          gen.comments_level,
        "comments_enabled":        gen.comments_level != "none",
        "histogram_data":          gen.generate_histogram_data(),
        "correlation_data":        gen.correlation_data,
        "paper_type_distribution": gen.generate_paper_type_distribution(),
        "review_completion_data":  gen.generate_review_completion_data(),
        "score_scatter_data":      gen.generate_score_scatter_data(),
        "ac_scoring_data":         gen.generate_ac_scoring_data(),
    }


def _render_report(gen, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(gen._resolve_template_dir())),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    html = env.get_template("review_report.html").render(**_build_template_data(gen))
    output_path = Path(output_dir) / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ARR Review Phase Report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--username",   default=os.environ.get("OPENREVIEW_USERNAME", ""))
    parser.add_argument("--password",   default=os.environ.get("OPENREVIEW_PASSWORD", ""))
    parser.add_argument("--venue_id",   default="aclweb.org/ACL/ARR/2025/February")
    parser.add_argument("--me",         default=os.environ.get("OPENREVIEW_ID", ""),
                        help="Your OpenReview tilde ID, e.g. ~Your_Name1")
    parser.add_argument("--role",       default="sac", choices=["sac", "pc"],
                        help="sac=Senior AC (default), pc=all papers (PC/test mode)")
    parser.add_argument("--output_dir", default="./reports")
    add_cache_args(parser)
    add_impersonate_arg(parser)
    add_comments_level_arg(parser, default="basic")
    add_append_date_arg(parser)
    args = parser.parse_args()

    if not args.username:
        args.username = input("OpenReview username: ")
    if not args.password:
        args.password = getpass.getpass("OpenReview password: ")
    if not args.me:
        args.me = input("Your OpenReview tilde ID (e.g. ~Your_Name1): ")

    if args.use_cache and args.save_cache:
        print("Error: --use-cache and --save-cache are mutually exclusive.")
        sys.exit(1)
    if args.use_cache and not cache_exists(args.cache_dir):
        print(f"Error: No cache found in '{args.cache_dir}/'. Run once with --save-cache first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating review report | venue: {args.venue_id} | role: {args.role}")

    try:
        gen = ARRReportGenerator(
            username=args.username,
            password=args.password,
            venue_id=args.venue_id,
            me=args.me,
            role=args.role,
            impersonate_group=args.impersonate or None,
            comments_level=args.comments_level,
        )

        if args.impersonate:
            print(f"[impersonate] Filtering papers for SAC: {args.me}")

        filename = make_filename(args.venue_id, "review_report", args.append_date)
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
