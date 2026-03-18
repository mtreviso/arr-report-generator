#!/usr/bin/env python3
"""
Generate Program Chair (PC) Dashboard for ARR Venues.

Fetches ALL active submissions (no SAC/AC filtering) and produces
an aggregate dashboard with overview stats, SAC-level progress,
track breakdown, attention queue, and full paper table.

Usage:
  python generate_pc_report.py \
    --username your@email.com \
    --password yourpassword \
    --venue_id aclweb.org/ACL/ARR/2025/May \
    --me ~Your_Name1

  # Save a cache after a full run (only need to do this once):
  python generate_pc_report.py ... --save-cache

  # Use the cache for fast iteration (skips ALL API calls):
  python generate_pc_report.py ... --use-cache
"""
import os, sys, argparse, getpass
from pathlib import Path
from datetime import datetime
import jinja2

from arr_pc_generator import PCReportGenerator
from args import add_cache_args, add_append_date_arg, add_comments_level_arg
from dev_cache import save_cache, load_cache, cache_exists
from utils import make_filename


def _build_template_data(gen):
    ac_scoring_data = gen.generate_ac_scoring_data()
    return {
        "title":                   f"PC Dashboard: {gen.venue_id}",
        "generated_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue_id":                gen.venue_id,
        "overview_stats":          gen.compute_overview_stats(),
        "papers":                  gen.papers_data,
        "ac_meta":                 gen.ac_meta_data,
        "sac_meta":                gen.sac_meta_data,
        "track_data":              gen.track_data,
        "attention_papers":        gen.attention_papers,
        **gen.attention_template_flags(),
        "comments_count":          len(gen.comments_data),
        "comments":                gen.comments_data,
        "comments_level":          gen.comments_level,
        "comments_enabled":        gen.comments_level != "none",
        "histogram_data":          gen.generate_histogram_data(),
        "correlation_data":        gen.correlation_data,
        "paper_type_distribution": gen.generate_paper_type_distribution(),
        "contribution_type_distribution": gen.generate_contribution_type_distribution(),
        "review_completion_data":  gen.generate_review_completion_data(),
        "score_scatter_data":      gen.generate_score_scatter_data(),
        "ac_scoring_data":         ac_scoring_data,
        "score_outliers":          gen.compute_score_outliers(),
        "high_disagreement":       gen.compute_high_disagreement_papers(),
        "reviewer_load":           gen.compute_reviewer_load_histogram(),
        "ac_load":                 gen.compute_ac_load_histogram(),
        "sac_load":                gen.compute_sac_load_histogram(),
        "ac_scoring_top":          ac_scoring_data[:15],
        "sac_scoring_top":         gen.compute_sac_scoring_data(),
        "score_by_type_data":      gen.generate_score_by_type_data(),
        "reviewer_load_quality":   gen.generate_reviewer_load_quality_data(),
    }


def _render_report(gen, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    if not gen.papers_data:
        p = Path(output_dir) / filename
        p.write_text(f"<html><body><h1>No papers in cache</h1><p>{gen.venue_id}</p></body></html>")
        return p
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(gen._resolve_template_dir())),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    html = env.get_template("pc_report.html").render(**_build_template_data(gen))
    output_path = Path(output_dir) / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate PC Dashboard for ARR/ACL Conferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--username",   default=os.environ.get("OPENREVIEW_USERNAME", ""))
    parser.add_argument("--password",   default=os.environ.get("OPENREVIEW_PASSWORD", ""))
    parser.add_argument("--venue_id",   required=True,
                        help="OpenReview venue ID, e.g. aclweb.org/ACL/ARR/2025/May")
    parser.add_argument("--me",         default=os.environ.get("OPENREVIEW_ID", ""),
                        help="Your OpenReview tilde ID, e.g. ~Your_Name1")
    parser.add_argument("--output_dir", default="./reports")
    add_cache_args(parser)
    add_comments_level_arg(parser, default="none")
    add_append_date_arg(parser)
    args = parser.parse_args()

    if not args.use_cache:
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
    print(f"Generating PC dashboard | venue: {args.venue_id} | comments: {args.comments_level}")

    try:
        gen = PCReportGenerator(
            username=args.username,
            password=args.password,
            venue_id=args.venue_id,
            me=args.me,
            comments_level=args.comments_level,
            skip_api_init=args.use_cache,
        )

        filename = make_filename(args.venue_id, "pc_report", args.append_date)
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
            print("NOTE: Fetching ALL papers -- may take several minutes for large venues.")
            path = gen.generate_report(output_dir=args.output_dir, filename=filename)

        print(f"\nReport generated: {path}")
        print("Open the HTML file in your browser to view the dashboard.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
