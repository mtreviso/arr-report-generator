#!/usr/bin/env python3
"""Generate Program Chair (PC) Dashboard for ARR Venues."""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import jinja2

from arr_pc_commitment_generator import PCCommitmentGenerator
from arr_pc_generator import PCReportGenerator
from args import add_args, prompt_for_missing_credentials, validate_cache_args, resolve_cache_dir
from dev_cache import load_cache, save_cache
from utils import make_filename


def _render_report(gen, output_dir, filename, phase):
    os.makedirs(output_dir, exist_ok=True)
    if not gen.papers_data:
        p = Path(output_dir) / filename
        p.write_text(f"<html><body><h1>No papers in cache</h1><p>{gen.venue_id}</p></body></html>", encoding="utf-8")
        return p
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(gen._resolve_template_dir())),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template_name = "pc_commitment_report.html" if phase == "commitment" else "pc_report.html"
    html = env.get_template(template_name).render(**gen.build_template_data())
    output_path = Path(output_dir) / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate PC Dashboard for ARR/ACL Conferences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(
        parser,
        include_impersonate=False,
        require_venue=True,
        default_comments="full",
        default_role=None,
    )
    args = parser.parse_args()
    args.venue_id = args.venue_id.rstrip("/")
    args.linked_venue_id = args.linked_venue_id.rstrip("/")
    args.cache_dir = resolve_cache_dir(args, "pc_report")

    validate_cache_args(args)
    if not args.use_cache:
        prompt_for_missing_credentials(args)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating PC dashboard | venue: {args.venue_id} | phase: {args.phase} | comments: {args.comments_level}")

    try:
        generator_cls = PCCommitmentGenerator if args.phase == "commitment" else PCReportGenerator
        gen = generator_cls(
            username=args.username,
            password=args.password,
            venue_id=args.venue_id,
            me=args.me,
            comments_level=args.comments_level,
            skip_api_init=args.use_cache,
            linked_venue_id=args.linked_venue_id if args.phase == 'commitment' else None,
        )

        report_slug = "pc_commitment_report" if args.phase == "commitment" else "pc_report"
        filename = make_filename(args.venue_id, report_slug, args.append_date)
        print(f"Output filename: {filename}")
        print(f"Cache directory: {args.cache_dir}")

        if args.use_cache:
            print(f"[cache] Loading from '{args.cache_dir}/' (skipping API calls)...")
            load_cache(gen, cache_dir=args.cache_dir)
            path = _render_report(gen, args.output_dir, filename, args.phase)
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
