#!/usr/bin/env python3
"""
Generate reviewer-facing reports for ARR and ACL venues.

Phases:
  --phase review      ARR review-phase report (default)
  --phase commitment  ACL commitment-phase report
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import jinja2
import openreview
import requests

from arr_commitment_generator import CommitmentReportGenerator
from arr_report_generator import ARRReportGenerator
from args import add_args, prompt_for_missing_credentials, validate_cache_args, resolve_cache_dir
from dev_cache import load_cache, save_cache
from utils import make_filename


PHASE_CONFIG = {
    "review": {
        "generator_cls": ARRReportGenerator,
        "report_slug": "review_report",
        "template_name": "review_report.html",
        "valid_roles": ["ac", "sac", "pc"],
        "print_label": "review report",
    },
    "commitment": {
        "generator_cls": CommitmentReportGenerator,
        "report_slug": "commitment_report",
        "template_name": "commitment_report.html",
        "valid_roles": ["ac", "pc"],
        "print_label": "commitment report",
    },
}


def _build_review_template_data(gen):
    return {
        "title": f"ARR Review Report: {gen.venue_id}",
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue_id": gen.venue_id,
        "papers": gen.papers_data,
        "ac_meta": gen.ac_meta_data,
        "attention_papers": gen.attention_papers,
        **gen.attention_template_flags(),
        "comments_count": len(gen.comments_data),
        "comments": gen.comments_data,
        "comments_level": gen.comments_level,
        "comments_enabled": gen.comments_level != "none",
        "histogram_data": gen.generate_histogram_data(),
        "correlation_data": gen.correlation_data,
        "paper_type_distribution": gen.generate_paper_type_distribution(),
        "contribution_type_distribution": gen.generate_contribution_type_distribution(),
        "review_completion_data": gen.generate_review_completion_data(),
        "score_scatter_data": gen.generate_score_scatter_data(),
        "ac_scoring_data": gen.generate_ac_scoring_data(),
        "score_by_type_data": gen.generate_score_by_type_data(),
        "reviewer_load_quality": gen.generate_reviewer_load_quality_data(),
    }



def _build_commitment_template_data(gen):
    return {
        "title": f"Commitment Phase Report: {gen.venue_id}",
        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "venue_id": gen.venue_id,
        "role": gen.role,
        "attention_papers": gen.attention_papers,
        **gen.attention_template_flags(),
        "papers": gen.papers_data,
        "comments_count": len(gen.comments_data),
        "comments": gen.comments_data,
        "comments_level": gen.comments_level,
        "comments_enabled": gen.comments_level != "none",
        "histogram_data": gen.generate_histogram_data(),
        "correlation_data": gen.correlation_data,
        "paper_type_distribution": gen.generate_paper_type_distribution(),
        "contribution_type_distribution": gen.generate_contribution_type_distribution(),
        "score_scatter_data": gen.generate_score_scatter_data(),
        "decision_stats": gen._compute_decision_stats(),
    }



def _build_template_data(gen, phase):
    if phase == "commitment":
        return _build_commitment_template_data(gen)
    return _build_review_template_data(gen)



def _render_report(gen, output_dir, filename, phase):
    os.makedirs(output_dir, exist_ok=True)
    if not gen.papers_data:
        if phase == "commitment":
            return gen._write_error_report(
                output_dir,
                filename,
                "No papers in cache",
                "Cache loaded but papers_data is empty. Re-run with --save-cache.",
            )
        p = Path(output_dir) / filename
        p.write_text(f"<html><body><h1>No papers in cache</h1><p>{gen.venue_id}</p></body></html>", encoding="utf-8")
        return p

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(gen._resolve_template_dir())),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template_name = PHASE_CONFIG[phase]["template_name"]
    html = env.get_template(template_name).render(**_build_template_data(gen, phase))
    output_path = Path(output_dir) / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path



def _resolve_phase_config(phase):
    try:
        return PHASE_CONFIG[phase]
    except KeyError:
        raise ValueError(f"Unsupported phase: {phase}")



def _apply_impersonation(client, group_id: str) -> str:
    if not group_id.strip() or group_id == "__DEFAULT_PROGRAM_CHAIRS__":
        raise ValueError("Default impersonation group needs venue context")
    url = f"{client.baseurl}/impersonate"
    headers = {
        "Authorization": f"Bearer {client.token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json={"groupId": group_id}, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Impersonation failed (HTTP {resp.status_code}): {resp.text}\n"
            f"Make sure your account can impersonate '{group_id}'."
        )
    data = resp.json()
    new_token = data.get("token") or data.get("access_token")
    if not new_token:
        raise RuntimeError(f"Impersonation response had no token. Response: {data}")
    client.token = new_token
    if hasattr(client, "session") and hasattr(client.session, "headers"):
        client.session.headers["Authorization"] = f"Bearer {new_token}"
    return group_id



def _get_group_cached(client, group_cache, missing_group_ids, group_id):
    if not group_id:
        return None
    if group_id in group_cache:
        return group_cache[group_id]
    if group_id in missing_group_ids:
        return None
    try:
        group = client.get_group(group_id)
        group_cache[group_id] = group
        return group
    except Exception:
        missing_group_ids.add(group_id)
        return None



def _group_contains_member(client, group_cache, missing_group_ids, group_id, member_id, visited=None):
    if not group_id or not member_id:
        return False
    if visited is None:
        visited = set()
    if group_id in visited:
        return False
    visited.add(group_id)

    group = _get_group_cached(client, group_cache, missing_group_ids, group_id)
    members = getattr(group, "members", None) or []
    for candidate in members:
        if candidate == member_id:
            return True
        if isinstance(candidate, str) and (candidate.startswith("~") or "@" in candidate):
            continue
        if isinstance(candidate, str) and "/" in candidate:
            if _group_contains_member(client, group_cache, missing_group_ids, candidate, member_id, visited):
                return True
    return False



def _resolve_impersonation_group(args):
    if not args.impersonate:
        return ""
    if args.impersonate == "__DEFAULT_PROGRAM_CHAIRS__":
        return f"{args.venue_id.rstrip('/')}/Program_Chairs"
    return args.impersonate



def _discover_actual_roles(args):
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=args.username,
        password=args.password,
    )
    impersonation_group = _resolve_impersonation_group(args)
    if impersonation_group:
        actual_group = _apply_impersonation(client, impersonation_group)
        print(f"[impersonate] Now acting as group: {actual_group}")

    venue_group = client.get_group(args.venue_id)
    submission_name = venue_group.content["submission_name"]["value"]
    prefix = f"{args.venue_id}/{submission_name}"
    group_cache = {}
    missing_group_ids = set()

    groups = client.get_all_groups(prefix=prefix)
    for group in groups:
        group_cache[group.id] = group

    actual_roles = set()
    if _group_contains_member(client, group_cache, missing_group_ids, f"{args.venue_id}/Program_Chairs", args.me):
        actual_roles.add("pc")

    for group_id in group_cache:
        if group_id.endswith("/Senior_Area_Chairs") and _group_contains_member(client, group_cache, missing_group_ids, group_id, args.me):
            actual_roles.add("sac")
        elif group_id.endswith("/Area_Chairs") and _group_contains_member(client, group_cache, missing_group_ids, group_id, args.me):
            actual_roles.add("ac")

    return actual_roles



def _validate_args(args, parser, cfg):
    validate_cache_args(args)
    if args.role not in cfg["valid_roles"]:
        parser.error(
            f"--role {args.role!r} is invalid for --phase {args.phase!r}. "
            f"Choose from: {', '.join(cfg['valid_roles'])}."
        )



def _validate_impersonated_role(args, parser):
    if args.use_cache or not args.impersonate:
        return

    actual_roles = _discover_actual_roles(args)
    if args.role not in actual_roles:
        found = ", ".join(sorted(actual_roles)) if actual_roles else "none detected"
        parser.error(
            f"--impersonate requires --role to be an actual role of {args.me} at {args.venue_id}. "
            f"You passed --role {args.role!r}, but detected roles were: {found}."
        )



def _create_generator(args, cfg):
    generator_cls = cfg["generator_cls"]
    return generator_cls(
        username=args.username,
        password=args.password,
        venue_id=args.venue_id,
        me=args.me,
        role=args.role,
        impersonate_group=args.impersonate or None,
        comments_level=args.comments_level,
        skip_api_init=args.use_cache,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Generate review-facing reports for ARR and ACL venues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args(
        parser,
        include_impersonate=True,
        require_venue=True,
        default_comments="full",
        default_role="sac",
    )
    args = parser.parse_args()
    args.venue_id = args.venue_id.rstrip("/")
    args.linked_venue_id = args.linked_venue_id.rstrip("/")
    args.cache_dir = resolve_cache_dir(args, "review_report")

    cfg = _resolve_phase_config(args.phase)
    _validate_args(args, parser, cfg)

    if not args.use_cache:
        prompt_for_missing_credentials(args)
        _validate_impersonated_role(args, parser)

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"Generating {cfg['print_label']} | venue: {args.venue_id} | phase: {args.phase} | "
        f"role: {args.role} | comments: {args.comments_level} | me: {args.me}"
    )

    try:
        gen = _create_generator(args, cfg)

        if args.impersonate:
            print(f"[impersonate] Filtering papers for: {args.me}")

        filename = make_filename(args.venue_id, cfg["report_slug"], args.append_date)
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
            path = gen.generate_report(output_dir=args.output_dir, filename=filename)

        print(f"\nReport generated: {path}")
        print("Open the HTML file in your browser to view the report.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
