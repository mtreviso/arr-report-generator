"""Shared utility helpers for report generators and CLI scripts."""

from __future__ import annotations

import re
from datetime import date

import requests

DEFAULT_IMPERSONATE_SENTINEL = "__DEFAULT_PROGRAM_CHAIRS__"


def resolve_impersonate_group(group_id: str | None, venue_id: str) -> str | None:
    """Resolve the effective impersonation group from CLI input."""
    if not group_id:
        return None
    if not str(group_id).strip() or group_id == DEFAULT_IMPERSONATE_SENTINEL:
        return venue_id.rstrip('/') + '/Program_Chairs'
    return str(group_id)


def impersonate_user(client, group_id: str, venue_id: str | None = None) -> str:
    """Patch client token to act as ``group_id`` and return the effective group ID."""
    effective_group = resolve_impersonate_group(group_id, venue_id or "") if venue_id else group_id
    url = f"{client.baseurl}/impersonate"
    headers = {
        "Authorization": f"Bearer {client.token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json={"groupId": effective_group}, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Impersonation failed (HTTP {resp.status_code}): {resp.text}\n"
            f"Make sure you have permission to impersonate '{effective_group}'.\n"
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

    print(f"[impersonate] Now acting as group: {effective_group}")
    return effective_group


def make_filename(venue_id: str, base: str, append_date: bool = False) -> str:
    """Build a report filename from the venue ID and an optional date suffix."""
    parts = [p for p in venue_id.strip("/").split("/") if p]
    slug_parts = parts[-3:] if len(parts) >= 3 else parts
    slug = "_".join(slug_parts)
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", slug)

    stem = f"{slug}_{base}"
    if append_date:
        stem += f"_{date.today().isoformat()}"
    return stem + ".html"
