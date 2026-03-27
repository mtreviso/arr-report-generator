import argparse
import collections
import json
import openreview
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import jinja2
from pathlib import Path
import re

class ARRReportGenerator:
    # Any reviewer with confidence at or below this threshold triggers the Low Conf flag.
    LOW_CONF_THRESHOLD = 2

    def __init__(self, username, password, venue_id, me, role='sac',
                 impersonate_group=None, comments_level='basic', skip_api_init=False):
        self.username = username
        self.password = password
        self.venue_id = venue_id
        self.me = me
        self.role = role
        self.impersonate_group = None
        self.comments_level = (comments_level or 'basic').lower()
        if self.comments_level not in {'none', 'basic', 'full'}:
            raise ValueError(f"comments_level must be one of ('none', 'basic', 'full'), got {comments_level!r}")
        self.reply_details = 'replies' if self.comments_level == 'full' else 'directReplies'

        # Data containers
        self.papers_data = []
        self.ac_meta_data = []
        self.comments_data = []
        self.correlation_data = None
        self.attention_papers = []
        self.reviewer_load = {}            # reviewer_id -> paper count
        self.reviewer_confidence_data = {} # reviewer_id -> [confidence scores]

        # Score distributions for visualization
        self.score_distributions = {
            'overall_assessment': [],
            'meta_review': []
        }

        # Caches
        self.ac_email_cache = {}
        self.profile_cache = {}

        # Group index / submission state
        self.client = None
        self.venue_group = None
        self.submission_name = 'Submission'
        self.submissions = []
        self.my_sac_groups = set() if role == 'pc' else set()
        self.group_index = {}
        self.group_index_complete = False
        self.missing_group_ids = set()

        # Cached dataframe for repeated report aggregations
        self._papers_df_cache = None
        self._papers_df_cache_sig = None

        if skip_api_init:
            return

        self.client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net',
                                                     username=username,
                                                     password=password)

        # Impersonate a group BEFORE any data fetching so all subsequent
        # API calls run under the impersonated identity.
        if impersonate_group:
            # Defaults to PC role if none group was provided
            if not impersonate_group.strip() or impersonate_group == "__DEFAULT_PROGRAM_CHAIRS__":
                impersonate_group = venue_id.rstrip('/') + '/' + 'Program_Chairs'
            self.impersonate_group = impersonate_group
            self._apply_impersonation(impersonate_group)

        self.venue_group = self.client.get_group(venue_id)
        self.submission_name = self.venue_group.content['submission_name']['value']
        self.submissions = self.client.get_all_notes(
            invitation=f'{venue_id}/-/{self.submission_name}',
            details=self.reply_details
        )

        # Get SAC groups for filtering
        if role == 'pc':
            self.my_sac_groups = set()
        else:
            try:
                self.my_sac_groups = {
                    g.id
                    for g in self.client.get_all_groups(members=me, prefix=f'{venue_id}/{self.submission_name}')
                    if g.id.endswith('Senior_Area_Chairs')
                }
            except Exception as e:
                print(f"Warning: could not fetch SAC groups via members filter ({e}). Falling back to full scan.")
                self.my_sac_groups = set()
                all_groups = self.client.get_all_groups(prefix=f'{venue_id}/{self.submission_name}')
                for g in all_groups:
                    if g.id.endswith('Senior_Area_Chairs'):
                        if hasattr(g, 'members') and me in g.members:
                            self.my_sac_groups.add(g.id)

        # Group index (bulk pre-fetch for speed)
        self._build_group_index()

    def _build_group_index(self):
        """Pre-fetch all groups under this venue/submission prefix into a dict."""
        prefix = f'{self.venue_id}/{self.submission_name}'
        self.group_index_complete = False
        self.missing_group_ids = set()
        try:
            print(f'Pre-fetching groups with prefix: {prefix}')
            groups = self.client.get_all_groups(prefix=prefix)
            self.group_index = {g.id: g for g in groups}
            self.group_index_complete = True
            print(f'Cached {len(self.group_index)} groups for fast lookup')
        except Exception as e:
            print(f'Warning: could not pre-fetch groups ({e}). Falling back to per-group API calls.')
            self.group_index = {}
            self.group_index_complete = False

    def _get_group_cached(self, group_id):
        if not group_id:
            return None
        group = self.group_index.get(group_id)
        if group is not None:
            return group
        if group_id in self.missing_group_ids:
            return None
        if self.group_index_complete:
            self.missing_group_ids.add(group_id)
            return None
        try:
            group = self.client.get_group(group_id)
            self.group_index[group_id] = group
            return group
        except Exception:
            self.missing_group_ids.add(group_id)
            return None

    def _papers_df(self):
        cache = getattr(self, '_papers_df_cache', None)
        cache_sig = getattr(self, '_papers_df_cache_sig', None)
        if not self.papers_data:
            self._papers_df_cache = pd.DataFrame()
            self._papers_df_cache_sig = (id(self.papers_data), 0)
            return self._papers_df_cache
        sig = (id(self.papers_data), len(self.papers_data))
        if cache is None or cache_sig != sig:
            self._papers_df_cache = pd.DataFrame(self.papers_data)
            self._papers_df_cache_sig = sig
        return self._papers_df_cache

    # -------------------------------------------------------------------------
    # Helper predicates
    # -------------------------------------------------------------------------

    def is_actual_review(self, reply):
        """Count a reply as an actual review only if its invitations include '/-/Official_Review'."""
        return any('/-/Official_Review' in invitation for invitation in reply.get('invitations', []))

    def is_meta_review(self, reply):
        """Check if a reply is a meta-review."""
        return any('/-/Meta_Review' in invitation for invitation in reply.get('invitations', []))

    def is_withdrawn(self, submission):
        withdrawal_conf = submission.content.get("withdrawal_confirmation", {}).get("value", "").strip()
        if withdrawal_conf:
            return True
        venue_val = submission.content.get("venue", {}).get("value", "").lower()
        if "withdrawn" in venue_val:
            return True
        return False

    def is_relevant_comment(self, reply):
        """Check if a reply is a relevant comment type."""
        invitations = reply.get("invitations", [])
        return any(
            part in inv
            for inv in invitations
            for part in ["/-/Author-Editor_Confidential_Comment", "/-/Comment", "/-/Review_Issue_Report"]
        )


    def _get_content_value(self, content, key, default=None):
        value = content.get(key, default) if isinstance(content, dict) else default
        if isinstance(value, dict) and 'value' in value:
            return value.get('value', default)
        return default if value is None else value

    def _normalize_multi_value_field(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]
        items = []
        for raw in raw_values:
            if raw is None:
                continue
            if isinstance(raw, dict) and 'value' in raw:
                raw = raw.get('value')
            if raw is None:
                continue
            if isinstance(raw, str):
                parts = re.split(r'\s*[;|]\s*', raw)
                if len(parts) == 1 and ',' in raw:
                    parts = [part.strip() for part in raw.split(',')]
                for part in parts:
                    label = re.sub(r'\s+', ' ', str(part).strip())
                    if label:
                        items.append(label)
            else:
                label = re.sub(r'\s+', ' ', str(raw).strip())
                if label:
                    items.append(label)
        deduped = []
        seen = set()
        for item in items:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _extract_contribution_types(self, *contents):
        candidate_keys = [
            'contribution_types',
            'contribution_type',
            'types_of_contribution',
            'type_of_contribution',
            'paper_contribution_types',
            'paper_contribution_type',
            'contribution_categories',
            'contribution_category',
            'contributions',
        ]
        contribution_types = []
        for content in contents:
            if not isinstance(content, dict):
                continue
            values = []
            for key in candidate_keys:
                if key in content:
                    values.extend(self._normalize_multi_value_field(self._get_content_value(content, key)))
            if not values:
                for key in content.keys():
                    key_l = str(key).lower()
                    if 'contribution' in key_l and any(token in key_l for token in ('type', 'category')):
                        values.extend(self._normalize_multi_value_field(self._get_content_value(content, key)))
            if values:
                contribution_types.extend(values)
                break
        deduped = []
        seen = set()
        for item in contribution_types:
            norm = item.casefold()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(item)
        return deduped

    def _extract_knows_authors(self, content):
        """Return True if the reviewer indicated they know (some of) the authors.

        ARR's radio field is typically named 'reviewer_identity' with options like:
          "No"  — reviewer does not know the authors
          "Yes" — reviewer knows at least one author
        or in some cycles with longer prose options. We handle all known variants.
        The full debug dump on the first review will show the exact field/value.
        """
        # Known ARR field names, ordered by likelihood
        KEYS = [
            "knowledge_of_or_educated_guess_at_author_identity",  # actual ARR field name (PC view)
            "reviewer_identity",           # most common in recent ARR cycles
            "reviewer_identity_awareness",
            "know_the_authors",
            "author_identity",
            "reviewer_knows_authors",
            "identity_of_authors",
            "authors_identity",
            "reviewer_author_identity",
            "author_awareness",
        ]
        # Phrases that unambiguously mean "I do NOT know the authors"
        NEGATIVE_PHRASES = (
            "do not know",
            "don't know",
            "unaware",
            "not aware",
            "no conflict",
            "i have no",
            "no, i do not",
            "no, the identity",
        )
        # Phrases used by ARR meaning "I DO know the authors"
        POSITIVE_PHRASES = (
            "yes, i know",
            "yes, the identity",
            "i know",
            "i am aware",
            "i'm aware",
            "aware of the identity",
            "know the identity",
            "know at least one",
            "know some of",
            "know one of",
            "submitting author",
        )

        for key in KEYS:
            val = content.get(key, {})
            if isinstance(val, dict):
                val = val.get("value", "")
            if val is None:
                continue
            v = str(val).strip().lower()
            if not v:
                continue
            # Explicit negatives (check before positives — "No, I do not know" must not match)
            if v in ("no", "false", "0", "n/a") or any(neg in v for neg in NEGATIVE_PHRASES):
                continue
            # Explicit affirmatives
            if v in ("yes", "true") or v.startswith("yes"):
                return True
            if any(pos in v for pos in POSITIVE_PHRASES):
                return True

        # Broad fallback: scan all keys whose name contains identity/know/aware/author
        for key, raw in content.items():
            if key in KEYS:
                continue  # already handled above
            kl = key.lower()
            if not any(t in kl for t in ("identity", "know", "aware")):
                continue
            if isinstance(raw, dict):
                raw = raw.get("value", "")
            if raw is None:
                continue
            v = str(raw).strip().lower()
            if not v:
                continue
            if any(neg in v for neg in NEGATIVE_PHRASES) or v in ("no", "false", "0"):
                continue
            if v in ("yes", "true") or v.startswith("yes"):
                return True
            if any(pos in v for pos in POSITIVE_PHRASES):
                return True

        return False

    # -------------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------------

    def format_scores_as_list(self, scores):
        if scores:
            avg = sum(scores) / len(scores)
            score_list = " / ".join(f"{s:.1f}" for s in scores)
            return f"{avg:.1f} ({score_list})"
        return ""

    def format_scores_as_avg_std(self, scores):
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            return f"{avg:.1f} ± {std:.1f}"
        return ""

    def parse_avg(self, s):
        """Extract the first numeric score from a formatted string."""
        try:
            if s is None:
                return float('nan')
            if isinstance(s, (int, float)):
                return float(s)
            match = re.search(r'-?\d+(?:\.\d+)?', str(s))
            return float(match.group(0)) if match else float('nan')
        except Exception:
            return float('nan')

    def parse_meta_review(self, s):
        """Alias for parse_avg — both extract the leading numeric value."""
        return self.parse_avg(s)

    def classify_comment_type(self, reply):
        """Determine the type of comment."""
        invitations = reply.get("invitations", [])
        if any("/-/Review_Issue_Report" in inv for inv in invitations):
            return "Review Issue"
        elif any("/-/Author-Editor_Confidential_Comment" in inv for inv in invitations):
            return "Author-Editor Confidential"
        elif any("/-/Comment" in inv for inv in invitations):
            return "Confidential Comment"
        else:
            return "Other"

    def extract_comment_text(self, reply):
        """Extract and process comment text from a reply."""
        content = reply.get("content", {})
        for key in ["comment", "justification", "text", "response", "value"]:
            if key in content:
                val = content[key]
                raw_text = val.get("value") if isinstance(val, dict) else val
                if raw_text:
                    return self.process_comment_text(raw_text)
        fallback = []
        for k, v in content.items():
            if isinstance(v, dict) and "value" in v:
                fallback.append(f"{k}: {v['value']}")
        return self.process_comment_text("\n".join(fallback)) if fallback else "(No comment text found)"

    def process_comment_text(self, text):
        """Process comment text to fix markdown rendering issues."""
        if not text:
            return text
        text = text.strip()
        lines = text.split('\n')
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, current_indent)
        if min_indent == float('inf'):
            min_indent = 0
        processed_lines = [line[min_indent:] if line.strip() else line for line in lines]
        processed_text = '\n'.join(processed_lines)
        processed_text = re.sub(r'```([\s\S]+?)```',
                                lambda m: '\n```\n' + m.group(1).strip() + '\n```\n',
                                processed_text)
        processed_text = re.sub(r'\n\n+', '\n\n', processed_text)
        processed_text = re.sub(r'^(\s*[-*+])\s+', r'\1 ', processed_text, flags=re.MULTILINE)
        return processed_text

    def infer_role_from_signature(self, signatures):
        if not signatures:
            return "Unknown"
        sig = signatures[0]
        if "/Authors" in sig:
            return "Author"
        elif "/Reviewer" in sig:
            return "Reviewer"
        elif "/Area_Chair" in sig:
            return "Area Chair"
        elif "/Senior_Area_Chairs" in sig:
            return "Senior Area Chair"
        elif "/Program_Chairs" in sig:
            return "Program Chair"
        elif sig.startswith("~"):
            return "User"
        else:
            return "Other"

    def format_timestamp(self, ms_since_epoch):
        if not ms_since_epoch:
            return ""
        dt = datetime.fromtimestamp(ms_since_epoch / 1000)
        return dt.strftime("%Y-%m-%d")

    # -------------------------------------------------------------------------
    # Profile / user resolution (cached)
    # -------------------------------------------------------------------------

    def _get_profile(self, user_id):
        if not user_id:
            return None
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]
        try:
            profile = self.client.get_profile(user_id)
            self.profile_cache[user_id] = profile
            return profile
        except Exception:
            self.profile_cache[user_id] = None
            return None

    def _apply_impersonation(self, group_id: str) -> None:
        """Patch client token to act as `group_id` before any API calls."""
        import requests as _requests
        url = f"{self.client.baseurl}/impersonate"
        headers = {
            "Authorization": f"Bearer {self.client.token}",
            "Content-Type":  "application/json",
        }
        resp = _requests.post(url, json={"groupId": group_id}, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Impersonation failed (HTTP {resp.status_code}): {resp.text}\n"
                f"Make sure your account can impersonate '{group_id}'.\n"
                f"Typical value: <venue_id>/Program_Chairs"
            )
        data = resp.json()
        new_token = data.get("token") or data.get("access_token")
        if not new_token:
            raise RuntimeError(
                f"Impersonation response had no token. Response: {data}"
            )
        self.client.token = new_token
        if hasattr(self.client, "session") and hasattr(self.client.session, "headers"):
            self.client.session.headers["Authorization"] = f"Bearer {new_token}"
        print(f"[impersonate] Now acting as group: {group_id}")

    def get_affiliation_for_user(self, user_id: str) -> str:
        """Return 'Position, Institution (Year–)' for the most recent history entry."""
        if not user_id:
            return ""
        profile = self._get_profile(user_id)
        if not profile:
            return ""
        content = getattr(profile, "content", None)
        if not isinstance(content, dict):
            return ""
        history = content.get("history") or {}
        if isinstance(history, dict):
            history = history.get("value") or []
        if not isinstance(history, list) or not history:
            return ""
        # Sort by start date descending; pick most recent entry
        def _start(h):
            return h.get("start") or 0
        entry = max(history, key=_start)
        parts = []
        position = entry.get("position", "")
        if position:
            parts.append(position)
        inst = entry.get("institution") or {}
        if isinstance(inst, dict):
            inst_name = inst.get("name") or inst.get("domain") or ""
        else:
            inst_name = str(inst)
        if inst_name:
            parts.append(inst_name)
        start_year = entry.get("start")
        end_year   = entry.get("end")
        if start_year:
            year_str = f"{start_year}–{end_year}" if end_year else f"{start_year}–"
            parts.append(f"({year_str})")
        return ", ".join(parts)

    def _sanitize_tilde_id(self, uid):
        """~Foo_Bar1 -> Foo Bar"""
        name = uid.strip().lstrip("~")
        name = re.sub(r'\d+$', '', name)
        name = name.replace("_", " ").strip()
        return name.title()

    def get_display_name_for_user(self, user_id):
        if not user_id:
            return "Unknown"
        default = user_id
        profile = self._get_profile(user_id)
        if not profile:
            return default
        name = None
        content = getattr(profile, "content", None)
        if isinstance(content, dict) and "names" in content:
            names = content["names"]
            if isinstance(names, dict) and "value" in names:
                names_val = names["value"]
            else:
                names_val = names
            if isinstance(names_val, list) and names_val:
                name_obj = names_val[0]
                if isinstance(name_obj, dict):
                    full_name = " ".join(
                        x for x in [
                            name_obj.get("first") or name_obj.get("given"),
                            name_obj.get("last") or name_obj.get("family"),
                        ] if x
                    ).strip()
                    preferred = name_obj.get("preferred")
                    if isinstance(preferred, str) and preferred.strip():
                        name = preferred.strip()
                    elif full_name:
                        name = full_name
        if not name and hasattr(profile, "name"):
            name = profile.name
        if not name and hasattr(profile, "id"):
            name = profile.id
        display_name = str(name) if name else str(default)
        return self._sanitize_tilde_id(display_name)

    def get_email_for_user(self, user_id):
        if not user_id:
            return ""
        if "@" in user_id:
            self.ac_email_cache[user_id] = user_id
            return user_id
        if user_id in self.ac_email_cache:
            return self.ac_email_cache[user_id] or ""
        email = ""
        try:
            profile = self._get_profile(user_id)
            if not profile:
                self.ac_email_cache[user_id] = ""
                return ""
            if hasattr(profile, "preferred_email") and profile.preferred_email:
                email = profile.preferred_email
            content = getattr(profile, "content", None)
            if not email and isinstance(content, dict):
                pe = content.get("preferred_email")
                if isinstance(pe, dict):
                    email = pe.get("value", "") or email
                elif isinstance(pe, str):
                    email = pe
                if not email and "emails" in content:
                    em = content["emails"]
                    if isinstance(em, dict):
                        vals = em.get("value") or em.get("values") or []
                        if isinstance(vals, list) and vals:
                            email = vals[0]
                        elif isinstance(vals, str):
                            email = vals
                    elif isinstance(em, list) and em:
                        email = em[0]
            if not email and hasattr(profile, "email"):
                email = profile.email
        except Exception:
            email = ""
        self.ac_email_cache[user_id] = email or ""
        return email or ""

    def _resolve_ac_user_id(self, ac_entry):
        if not ac_entry:
            return ac_entry
        if ac_entry.startswith("~") or "@" in ac_entry:
            return ac_entry
        if "/Area_Chair_" in ac_entry:
            g = self._get_group_cached(ac_entry)
            if g and getattr(g, "members", None):
                return g.members[0]
        return ac_entry

    # -------------------------------------------------------------------------
    # Data processing
    # -------------------------------------------------------------------------

    def _infer_anonymity(self, content):
        """
        Return 'Yes', 'No', or '' based on any preprint/anonymity-related field.
        'Yes'  → paper has no public preprint (anonymous).
        'No'   → a non-anonymous preprint exists.
        ''     → could not determine.

        ARR uses various field names across rounds; we scan broadly rather than
        hard-coding a single name.  Typical field: 'anonymous_preprint' with
        values like "Yes" / "No" (where Yes = preprint IS anonymous, i.e. no
        public version).
        """
        def _extract(d, key):
            val = d.get(key, {})
            if isinstance(val, dict):
                val = val.get("value", "")
            return str(val).strip()

        # 1) "preprint" field: ARR stores a URL here when a preprint exists,
        #    or leaves it empty / "no" when there isn't one.
        #    Non-empty value (that isn't "no") → has a preprint → Non-anon.
        preprint_val = _extract(content, "preprint")
        if preprint_val:
            v = preprint_val.lower()
            if v in ("no", "false", "0") or v.startswith("no"):
                return "Yes"
            return "No"

        # ARR prose values for preprint_status (binding option strings)
        ANON_PROSE = (
            "no non-anonymous preprint",
            "there is no non-anonymous",
            "will not be posted",
            "no preprint",
        )
        NON_ANON_PROSE = (
            "non-anonymous preprint exists",
            "preprint is available",
            "posted a preprint",
            "there is a non-anonymous",
        )

        BOOL_KEYS = [
            "anonymous_preprint", "anonymized", "preprint_status",
            "is_anonymous", "nonAnonymousPreprint",
            "non_anonymous_preprint", "preprint_availability",
        ]
        for field in BOOL_KEYS:
            val = _extract(content, field)
            if not val:
                continue
            v = val.lower()
            # Prose-first: check full ARR option strings before simple yes/no
            if any(p in v for p in ANON_PROSE):
                return "Yes"
            if any(p in v for p in NON_ANON_PROSE):
                return "No"
            if v in ("yes", "true") or v.startswith("yes"):
                return "Yes"
            if v in ("no", "false", "0") or v.startswith("no"):
                return "No"

        # 3) Broad scan: any remaining key containing "anon" or "preprint"
        for key, raw in content.items():
            kl = key.lower()
            if "anon" not in kl and "preprint" not in kl:
                continue
            if isinstance(raw, dict):
                raw = raw.get("value", "")
            v = str(raw).strip().lower()
            if not v:
                continue
            if v in ("yes", "true") or v.startswith("yes"):
                return "Yes"
            if v in ("no", "false", "0") or v.startswith("no"):
                return "No"

        return ""

    def _get_submission_replies(self, submission):
        details = getattr(submission, 'details', None) or {}
        replies = details.get('replies')
        if replies is not None:
            return replies
        replies = details.get('directReplies')
        if replies is not None:
            return replies
        return []

    def _reply_attr(self, reply, key, default=None):
        if isinstance(reply, dict):
            return reply.get(key, default)
        return getattr(reply, key, default)

    def _reply_content(self, reply):
        content = self._reply_attr(reply, 'content', {})
        return content if isinstance(content, dict) else {}

    def _reply_invitations(self, reply):
        invitations = self._reply_attr(reply, 'invitations', []) or []
        if isinstance(invitations, (list, tuple, set)):
            return list(invitations)
        return [invitations]

    def _reply_link(self, reply):
        forum_id = self._reply_attr(reply, 'forum', '')
        note_id = self._reply_attr(reply, 'id', '')
        return f"https://openreview.net/forum?id={forum_id}&noteId={note_id}" if forum_id and note_id else ""

    def _reply_matches_invitation(self, reply, patterns):
        invitations = [str(inv) for inv in self._reply_invitations(reply)]
        return any(any(pattern in inv for pattern in patterns) for inv in invitations)

    def _extract_numeric_score(self, content, field):
        try:
            value = self._get_content_value(content, field)
            if value in (None, ''):
                return None
            return float(value)
        except Exception:
            return None

    def _extract_meta_review_score(self, content):
        for field in ('overall_assessment', 'overall_rating', 'score'):
            value = self._get_content_value(content, field)
            if value not in (None, ''):
                return value
        return ''

    def _is_emergency_declaration_reply(self, reply):
        invitations = [str(inv) for inv in self._reply_invitations(reply)]
        content = self._reply_content(reply)
        lowered_invitations = [inv.lower() for inv in invitations]

        explicit_patterns = [
            '/-/emergency_declaration',
            '/-/emergencydeclaration',
        ]
        if any(any(pattern in inv for pattern in explicit_patterns) for inv in lowered_invitations):
            return True

        title = str(self._get_content_value(content, 'title', '') or '').strip().lower()
        if 'emergency declaration' in title:
            return True

        if any('emergency_declaration' in inv or 'emergencydeclaration' in inv for inv in lowered_invitations):
            return True

        return False

    def _resolve_reviewer_id(self, sig):
        """Resolve an anonymous reviewer signature to a real user ID."""
        if not sig:
            return sig
        if sig.startswith("~") or "@" in sig:
            return sig
        if '/Reviewer_' in sig:
            grp = self._get_group_cached(sig)
            if grp and getattr(grp, 'members', None):
                return grp.members[0]
        return sig

    def process_papers_data(self):
        """Process all papers data."""
        base_url = "https://openreview.net/forum?id="

        for submission in tqdm(self.submissions):
            # Skip withdrawn or desk rejected papers.
            if self.is_withdrawn(submission):
                print(f"Skipping withdrawn paper: {submission.id}")
                continue
            if "venue" in submission.content and "desk rejected" in submission.content["venue"]["value"].lower():
                print(f"Skipping desk rejected paper: {submission.id}")
                continue

            prefix = f'{self.venue_id}/{self.submission_name}{submission.number}'
            # Process only submissions in your SAC batch (skip for PC role)
            if self.role != 'pc' and not (set(submission.readers) & self.my_sac_groups):
                continue

            # Retrieve the assigned Area Chair (use cached groups first)
            ac_group_id = f'{prefix}/Area_Chairs'
            area_chairs_group = self._get_group_cached(ac_group_id)

            if not area_chairs_group or not getattr(area_chairs_group, "members", None):
                continue

            ac_entry = area_chairs_group.members[0]
            ac_user_id = self._resolve_ac_user_id(ac_entry)
            ac_name = self.get_display_name_for_user(ac_user_id)
            ac_affiliation = self.get_affiliation_for_user(ac_user_id)
            ac_email = ""  # OpenReview forbids SACs to view emails

            paper_type = self._get_content_value(submission.content, "paper_type", "")
            contribution_types = self._extract_contribution_types(submission.content)

            # --- Per-paper reply scan ---
            completed_reviews = 0
            meta_review_score = ""
            confidence_scores = []
            soundness_scores = []
            excitement_scores = []
            overall_assessment_scores = []
            has_review_issue = False
            review_issue_link = ""
            review_issue_count = 0
            has_confidential = False
            has_emergency_declaration = False
            emergency_declaration_link = ""
            emergency_declaration_count = 0
            knows_authors_count = 0  # reviewers who indicated they know the authors

            replies = self._get_submission_replies(submission)

            # On first paper, surface all invitation patterns to help identify emergency fields
            if not self.papers_data and replies:
                all_invs = set()
                for r in replies:
                    for inv in r.get("invitations", []):
                        # Show the suffix after /-/ which is the meaningful part
                        if "/-/" in inv:
                            all_invs.add(inv.split("/-/")[-1])
                if all_invs:
                    print(f"[DEBUG] Reply invitation types on paper #{submission.number}: {sorted(all_invs)}")

            for reply in replies:
                invitations = reply.get("invitations", [])

                # Review issue?
                if any("/-/Review_Issue_Report" in inv for inv in invitations):
                    has_review_issue = True
                    review_issue_count += 1
                    if not review_issue_link:
                        forum_id = reply.get("forum", "")
                        note_id  = reply.get("id", "")
                        review_issue_link = f"https://openreview.net/forum?id={forum_id}&noteId={note_id}"

                # Emergency declaration / emergency-review request
                if self._is_emergency_declaration_reply(reply):
                    has_emergency_declaration = True
                    emergency_declaration_count += 1
                    if not emergency_declaration_link:
                        emergency_declaration_link = self._reply_link(reply)

                # Confidential comment?
                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = self._reply_content(reply)
                    for field, bucket in (
                        ('confidence', confidence_scores),
                        ('soundness', soundness_scores),
                        ('excitement', excitement_scores),
                        ('overall_assessment', overall_assessment_scores),
                    ):
                        value = self._extract_numeric_score(content, field)
                        if value is not None:
                            bucket.append(value)
                    # Single-blind detection: reviewer claims to know the authors
                    # Debug: on the very first review processed, log all identity/know/aware fields
                    # if not self.papers_data and completed_reviews == 1:
                    #     sb_related = {k: v for k, v in content.items()
                    #                   if any(t in k.lower() for t in ("identity", "know", "aware", "author"))}
                    #     if sb_related:
                    #         print(f"[DEBUG SB] Identity/author-awareness fields in first review: {list(sb_related.keys())}")
                    #         for k, v in sb_related.items():
                    #             print(f"  {k!r}: {v!r}")
                    #     else:
                    #         print(f"[DEBUG SB] No identity/know/aware fields found. Review keys: {sorted(content.keys())}")
                    if self._extract_knows_authors(content):
                        knows_authors_count += 1
                    # Track per-reviewer confidence for load-vs-quality chart
                    sigs = reply.get("signatures", [])
                    if sigs:
                        rev_uid = self._resolve_reviewer_id(sigs[0])
                        try:
                            conf_val = self._extract_numeric_score(content, 'confidence')
                            if conf_val is not None:
                                self.reviewer_confidence_data.setdefault(rev_uid, []).append(float(conf_val))
                        except:
                            pass

                if self.is_meta_review(reply):
                    content = self._reply_content(reply)
                    meta_review_score = self._extract_meta_review_score(content)

            # Expected reviews from Reviewers group
            expected_reviews = 0
            reviewers_group_id = f'{prefix}/Reviewers'
            reviewers_group = self._get_group_cached(reviewers_group_id)
            if reviewers_group and getattr(reviewers_group, "members", None):
                expected_reviews = len(reviewers_group.members)
                for rev_id in reviewers_group.members:
                    actual_rev_id = self._resolve_reviewer_id(rev_id)
                    self.reviewer_load[actual_rev_id] = self.reviewer_load.get(actual_rev_id, 0) + 1

            # Emergency reviewer group
            has_emergency_reviewer = False
            emergency_reviewer_count = 0
            EMERGENCY_GROUP_SUFFIXES = [
                "/Emergency_Reviewers",
                "/Emergency_Reviewer",
                "/Emergency_Review_Assignees",
            ]
            for suffix in EMERGENCY_GROUP_SUFFIXES:
                erg = self._get_group_cached(f'{prefix}{suffix}')
                if erg and getattr(erg, 'members', None):
                    has_emergency_reviewer = True
                    emergency_reviewer_count = len(erg.members)
                    break

            status = "✓" if completed_reviews >= 3 else ""

            # Anonymity — log content keys on first paper to help identify the right field
            if not self.papers_data:
                anon_related = {k: v for k, v in submission.content.items()
                                if any(t in k.lower() for t in ("anon", "preprint", "non_anon", "nonanon"))}
                if anon_related:
                    print(f"[DEBUG] Anonymity-related fields found: {list(anon_related.keys())}")
                    for k, v in anon_related.items():
                        print(f"  {k!r}: {v!r}")
                else:
                    print(f"[DEBUG] No anon/preprint fields found. All content keys: {sorted(submission.content.keys())}")
            is_anonymous = self._infer_anonymity(submission.content)

            # Low confidence
            has_low_confidence = any(s <= self.LOW_CONF_THRESHOLD for s in confidence_scores)
            low_conf_reviewers = ", ".join(
                str(round(s)) for s in confidence_scores if s <= self.LOW_CONF_THRESHOLD
            )

            # Format scores
            reviewer_confidence = self.format_scores_as_avg_std(confidence_scores)
            reviewer_soundness  = self.format_scores_as_avg_std(soundness_scores)
            reviewer_excitement = self.format_scores_as_avg_std(excitement_scores)
            reviewer_overall    = self.format_scores_as_avg_std(overall_assessment_scores)

            confidence_list = " / ".join(f"{s:.1f}" for s in confidence_scores) if confidence_scores else ""
            soundness_list  = " / ".join(f"{s:.1f}" for s in soundness_scores)  if soundness_scores  else ""
            excitement_list = " / ".join(f"{s:.1f}" for s in excitement_scores) if excitement_scores else ""
            overall_list    = " / ".join(f"{s:.1f}" for s in overall_assessment_scores) if overall_assessment_scores else ""

            paper_title = submission.content.get("title", {}).get("value", "Untitled")

            paper_data = {
                "Paper #":               submission.number,
                "Paper ID":              submission.id,
                "Title":                 paper_title,
                "Paper Type":            paper_type,
                "Contribution Types":    "; ".join(contribution_types),
                "Contribution Type List": contribution_types,
                "Area Chair":            ac_name,
                "Area Chair ID":         ac_user_id,
                "Area Chair Email":      ac_email,
                "Area Chair Affiliation": ac_affiliation,
                "Completed Reviews":     completed_reviews,
                "Expected Reviews":      expected_reviews,
                "Ready for Rebuttal":    status,
                # New fields
                "Is Anonymous":          is_anonymous,
                "Has Review Issue":      has_review_issue,
                "Review Issue Link":     review_issue_link,
                "Has Confidential":      has_confidential,
                "Has Low Confidence":    has_low_confidence,
                "Low Confidence Reviewers": low_conf_reviewers,
                "Has Emergency Declaration": has_emergency_declaration,
                "Emergency Declaration Link": emergency_declaration_link,
                "Emergency Declaration Count": emergency_declaration_count,
                "Has Emergency Reviewer": has_emergency_reviewer,
                "Emergency Reviewer Count": emergency_reviewer_count,
                "Has Emergency Assigned": bool(has_emergency_declaration and has_emergency_reviewer),
                "Has Emergency Unmet": bool(has_emergency_declaration and not has_emergency_reviewer),
                "Review Issue Count":    review_issue_count,
                "Ethics Flag":           "",   # populated in PCReportGenerator; base always ""
                # Single-blind / knows-authors
                "Knows Authors Count":   knows_authors_count,
                "Has Compromised Review": knows_authors_count > 0,
                # Scores
                "Reviewer Confidence":   reviewer_confidence,
                "Confidence List":       confidence_list,
                "Soundness Score":       reviewer_soundness,
                "Soundness List":        soundness_list,
                "Excitement Score":      reviewer_excitement,
                "Excitement List":       excitement_list,
                "Overall Assessment":    reviewer_overall,
                "Overall List":          overall_list,
                "Meta Review Score":     meta_review_score,
            }

            self.papers_data.append(paper_data)

            if overall_assessment_scores:
                self.score_distributions['overall_assessment'].append(
                    sum(overall_assessment_scores) / len(overall_assessment_scores)
                )
            if meta_review_score:
                try:
                    self.score_distributions['meta_review'].append(float(meta_review_score))
                except:
                    pass

        print('Done!')
        print('Total processed papers:', len(self.papers_data))

    def compute_ac_meta_data(self):
        """Compute metadata aggregated by Area Chair."""
        if not self.papers_data:
            return
        df = self._papers_df()
        meta_df = df.groupby("Area Chair").agg(
            Completed_Reviews=("Completed Reviews", "sum"),
            Expected_Reviews=("Expected Reviews", "sum"),
            Papers_Ready=("Completed Reviews", lambda x: (x >= 3).sum()),
            Num_Papers=("Paper #", "count"),
            Meta_Reviews_Num=("Meta Review Score", lambda x: (x != "").sum()),
            Late_Papers=("Completed Reviews", lambda x: (x < df.loc[x.index, "Expected Reviews"]).sum()),
            Emergency_Declared=("Has Emergency Declaration", "sum"),
            Emergency_Assigned=("Has Emergency Assigned", "sum"),
            Low_Conf_Papers=("Has Low Confidence", "sum"),
            SB_Papers=("Has Compromised Review", "sum"),
        ).reset_index()

        meta_df["All Reviews Ready"] = meta_df.apply(
            lambda row: "✓" if row["Papers_Ready"] == row["Num_Papers"] else "", axis=1
        )
        meta_df["All Meta-reviews Ready"] = meta_df.apply(
            lambda row: "✓" if row["Meta_Reviews_Num"] == row["Num_Papers"] else "", axis=1
        )
        meta_df["Meta_Reviews_Done"] = meta_df.apply(
            lambda row: f"{row['Meta_Reviews_Num']} of {row['Num_Papers']}", axis=1
        )
        meta_df["Emergency_Unassigned"] = (
            meta_df["Emergency_Declared"] - meta_df["Emergency_Assigned"]
        ).clip(lower=0)

        # Per-AC score means (computed separately to use parse_avg)
        parse_fn = self.parse_avg
        def _nanmean_parsed(series):
            vals = [parse_fn(v) for v in series if v]
            vals = [v for v in vals if not np.isnan(v)]
            return round(float(np.mean(vals)), 2) if vals else None

        ac_conf    = df.groupby("Area Chair")["Reviewer Confidence"].apply(_nanmean_parsed)
        ac_overall = df.groupby("Area Chair")["Overall Assessment"].apply(_nanmean_parsed)
        ac_meta    = df.groupby("Area Chair")["Meta Review Score"].apply(_nanmean_parsed)
        meta_df["Avg_Confidence"] = meta_df["Area Chair"].map(ac_conf)
        meta_df["Avg_Overall"]    = meta_df["Area Chair"].map(ac_overall)
        meta_df["Avg_Meta"]       = meta_df["Area Chair"].map(ac_meta)

        # Carry forward the tilde ID
        ac_id_map = df.groupby("Area Chair")["Area Chair ID"].first().to_dict()
        meta_df["Area Chair ID"] = meta_df["Area Chair"].map(ac_id_map).fillna("")
        # Affiliation: look up by tilde ID
        meta_df["Area Chair Affiliation"] = meta_df["Area Chair ID"].map(
            lambda uid: self.get_affiliation_for_user(uid) if uid else ""
        )
        # Email: look up by tilde ID (ac_email_cache is keyed by user_id, not name)
        meta_df["Area Chair Email"] = meta_df["Area Chair ID"].map(
            lambda uid: self.ac_email_cache.get(uid, "")
        )
        # If PC mode and SAC info available per paper, attach SAC name / profile / affiliation per AC
        if "Senior Area Chair" in df.columns:
            sac_map = df.groupby("Area Chair")["Senior Area Chair"].first().to_dict()
            meta_df["Senior Area Chair"] = meta_df["Area Chair"].map(sac_map).fillna("")
            if "Senior Area Chair ID" in df.columns:
                sac_id_map = df.groupby("Area Chair")["Senior Area Chair ID"].first().to_dict()
                meta_df["Senior Area Chair ID"] = meta_df["Area Chair"].map(sac_id_map).fillna("")
            else:
                meta_df["Senior Area Chair ID"] = ""
            if "Senior Area Chair Affiliation" in df.columns:
                sac_aff_map = df.groupby("Area Chair")["Senior Area Chair Affiliation"].first().to_dict()
                meta_df["Senior Area Chair Affiliation"] = meta_df["Area Chair"].map(sac_aff_map).fillna("")
            else:
                meta_df["Senior Area Chair Affiliation"] = ""
        else:
            meta_df["Senior Area Chair"] = ""
            meta_df["Senior Area Chair ID"] = ""
            meta_df["Senior Area Chair Affiliation"] = ""
        meta_df.drop(columns=["Meta_Reviews_Num"], inplace=True)
        self.ac_meta_data = meta_df.to_dict(orient='records')

    def compute_correlation_data(self):
        """Compute correlation between different review scores."""
        if not self.papers_data:
            return
        df = self._papers_df()
        corr_data = pd.DataFrame({
            "Overall_Assessment_Avg":  df["Overall Assessment"].apply(self.parse_avg),
            "Reviewer_Confidence_Avg": df["Reviewer Confidence"].apply(self.parse_avg),
            "Soundness_Score_Avg":     df["Soundness Score"].apply(self.parse_avg),
            "Excitement_Score_Avg":    df["Excitement Score"].apply(self.parse_avg),
            "Meta_Review_Score":       df["Meta Review Score"].apply(self.parse_meta_review)
        })
        corr_table = corr_data.corr().fillna(0).round(2)
        self.score_correlation_data = corr_data
        correlation_matrix = []
        labels = corr_table.index.tolist()
        for i, row_label in enumerate(labels):
            row_data = []
            for j, col_label in enumerate(labels):
                value = corr_table.iloc[i, j]
                if j > i:
                    row_data.append({'value': value, 'color': self._get_correlation_color(value)})
                else:
                    row_data.append(None)
            correlation_matrix.append({'label': row_label.replace('_', ' '), 'cells': row_data})
        self.correlation_data = {
            'labels': [l.replace('_', ' ') for l in labels],
            'matrix': correlation_matrix
        }

    def _get_correlation_color(self, value):
        if abs(value) < 0.3:   return 'bg-gray-100'
        elif abs(value) < 0.5: return 'bg-blue-100'
        elif abs(value) < 0.7: return 'bg-blue-200'
        elif abs(value) < 0.9: return 'bg-blue-300'
        else:                  return 'bg-blue-500'

    def generate_paper_type_distribution(self):
        if not self.papers_data:
            return {'labels': [], 'counts': [], 'rows': []}
        df = self._papers_df()
        rows = []
        for label, group in df.groupby('Paper Type'):
            label = str(label or '').strip()
            if not label:
                continue
            overall_vals = group['Overall Assessment'].apply(self.parse_avg).dropna()
            meta_vals = group['Meta Review Score'].apply(self.parse_avg).dropna()
            soundness_vals = group['Soundness Score'].apply(self.parse_avg).dropna()
            excitement_vals = group['Excitement Score'].apply(self.parse_avg).dropna()
            rows.append({
                'label': label,
                'count': len(group),
                'avg_overall': round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
                'avg_meta': round(float(meta_vals.mean()), 2) if len(meta_vals) else None,
                'avg_soundness': round(float(soundness_vals.mean()), 2) if len(soundness_vals) else None,
                'avg_excitement': round(float(excitement_vals.mean()), 2) if len(excitement_vals) else None,
            })
        rows.sort(key=lambda row: row['count'], reverse=True)
        return {
            'labels': [row['label'] for row in rows],
            'counts': [row['count'] for row in rows],
            'rows': rows,
        }

    def generate_contribution_type_distribution(self):
        if not self.papers_data:
            return {'labels': [], 'counts': [], 'rows': []}
        counts = collections.Counter()
        stats = collections.defaultdict(lambda: {
            'paper_ids': set(),
            'overall': [],
            'meta': [],
            'soundness': [],
            'excitement': [],
        })
        for paper in self.papers_data:
            contribution_types = paper.get('Contribution Type List') or []
            if not contribution_types:
                contribution_types = self._normalize_multi_value_field(paper.get('Contribution Types'))
            if not contribution_types:
                continue
            overall = self.parse_avg(paper.get('Overall Assessment'))
            meta = self.parse_avg(paper.get('Meta Review Score'))
            soundness = self.parse_avg(paper.get('Soundness Score'))
            excitement = self.parse_avg(paper.get('Excitement Score'))
            paper_id = paper.get('Paper ID') or paper.get('Paper #')
            for contribution_type in contribution_types:
                counts.update([contribution_type])
                bucket = stats[contribution_type]
                if paper_id is not None:
                    bucket['paper_ids'].add(paper_id)
                if not np.isnan(overall):
                    bucket['overall'].append(float(overall))
                if not np.isnan(meta):
                    bucket['meta'].append(float(meta))
                if not np.isnan(soundness):
                    bucket['soundness'].append(float(soundness))
                if not np.isnan(excitement):
                    bucket['excitement'].append(float(excitement))
        rows = []
        for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            bucket = stats[label]
            rows.append({
                'label': label,
                'count': count,
                'paper_count': len(bucket['paper_ids']),
                'avg_overall': round(float(np.mean(bucket['overall'])), 2) if bucket['overall'] else None,
                'avg_meta': round(float(np.mean(bucket['meta'])), 2) if bucket['meta'] else None,
                'avg_soundness': round(float(np.mean(bucket['soundness'])), 2) if bucket['soundness'] else None,
                'avg_excitement': round(float(np.mean(bucket['excitement'])), 2) if bucket['excitement'] else None,
            })
        return {
            'labels': [row['label'] for row in rows],
            'counts': [row['count'] for row in rows],
            'rows': rows,
        }

    def generate_review_completion_data(self):
        if not self.ac_meta_data:
            return []
        ac_completion = []
        for ac in self.ac_meta_data:
            completed = ac.get("Completed_Reviews", 0)
            expected  = ac.get("Expected_Reviews", 0)
            percentage = round((completed / expected) * 100) if expected > 0 else 0
            ac_completion.append({'name': ac.get("Area Chair", "Unknown"), 'percentage': percentage})
        ac_completion.sort(key=lambda x: x['percentage'], reverse=True)
        return ac_completion

    def generate_ac_scoring_data(self):
        if not self.papers_data:
            return []
        df = self._papers_df()
        ac_id_map = df.groupby("Area Chair")["Area Chair ID"].first().to_dict() if "Area Chair ID" in df.columns else {}
        result = []
        for ac_name in df['Area Chair'].unique():
            ac_papers = df[df['Area Chair'] == ac_name]
            overall_scores = [s for s in [self.parse_avg(x) for x in ac_papers['Overall Assessment'] if x] if not np.isnan(s)]
            meta_scores    = [s for s in [self.parse_meta_review(x) for x in ac_papers['Meta Review Score'] if x] if not np.isnan(s)]
            if overall_scores or meta_scores:
                result.append({
                    'name':          ac_name,
                    'user_id':       ac_id_map.get(ac_name, ""),
                    'overall_avg':   round(np.mean(overall_scores), 2) if overall_scores else None,
                    'meta_avg':      round(np.mean(meta_scores), 2)    if meta_scores    else None,
                    'overall_count': len(overall_scores),
                    'meta_count':    len(meta_scores),
                    'difference':    round(np.mean(meta_scores) - np.mean(overall_scores), 2)
                                     if overall_scores and meta_scores else None,
                })
        result.sort(key=lambda x: abs(x['difference']) if x['difference'] is not None else 0, reverse=True)
        return result

    def generate_score_scatter_data(self):
        if not self.papers_data:
            return {'scatter': [], 'differences': {'labels': [], 'counts': []}}

        scatter_data = []
        diff_counts = {}

        for paper in self.papers_data:
            overall = self.parse_avg(paper.get("Overall Assessment"))
            meta_val = paper.get("Meta Review Score") or paper.get("AC Score", "")
            meta = self.parse_meta_review(meta_val)
            if np.isnan(overall) or np.isnan(meta):
                continue
            paper_number = paper.get("Paper #", len(scatter_data) + 1)
            scatter_data.append({'x': float(overall), 'y': float(meta), 'paper': paper_number})
            rd = round((float(meta) - float(overall)) * 2) / 2
            diff_counts[rd] = diff_counts.get(rd, 0) + 1

        diff_data = sorted(diff_counts.items())
        return {
            'scatter': scatter_data,
            'differences': {
                'labels': [str(d[0]) for d in diff_data],
                'counts': [d[1] for d in diff_data],
            },
        }

    def generate_score_by_type_data(self):
        """Scores broken down by paper type, contribution type, and by anonymity."""
        if not self.papers_data:
            return {'by_paper_type': [], 'by_contribution_type': [], 'by_anonymity': []}
        df = self._papers_df()

        def _score_row(label, count, overall_vals, meta_vals, conf_vals, soundness_vals, excitement_vals):
            return {
                'label': label,
                'count': count,
                'avg_overall': round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
                'avg_meta': round(float(meta_vals.mean()), 2) if len(meta_vals) else None,
                'avg_conf': round(float(conf_vals.mean()), 2) if len(conf_vals) else None,
                'avg_soundness': round(float(soundness_vals.mean()), 2) if len(soundness_vals) else None,
                'avg_excitement': round(float(excitement_vals.mean()), 2) if len(excitement_vals) else None,
            }

        def _group_scores(group_col, label_map=None):
            rows = []
            for val, grp in df.groupby(group_col):
                if not str(val).strip():
                    continue
                label = label_map.get(val, val) if label_map else val
                rows.append(_score_row(
                    label=label,
                    count=len(grp),
                    overall_vals=grp['Overall Assessment'].apply(self.parse_avg).dropna(),
                    meta_vals=grp['Meta Review Score'].apply(self.parse_avg).dropna(),
                    conf_vals=grp['Reviewer Confidence'].apply(self.parse_avg).dropna(),
                    soundness_vals=grp['Soundness Score'].apply(self.parse_avg).dropna(),
                    excitement_vals=grp['Excitement Score'].apply(self.parse_avg).dropna(),
                ))
            rows.sort(key=lambda row: row['count'], reverse=True)
            return rows

        anon_map = {'Yes': 'Anonymous', 'No': 'Non-anonymous (preprint)', '': 'Unknown'}
        by_anon = _group_scores('Is Anonymous', anon_map)
        by_anon = [row for row in by_anon if row['label'] != 'Unknown' or row['count'] > 0]

        by_type = _group_scores('Paper Type')

        contribution_rows = []
        contribution_acc = collections.defaultdict(list)
        for paper in self.papers_data:
            contribution_types = paper.get('Contribution Type List') or []
            if not contribution_types:
                contribution_types = self._normalize_multi_value_field(paper.get('Contribution Types'))
            if not contribution_types:
                continue
            parsed = {
                'overall': self.parse_avg(paper.get('Overall Assessment')),
                'meta': self.parse_avg(paper.get('Meta Review Score')),
                'conf': self.parse_avg(paper.get('Reviewer Confidence')),
                'soundness': self.parse_avg(paper.get('Soundness Score')),
                'excitement': self.parse_avg(paper.get('Excitement Score')),
            }
            for label in contribution_types:
                contribution_acc[label].append(parsed)
        for label, rows in sorted(contribution_acc.items(), key=lambda item: len(item[1]), reverse=True):
            overall_vals = pd.Series([row['overall'] for row in rows], dtype='float64').dropna()
            meta_vals = pd.Series([row['meta'] for row in rows], dtype='float64').dropna()
            conf_vals = pd.Series([row['conf'] for row in rows], dtype='float64').dropna()
            soundness_vals = pd.Series([row['soundness'] for row in rows], dtype='float64').dropna()
            excitement_vals = pd.Series([row['excitement'] for row in rows], dtype='float64').dropna()
            contribution_rows.append(_score_row(
                label=label,
                count=len(rows),
                overall_vals=overall_vals,
                meta_vals=meta_vals,
                conf_vals=conf_vals,
                soundness_vals=soundness_vals,
                excitement_vals=excitement_vals,
            ))

        return {
            'by_paper_type': by_type,
            'by_contribution_type': contribution_rows,
            'by_anonymity': by_anon,
        }

    def generate_reviewer_load_quality_data(self):
        """Per-reviewer: papers assigned vs average confidence score (for scatter plot)."""
        results = []
        for rev_id, load in self.reviewer_load.items():
            confs = self.reviewer_confidence_data.get(rev_id, [])
            avg_conf = round(float(np.mean(confs)), 2) if confs else None
            results.append({
                "load": load,
                "avg_confidence": avg_conf,
            })
        # Aggregate into (load, avg_confidence) pairs — bucket by load
        buckets = {}
        for r in results:
            l = r["load"]
            if l not in buckets:
                buckets[l] = []
            if r["avg_confidence"] is not None:
                buckets[l].append(r["avg_confidence"])
        agg = []
        for l in sorted(buckets.keys()):
            confs = buckets[l]
            agg.append({
                "load":          l,
                "avg_confidence": round(float(np.mean(confs)), 2) if confs else None,
                "reviewer_count": len(results) and sum(1 for r in results if r["load"] == l),
            })
        total_reviewers = len(self.reviewer_load)
        avg_load = round(sum(self.reviewer_load.values()) / total_reviewers, 2) if total_reviewers else 0
        return {
            "points": agg,
            "raw":    results,
            "total_reviewers": total_reviewers,
            "avg_load": avg_load,
        }

    def process_comments_data(self):
        """Process all relevant comments."""
        self.comments_data = []
        if self.comments_level == 'none' or not self.papers_data:
            return
        base_url = "https://openreview.net/forum"
        paper_numbers = {p["Paper #"] for p in self.papers_data}
        for submission in self.submissions:
            if submission.number not in paper_numbers:
                continue
            for reply in self._get_submission_replies(submission):
                if self.is_relevant_comment(reply):
                    forum_id = reply.get("forum", "")
                    note_id  = reply.get("id", "")
                    link     = f"{base_url}?id={forum_id}&noteId={note_id}"
                    signatures = reply.get("signatures", [])
                    role = self.infer_role_from_signature(signatures)
                    # Resolve display name for non-anonymous signers (tilde IDs)
                    author_name = ""
                    if signatures:
                        sig = signatures[0]
                        if sig.startswith("~"):
                            author_name = self.get_display_name_for_user(sig)
                    self.comments_data.append({
                        "Paper #":    submission.number,
                        "Paper ID":   submission.id,
                        "Type":       self.classify_comment_type(reply),
                        "Role":       role,
                        "AuthorName": author_name,
                        "Date":       self.format_timestamp(reply.get("tcdate")),
                        "Content":    self.extract_comment_text(reply),
                        "Link":       link,
                        "ReplyTo":    reply.get("replyto"),
                        "NoteId":     note_id,
                        "AuthorID":   signatures[0] if signatures and signatures[0].startswith("~") else "",
                    })

    def process_data(self):
        self.process_papers_data()
        self.compute_ac_meta_data()
        self.compute_correlation_data()
        self.compute_attention_papers()
        if self.comments_level != 'none':
            self.process_comments_data()
        else:
            self.comments_data = []

    def generate_histogram_data(self):
        bins_overall = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                        4.0, 4.25, 4.5, 4.75, 5.0, 5.25]
        bins_meta = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        histogram_data = {
            'overall_assessment': {'counts': [], 'labels': [f"{x:.2f}" for x in bins_overall[:-1]]},
            'meta_review':        {'counts': [], 'labels': [f"{x:.2f}" for x in bins_meta[:-1]]}
        }
        if self.score_distributions['overall_assessment']:
            hist, _ = np.histogram(self.score_distributions['overall_assessment'], bins=bins_overall)
            histogram_data['overall_assessment']['counts'] = hist.tolist()
        if self.score_distributions['meta_review']:
            hist, _ = np.histogram(self.score_distributions['meta_review'], bins=bins_meta)
            histogram_data['meta_review']['counts'] = hist.tolist()
        return histogram_data

    def organize_comments_by_paper(self):
        if not self.comments_data:
            return []
        comments_by_paper = {}
        for comment in self.comments_data:
            p = comment["Paper #"]
            comments_by_paper.setdefault(p, []).append(comment)
        comment_trees = []
        for paper_num, comments in comments_by_paper.items():
            comment_map = {c["NoteId"]: c for c in comments}
            roots = [c for c in comments if not c["ReplyTo"] or c["ReplyTo"] not in comment_map]
            paper_tree = {"paper_num": paper_num, "threads": [self._build_comment_tree(r, comments) for r in roots]}
            comment_trees.append(paper_tree)
        return comment_trees

    def _build_comment_tree(self, root_comment, all_comments):
        root_id = root_comment["NoteId"]
        children = [self._build_comment_tree(c, all_comments) for c in all_comments if c["ReplyTo"] == root_id]
        return {"comment": root_comment, "children": children}


    def _get_emergency_reviewer_count(self, paper):
        try:
            return max(0, int(paper.get('Emergency Reviewer Count') or 0))
        except Exception:
            return 0

    def _derive_attention_paper(self, paper):
        expected_reviews = int(paper.get('Expected Reviews') or 0)
        completed_reviews = int(paper.get('Completed Reviews') or 0)
        emergency_added = self._get_emergency_reviewer_count(paper)
        initial_expected = max(0, expected_reviews - emergency_added)
        missing_reviews = max(0, expected_reviews - completed_reviews)
        has_issue = bool(paper.get('Has Review Issue'))
        has_ethics = bool((paper.get('Ethics Flag') or '').strip())
        has_missing = expected_reviews > 0 and completed_reviews < expected_reviews
        has_emergency_decl = bool(paper.get('Has Emergency Declaration'))

        if not (has_issue or has_ethics or has_missing or has_emergency_decl):
            return None

        enriched = dict(paper)
        enriched['Missing Reviews'] = missing_reviews
        enriched['Initial Expected Reviews'] = initial_expected
        enriched['Emergency Assigned Later'] = emergency_added
        enriched['Has Missing Reviews'] = has_missing
        return enriched

    def compute_attention_papers(self):
        if not self.papers_data:
            self.attention_papers = []
            return
        rows = []
        for paper in self.papers_data:
            enriched = self._derive_attention_paper(paper)
            if enriched is not None:
                rows.append(enriched)
        rows.sort(key=lambda p: (
            0 if p.get('Has Review Issue') else 1,
            0 if p.get('Missing Reviews', 0) else 1,
            0 if p.get('Has Emergency Declaration') else 1,
            -int(p.get('Missing Reviews', 0) or 0),
            p.get('Paper #', 0),
        ))
        self.attention_papers = rows

    def attention_template_flags(self):
        return {
            'attention_has_sac': any(str(p.get('Senior Area Chair') or '').strip() for p in self.attention_papers),
            'attention_has_ac': any(str(p.get('Area Chair') or '').strip() for p in self.attention_papers),
        }

    def _user_context(self):
        """Return template variables describing the current user and impersonation."""
        return {
            'report_user': self.me or '',
            'report_role': (self.role or '').upper(),
            'report_username': self.username or '',
            'report_impersonate_group': self.impersonate_group or '',
        }

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def _resolve_template_dir(self):
        """Find the templates directory: alongside this file, or in CWD."""
        candidates = [
            Path(__file__).parent / "templates",
            Path("templates"),
        ]
        for p in candidates:
            if p.exists() and (p / "review_report.html").exists():
                return p
        raise FileNotFoundError(
            "Could not find the 'templates/' directory. "
            "Make sure it is in the same folder as arr_report_generator.py "
            "or in your current working directory."
        )

    def generate_report(self, output_dir=".", filename="review_report.html"):
        """Generate the HTML report using external Jinja2 templates."""
        os.makedirs(output_dir, exist_ok=True)

        self.process_data()

        paper_type_distribution  = self.generate_paper_type_distribution()
        contribution_type_distribution = self.generate_contribution_type_distribution()
        review_completion_data   = self.generate_review_completion_data()
        score_scatter_data       = self.generate_score_scatter_data()
        ac_scoring_data          = self.generate_ac_scoring_data()
        score_by_type_data       = self.generate_score_by_type_data()
        reviewer_load_quality    = self.generate_reviewer_load_quality_data()

        template_data = {
            "title":                 f"ARR Review Report: {self.venue_id}",
            "generated_date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":              self.venue_id,
            **self._user_context(),
            "papers":                self.papers_data,
            "ac_meta":               self.ac_meta_data,
            "attention_papers":      self.attention_papers,
            **self.attention_template_flags(),
            "comments_count":        len(self.comments_data),
            "comments":              self.comments_data,
            "comments_level":        self.comments_level,
            "comments_enabled":      self.comments_level != "none",
            "histogram_data":        self.generate_histogram_data(),
            "correlation_data":      self.correlation_data,
            "paper_type_distribution": paper_type_distribution,
            "contribution_type_distribution": contribution_type_distribution,
            "review_completion_data":  review_completion_data,
            "score_scatter_data":    score_scatter_data,
            "ac_scoring_data":       ac_scoring_data,
            "score_by_type_data":    score_by_type_data,
            "reviewer_load_quality": reviewer_load_quality,
        }

        template_dir = self._resolve_template_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        template    = env.get_template("review_report.html")
        html_content = template.render(**template_data)

        output_path = Path(output_dir) / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate ARR Review Report')
    parser.add_argument('--username', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--venue_id', required=True)
    parser.add_argument('--me', required=True)
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()
    generator = ARRReportGenerator(
        username=args.username, password=args.password,
        venue_id=args.venue_id, me=args.me
    )
    report_path = generator.generate_report(output_dir=args.output_dir)
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
