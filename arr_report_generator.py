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

    def __init__(self, username, password, venue_id, me, role='sac'):
        self.username = username
        self.password = password
        self.venue_id = venue_id
        self.me = me
        self.role = role
        self.client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', 
                                                     username=username, 
                                                     password=password)
        self.venue_group = self.client.get_group(venue_id)
        self.submission_name = self.venue_group.content['submission_name']['value']
        self.submissions = self.client.get_all_notes(
            invitation=f'{venue_id}/-/{self.submission_name}', 
            details='replies'
        )

        # Get SAC groups for filtering
        try:
            self.my_sac_groups = {
                g.id
                for g in self.client.get_all_groups(members=me, prefix=f'{venue_id}/{self.submission_name}')
                if g.id.endswith('Senior_Area_Chairs')
            }
        except:
            # Fix for old API version
            self.my_sac_groups = set()
            all_groups = self.client.get_all_groups(prefix=f'{venue_id}/{self.submission_name}')
            for g in all_groups:
                if g.id.endswith('Senior_Area_Chairs'):
                    if hasattr(g, 'members') and me in g.members:
                        self.my_sac_groups.add(g.id)

        # Data containers
        self.papers_data = []
        self.ac_meta_data = []
        self.comments_data = []
        self.correlation_data = None
        
        # Score distributions for visualization
        self.score_distributions = {
            'overall_assessment': [],
            'meta_review': []
        }

        # Caches
        self.ac_email_cache = {}
        self.profile_cache = {}

        # Group index (bulk pre-fetch for speed)
        self.group_index = {}
        self._build_group_index()

    def _build_group_index(self):
        """Pre-fetch all groups under this venue/submission prefix into a dict."""
        prefix = f'{self.venue_id}/{self.submission_name}'
        try:
            print(f'Pre-fetching groups with prefix: {prefix}')
            groups = self.client.get_all_groups(prefix=prefix)
            self.group_index = {g.id: g for g in groups}
            print(f'Cached {len(self.group_index)} groups for fast lookup')
        except Exception as e:
            print(f'Warning: could not pre-fetch groups ({e}). Falling back to per-group API calls.')
            self.group_index = {}

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
        """Extract the average score from a formatted string."""
        try:
            return float(s.split()[0])
        except Exception:
            return float('nan')

    def parse_meta_review(self, s):
        try:
            return float(s)
        except Exception:
            return float('nan')

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

    def _sanitize_tilde_id(self, uid):
        """~Foo_Bar1 → Foo Bar"""
        if not uid or not uid.startswith("~"):
            return uid
        import re
        name = uid.lstrip("~")
        name = re.sub(r'\\d+$', '', name)
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
        return str(name) if name else self._sanitize_tilde_id(str(default))

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
            g = self.group_index.get(ac_entry)
            if g is None:
                try:
                    g = self.client.get_group(ac_entry)
                    self.group_index[ac_entry] = g
                except Exception:
                    g = None
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
                return "Yes"   # explicitly said no preprint → anonymous
            return "No"        # URL or any other non-empty value → non-anon

        # 2) Boolean / yes-no fields
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
            if v in ("yes", "true", "1") or v.startswith("yes"):
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
            if v in ("yes", "true", "1") or v.startswith("yes"):
                return "Yes"
            if v in ("no", "false", "0") or v.startswith("no"):
                return "No"

        return ""

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
            area_chairs_group = self.group_index.get(ac_group_id)
            if area_chairs_group is None:
                try:
                    area_chairs_group = self.client.get_group(ac_group_id)
                    self.group_index[ac_group_id] = area_chairs_group
                except Exception:
                    area_chairs_group = None

            if not area_chairs_group or not getattr(area_chairs_group, "members", None):
                continue

            ac_entry = area_chairs_group.members[0]
            ac_user_id = self._resolve_ac_user_id(ac_entry)
            ac_name = self.get_display_name_for_user(ac_user_id)
            ac_email = ""  # OpenReview forbids SACs to view emails

            paper_type = submission.content.get("paper_type", {}).get("value", "")

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

            replies = submission.details.get("replies", [])

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

                # Emergency review declaration (AC requests emergency reviewer)?
                EMERGENCY_DECL_PATTERNS = [
                    "/-/Emergency_Review_Request",
                    "/-/Emergency_Reviewer_Recruitment",
                    "/-/Emergency_Reviewer_Request",
                    "/-/Emergency_Review",
                ]
                if any(any(p in inv for p in EMERGENCY_DECL_PATTERNS) for inv in invitations):
                    has_emergency_declaration = True
                    emergency_declaration_count += 1
                    if not emergency_declaration_link:
                        forum_id = reply.get("forum", "")
                        note_id  = reply.get("id", "")
                        emergency_declaration_link = f"https://openreview.net/forum?id={forum_id}&noteId={note_id}"

                # Confidential comment?
                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = reply.get("content", {})
                    try:
                        val = content.get("confidence", {}).get("value", None)
                        if val is not None:
                            confidence_scores.append(float(val))
                    except:
                        pass
                    try:
                        val = content.get("soundness", {}).get("value", None)
                        if val is not None:
                            soundness_scores.append(float(val))
                    except:
                        pass
                    try:
                        val = content.get("excitement", {}).get("value", None)
                        if val is not None:
                            excitement_scores.append(float(val))
                    except:
                        pass
                    try:
                        val = content.get("overall_assessment", {}).get("value", None)
                        if val is not None:
                            overall_assessment_scores.append(float(val))
                    except:
                        pass

                if self.is_meta_review(reply):
                    content = reply.get("content", {})
                    if "overall_assessment" in content:
                        meta_review_score = content["overall_assessment"].get("value", "")
                    elif "overall_rating" in content:
                        meta_review_score = content["overall_rating"].get("value", "")
                    elif "score" in content:
                        meta_review_score = content["score"].get("value", "")

            # Expected reviews from Reviewers group
            expected_reviews = 0
            reviewers_group_id = f'{prefix}/Reviewers'
            reviewers_group = self.group_index.get(reviewers_group_id)
            if reviewers_group is None:
                try:
                    reviewers_group = self.client.get_group(reviewers_group_id)
                    self.group_index[reviewers_group_id] = reviewers_group
                except Exception:
                    reviewers_group = None
            if reviewers_group and getattr(reviewers_group, "members", None):
                expected_reviews = len(reviewers_group.members)

            # Emergency reviewer group
            has_emergency_reviewer = False
            emergency_reviewer_count = 0
            EMERGENCY_GROUP_SUFFIXES = [
                "/Emergency_Reviewers",
                "/Emergency_Reviewer",
                "/Emergency_Review_Assignees",
            ]
            for suffix in EMERGENCY_GROUP_SUFFIXES:
                erg = self.group_index.get(f'{prefix}{suffix}')
                if erg is None:
                    try:
                        erg = self.client.get_group(f'{prefix}{suffix}')
                        self.group_index[f'{prefix}{suffix}'] = erg
                    except Exception:
                        erg = None
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
                "Area Chair":            ac_name,
                "Area Chair ID":         ac_user_id,
                "Area Chair Email":      ac_email,
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
                "Review Issue Count":    review_issue_count,
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
        df = pd.DataFrame(self.papers_data)
        meta_df = df.groupby("Area Chair").agg(
            Completed_Reviews=("Completed Reviews", "sum"),
            Expected_Reviews=("Expected Reviews", "sum"),
            Papers_Ready=("Completed Reviews", lambda x: (x >= 3).sum()),
            Num_Papers=("Paper #", "count"),
            Meta_Reviews_Num=("Meta Review Score", lambda x: (x != "").sum()),
            Late_Papers=("Completed Reviews", lambda x: (x < df.loc[x.index, "Expected Reviews"]).sum()),
            Emergency_Declared=("Has Emergency Declaration", "sum"),
            Emergency_Assigned=("Has Emergency Reviewer", "sum"),
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
        meta_df["Area Chair Email"] = meta_df["Area Chair"].map(
            lambda ac: self.ac_email_cache.get(ac, "")
        )
        # If PC mode and SAC info available per paper, attach SAC name per AC
        if "Senior Area Chair" in df.columns:
            sac_map = df.groupby("Area Chair")["Senior Area Chair"].first().to_dict()
            meta_df["Senior Area Chair"] = meta_df["Area Chair"].map(sac_map).fillna("")
        else:
            meta_df["Senior Area Chair"] = ""
        meta_df.drop(columns=["Meta_Reviews_Num"], inplace=True)
        self.ac_meta_data = meta_df.to_dict(orient='records')

    def compute_correlation_data(self):
        """Compute correlation between different review scores."""
        if not self.papers_data:
            return
        df = pd.DataFrame(self.papers_data)
        corr_data = pd.DataFrame({
            "Overall_Assessment_Avg":  df["Overall Assessment"].apply(self.parse_avg),
            "Reviewer_Confidence_Avg": df["Reviewer Confidence"].apply(self.parse_avg),
            "Soundness_Score_Avg":     df["Soundness Score"].apply(self.parse_avg),
            "Excitement_Score_Avg":    df["Excitement Score"].apply(self.parse_avg),
            "Meta_Review_Score":       df["Meta Review Score"].apply(
                lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip()) else np.nan
            )
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
            return {'labels': [], 'counts': []}
        df = pd.DataFrame(self.papers_data)
        type_counts = df["Paper Type"].value_counts().to_dict()
        return {'labels': list(type_counts.keys()), 'counts': list(type_counts.values())}

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
        df = pd.DataFrame(self.papers_data)
        result = []
        for ac_name in df['Area Chair'].unique():
            ac_papers = df[df['Area Chair'] == ac_name]
            overall_scores = [s for s in [self.parse_avg(x) for x in ac_papers['Overall Assessment'] if x] if not np.isnan(s)]
            meta_scores    = [s for s in [self.parse_meta_review(x) for x in ac_papers['Meta Review Score'] if x] if not np.isnan(s)]
            if overall_scores or meta_scores:
                result.append({
                    'name':          ac_name,
                    'email':         self.ac_email_cache.get(ac_name, ""),
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
        if not hasattr(self, 'score_correlation_data') or self.score_correlation_data is None:
            return {'scatter': [], 'differences': {'labels': [], 'counts': []}}
        df = self.score_correlation_data
        scatter_data = []
        for i, row in df.iterrows():
            if not np.isnan(row["Overall_Assessment_Avg"]) and not np.isnan(row["Meta_Review_Score"]):
                paper_number = self.papers_data[i]["Paper #"] if i < len(self.papers_data) else i
                scatter_data.append({'x': float(row["Overall_Assessment_Avg"]), 'y': float(row["Meta_Review_Score"]), 'paper': paper_number})
        diff_counts = {}
        for pt in scatter_data:
            rd = round((pt['y'] - pt['x']) * 2) / 2
            diff_counts[rd] = diff_counts.get(rd, 0) + 1
        diff_data = sorted(diff_counts.items())
        return {
            'scatter': scatter_data,
            'differences': {'labels': [str(d[0]) for d in diff_data], 'counts': [d[1] for d in diff_data]}
        }

    def process_comments_data(self):
        """Process all relevant comments."""
        base_url = "https://openreview.net/forum"
        paper_numbers = {p["Paper #"] for p in self.papers_data}
        for submission in self.submissions:
            if submission.number not in paper_numbers:
                continue
            for reply in submission.details.get("replies", []):
                if self.is_relevant_comment(reply):
                    forum_id = reply.get("forum", "")
                    note_id  = reply.get("id", "")
                    link     = f"{base_url}?id={forum_id}&noteId={note_id}"
                    self.comments_data.append({
                        "Paper #":  submission.number,
                        "Paper ID": submission.id,
                        "Type":     self.classify_comment_type(reply),
                        "Role":     self.infer_role_from_signature(reply.get("signatures", [])),
                        "Date":     self.format_timestamp(reply.get("tcdate")),
                        "Content":  self.extract_comment_text(reply),
                        "Link":     link,
                        "ReplyTo":  reply.get("replyto"),
                        "NoteId":   note_id,
                    })

    def process_data(self):
        self.process_papers_data()
        self.compute_ac_meta_data()
        self.compute_correlation_data()
        self.process_comments_data()

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

    def generate_report(self, output_dir="."):
        """Generate the HTML report using external Jinja2 templates."""
        os.makedirs(output_dir, exist_ok=True)

        self.process_data()

        paper_type_distribution  = self.generate_paper_type_distribution()
        review_completion_data   = self.generate_review_completion_data()
        score_scatter_data       = self.generate_score_scatter_data()
        ac_scoring_data          = self.generate_ac_scoring_data()

        template_data = {
            "title":                 f"ARR Review Report: {self.venue_id}",
            "generated_date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":              self.venue_id,
            "papers":                self.papers_data,
            "ac_meta":               self.ac_meta_data,
            "comments_count":        len(self.comments_data),
            "comments":              self.comments_data,
            "comment_trees":         self.organize_comments_by_paper(),
            "histogram_data":        self.generate_histogram_data(),
            "correlation_data":      self.correlation_data,
            "paper_type_distribution": paper_type_distribution,
            "review_completion_data":  review_completion_data,
            "score_scatter_data":    score_scatter_data,
            "ac_scoring_data":       ac_scoring_data,
        }

        template_dir = self._resolve_template_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        template    = env.get_template("review_report.html")
        html_content = template.render(**template_data)

        output_path = Path(output_dir) / "review_report.html"
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
