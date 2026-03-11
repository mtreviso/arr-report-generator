"""
Commitment Phase Report Generator for ACL Conferences

Supports --role sac (default), ac, or pc:
  sac: papers where you are in Senior_Area_Chairs group (fixes previous AC-only bug)
  ac:  papers where you are in Area_Chairs group
  pc:  all papers that have a paper_link (no assignment filter)
"""

import openreview
import numpy as np
import os
import jinja2
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from arr_report_generator import ARRReportGenerator

VALID_ROLES = ('sac', 'ac', 'pc')


class CommitmentReportGenerator(ARRReportGenerator):

    def __init__(self, username, password, venue_id, me, role='sac'):
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        self.role = role

        # Direct init — different flow from ARRReportGenerator.__init__
        self.username  = username
        self.password  = password
        self.venue_id  = venue_id
        self.me        = me
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=username,
            password=password
        )
        self.venue_group     = self.client.get_group(venue_id)
        self.submission_name = self.venue_group.content['submission_name']['value']

        # Data containers
        self.papers_data         = []
        self.comments_data       = []
        self.ac_meta_data        = []
        self.correlation_data    = None
        self.score_distributions = {'overall_assessment': [], 'meta_review': []}

        # Caches
        self.ac_email_cache       = {}
        self.profile_cache        = {}
        self.linked_note_cache    = {}
        self.linked_replies_cache = {}

        # Build group index then find assigned submissions
        self.group_index = {}
        self._build_group_index()
        print(f"Role: {role.upper()} | Venue: {venue_id} | Me: {me}")
        self._find_submissions()

    # -----------------------------------------------------------------------
    # Group index + assignment discovery
    # -----------------------------------------------------------------------

    def _build_group_index(self):
        prefix = f'{self.venue_id}/{self.submission_name}'
        try:
            print(f"Pre-fetching groups with prefix: {prefix}")
            groups = self.client.get_all_groups(prefix=prefix)
            self.group_index = {g.id: g for g in groups}
            print(f"Cached {len(self.group_index)} groups for fast lookup")
        except Exception as e:
            print(f"Warning: could not pre-fetch groups ({e}). Falling back to per-group calls.")
            self.group_index = {}

    def _is_me_in_group(self, group_id):
        group = self.group_index.get(group_id)
        if group is None:
            try:
                group = self.client.get_group(group_id)
                self.group_index[group_id] = group
            except Exception:
                return False
        return bool(getattr(group, 'members', None)) and self.me in group.members

    def _find_submissions(self):
        """Fetch all submissions, then filter by role-based assignment."""
        print(f"Retrieving submissions for {self.venue_id}...")
        try:
            all_subs = self.client.get_all_notes(
                invitation=f'{self.venue_id}/-/{self.submission_name}',
                details='replies'
            )
            print(f"Retrieved {len(all_subs)} total submissions")
        except Exception as e:
            print(f"Error retrieving submissions: {e}")
            self.submissions = []
            return

        self.submissions = []
        role = self.role

        for sub in tqdm(all_subs, desc="Filtering assigned papers"):
            # Must have a linked ARR submission to be processable
            link = sub.content.get("paper_link", {}).get("value", "").strip()
            if not link:
                continue

            # Skip withdrawn / desk-rejected
            try:
                venue_val = sub.content.get('venue', {}).get('value', '').lower()
                if 'withdrawn' in venue_val or 'desk rejected' in venue_val:
                    continue
                if sub.content.get('withdrawal_confirmation', {}).get('value', '').strip():
                    continue
            except Exception:
                pass

            num = sub.number
            if role == 'pc':
                self.submissions.append(sub)
            elif role == 'sac':
                if self._is_me_in_group(f"{self.venue_id}/{self.submission_name}{num}/Senior_Area_Chairs"):
                    self.submissions.append(sub)
            elif role == 'ac':
                if self._is_me_in_group(f"{self.venue_id}/{self.submission_name}{num}/Area_Chairs"):
                    self.submissions.append(sub)

        print(f"Found {len(self.submissions)} papers for role={role}")

    # -----------------------------------------------------------------------
    # Note/reply predicates (override base class which expects dicts)
    # -----------------------------------------------------------------------

    def is_actual_review(self, reply):
        invs = getattr(reply, 'invitations', None) or reply.get('invitations', [])
        return any('/-/Official_Review' in i for i in invs)

    def is_meta_review(self, reply):
        invs = getattr(reply, 'invitations', None) or reply.get('invitations', [])
        return any('/-/Meta_Review' in i for i in invs)

    def is_relevant_comment(self, reply):
        invs = getattr(reply, 'invitations', None) or reply.get('invitations', [])
        PARTS = ("/-/Author-Editor_Confidential_Comment", "/-/Comment",
                 "/-/Review_Issue_Report", "/-/Meta-Review_Issue_Report")
        return any(p in i for i in invs for p in PARTS)

    def classify_comment_type(self, reply):
        invs = getattr(reply, 'invitations', None) or reply.get('invitations', [])
        if any("/-/Review_Issue_Report"         in i for i in invs): return "Review Issue"
        if any("/-/Meta-Review_Issue_Report"    in i for i in invs): return "Meta Review Issue"
        if any("/-/Author-Editor_Confidential_Comment" in i for i in invs): return "Author-Editor Confidential"
        if any("/-/Comment"                     in i for i in invs): return "Confidential Comment"
        return "Other"

    def extract_comment_text(self, reply):
        content = getattr(reply, 'content', None) or reply.get("content", {})
        for key in ["comment", "justification", "text", "response", "value"]:
            if key in content:
                val = content[key]
                if isinstance(val, dict):
                    val = val.get("value", "")
                return str(val) if val else ""
        parts = []
        for k, v in content.items():
            if isinstance(v, dict) and "value" in v:
                parts.append(f"{k}: {v['value']}")
        return "\n".join(parts) if parts else "(No comment text found)"

    def infer_role_from_signature(self, signatures):
        if not signatures:
            return "Unknown"
        sig = signatures[0]
        if "/Authors"           in sig: return "Author"
        if "/Reviewer"          in sig: return "Reviewer"
        if "/Area_Chair"        in sig: return "Area Chair"
        if "/Senior_Area_Chairs" in sig: return "Senior Area Chair"
        if "/Program_Chairs"    in sig: return "Program Chair"
        if sig.startswith("~"):         return "User"
        return "Other"

    # -----------------------------------------------------------------------
    # Linked-forum helpers (cached)
    # -----------------------------------------------------------------------

    def _parse_linked_forum_id(self, paper_link):
        if not paper_link:
            return None
        try:
            qs = parse_qs(urlparse(paper_link).query)
            return qs.get("id", qs.get("noteId", [None]))[0]
        except Exception:
            return None

    def _get_linked_note(self, forum_id):
        if forum_id in self.linked_note_cache:
            return self.linked_note_cache[forum_id]
        try:
            note = self.client.get_note(id=forum_id)
        except Exception:
            note = None
        self.linked_note_cache[forum_id] = note
        return note

    def _get_linked_replies(self, forum_id):
        if forum_id in self.linked_replies_cache:
            return self.linked_replies_cache[forum_id]
        try:
            replies = self.client.get_notes(forum=forum_id)
        except Exception:
            replies = []
        self.linked_replies_cache[forum_id] = replies
        return replies

    # -----------------------------------------------------------------------
    # Data processing
    # -----------------------------------------------------------------------

    def process_papers_data(self):
        if not self.submissions:
            print("No submissions to process.")
            return

        print(f"Processing {len(self.submissions)} papers...")
        base_url = "https://openreview.net/forum?id="

        for sub in tqdm(self.submissions, desc="Processing papers"):
            # SAC meta-review on commitment submission
            recommendation    = ""
            presentation_mode = ""
            award             = ""
            try:
                metas = self.client.get_notes(
                    invitation=f"{self.venue_id}/{self.submission_name}{sub.number}/-/Meta_Review"
                )
                if metas:
                    c = metas[0].content
                    recommendation    = c.get("recommendation",    {}).get("value", "")
                    presentation_mode = c.get("presentation_mode", {}).get("value", "")
                    af = c.get("award", {}).get("value", [])
                    award = ", ".join(af) if isinstance(af, list) else af
            except Exception:
                pass

            paper_title      = sub.content.get('title', {}).get('value', 'Untitled')
            paper_type       = sub.content.get('paper_type', {}).get('value', '')
            response_to_meta = sub.content.get('response_to_metareview', {}).get('value', '')

            paper_link   = sub.content.get('paper_link', {}).get('value', '')
            linked_forum = self._parse_linked_forum_id(paper_link)
            linked_note  = self._get_linked_note(linked_forum) if linked_forum else None
            replies      = self._get_linked_replies(linked_forum) if linked_forum else []

            if linked_note and not paper_type:
                paper_type = linked_note.content.get('paper_type', {}).get('value', '')

            prev_url = ""
            if linked_note and 'previous_URL' in linked_note.content:
                prev_url = linked_note.content['previous_URL'].get('value', '').strip()

            is_anonymous = self._infer_anonymity(sub.content)
            if not is_anonymous and linked_note:
                is_anonymous = self._infer_anonymity(linked_note.content)

            # Scan replies
            completed_reviews         = 0
            expected_reviews          = 0
            meta_review_score         = ""
            confidence_scores         = []
            soundness_scores          = []
            excitement_scores         = []
            overall_assessment_scores = []
            ethics_flag               = ""
            preprint_flag             = ""
            has_review_issue          = False
            review_issue_link         = ""
            review_issue_count        = 0
            has_meta_review_issue     = False
            meta_review_issue_link    = ""
            has_confidential          = False
            has_emergency_declaration = False
            emergency_declaration_link = ""
            emergency_declaration_count = 0
            reviewer_ethics           = []
            preprinters               = []

            for reply in replies:
                invs = getattr(reply, 'invitations', [])

                if any("/-/Review_Issue_Report" in i for i in invs):
                    has_review_issue = True
                    review_issue_count += 1
                    if not review_issue_link:
                        review_issue_link = (
                            f"https://openreview.net/forum?id={getattr(reply,'forum','')}"
                            f"&noteId={getattr(reply,'id','')}"
                        )

                EMERGENCY_DECL_PATTERNS = [
                    "/-/Emergency_Review_Request",
                    "/-/Emergency_Reviewer_Recruitment",
                    "/-/Emergency_Reviewer_Request",
                    "/-/Emergency_Review",
                ]
                if any(any(p in i for p in EMERGENCY_DECL_PATTERNS) for i in invs):
                    has_emergency_declaration = True
                    emergency_declaration_count += 1
                    if not emergency_declaration_link:
                        emergency_declaration_link = (
                            f"https://openreview.net/forum?id={getattr(reply,'forum','')}"
                            f"&noteId={getattr(reply,'id','')}"
                        )

                if any("/-/Meta-Review_Issue_Report" in i for i in invs):
                    has_meta_review_issue = True
                    if not meta_review_issue_link:
                        meta_review_issue_link = (
                            f"https://openreview.net/forum?id={getattr(reply,'forum','')}"
                            f"&noteId={getattr(reply,'id','')}"
                        )

                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = getattr(reply, 'content', {})
                    for field, lst in [('confidence', confidence_scores),
                                       ('soundness', soundness_scores),
                                       ('excitement', excitement_scores),
                                       ('overall_assessment', overall_assessment_scores)]:
                        try:
                            v = content.get(field, {}).get('value')
                            if v is not None:
                                lst.append(float(v))
                        except Exception:
                            pass

                    ne = content.get('needs_ethics_review', {}).get('value', '').strip().lower()
                    ec = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if ne == "yes" or (ec and "no" not in ec):
                        if hasattr(reply, 'number'):
                            reviewer_ethics.append(str(reply.number))

                    src = content.get('Knowledge_of_paper_source', {}).get('value', [])
                    for s in (src if isinstance(src, list) else [src]):
                        if isinstance(s, str) and any(t in s.lower() for t in ("arxiv", "preprint")):
                            if hasattr(reply, 'number'):
                                preprinters.append(f"R{reply.number}")
                            break

                if self.is_meta_review(reply):
                    content = getattr(reply, 'content', {})
                    meta_review_score = (
                        content.get('overall_assessment', {}).get('value', '') or
                        content.get('overall_rating',    {}).get('value', '') or
                        content.get('score',             {}).get('value', '')
                    )
                    ec2 = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if ec2 and "no concerns" not in ec2:
                        ethics_flag = "AC: yes"

            # Expected reviews
            try:
                rg_id = f"{self.venue_id}/{self.submission_name}{sub.number}/Reviewers"
                rg = self.group_index.get(rg_id) or self.client.get_group(rg_id)
                self.group_index[rg_id] = rg
                expected_reviews = len(rg.members) if getattr(rg, 'members', None) else max(3, completed_reviews)
            except Exception:
                expected_reviews = max(3, completed_reviews)

            # Emergency reviewer group
            has_emergency_reviewer = False
            emergency_reviewer_count = 0
            for suffix in ["/Emergency_Reviewers", "/Emergency_Reviewer", "/Emergency_Review_Assignees"]:
                erg_id = f"{self.venue_id}/{self.submission_name}{sub.number}{suffix}"
                erg = self.group_index.get(erg_id)
                if erg is None:
                    try:
                        erg = self.client.get_group(erg_id)
                        self.group_index[erg_id] = erg
                    except Exception:
                        erg = None
                if erg and getattr(erg, 'members', None):
                    has_emergency_reviewer = True
                    emergency_reviewer_count = len(erg.members)
                    break

            if reviewer_ethics:
                suffix = "Reviewer: " + ", ".join(reviewer_ethics)
                ethics_flag = suffix if not ethics_flag else f"{ethics_flag}; {suffix}"
            if preprinters:
                preprint_flag = "Yes (" + ", ".join(preprinters) + ")"
            if not is_anonymous:
                is_anonymous = "No" if preprint_flag else "Yes"

            has_low_confidence = any(s <= self.LOW_CONF_THRESHOLD for s in confidence_scores)
            low_conf_reviewers = ", ".join(str(round(s)) for s in confidence_scores if s <= self.LOW_CONF_THRESHOLD)

            fmt = self.format_scores_as_avg_std
            paper_data = {
                "Paper #":               sub.number,
                "Paper ID":              sub.id,
                "Title":                 paper_title,
                "Submission Link":       f"{base_url}{sub.id}",
                "Linked Forum":          f"{base_url}{linked_forum}" if linked_forum else "",
                "Paper Type":            paper_type,
                "Completed Reviews":     completed_reviews,
                "Expected Reviews":      expected_reviews,
                "Resubmission":          prev_url,
                "Response to Meta-review": response_to_meta,
                "Is Anonymous":          is_anonymous,
                "Has Review Issue":      has_review_issue,
                "Review Issue Link":     review_issue_link,
                "Has Meta Review Issue": has_meta_review_issue,
                "Meta Review Issue Link": meta_review_issue_link,
                "Has Confidential":      "✓" if has_confidential else "",
                "Has Low Confidence":    has_low_confidence,
                "Low Confidence Reviewers": low_conf_reviewers,
                "Has Emergency Declaration": has_emergency_declaration,
                "Emergency Declaration Link": emergency_declaration_link,
                "Emergency Declaration Count": emergency_declaration_count,
                "Has Emergency Reviewer": has_emergency_reviewer,
                "Emergency Reviewer Count": emergency_reviewer_count,
                "Review Issue Count":    review_issue_count,
                "Ethics Flag":           ethics_flag,
                "Preprint Flag":         preprint_flag,
                "Reviewer Confidence":   fmt(confidence_scores),
                "Confidence List":       " / ".join(f"{s:.1f}" for s in confidence_scores),
                "Soundness Score":       fmt(soundness_scores),
                "Soundness List":        " / ".join(f"{s:.1f}" for s in soundness_scores),
                "Excitement Score":      fmt(excitement_scores),
                "Excitement List":       " / ".join(f"{s:.1f}" for s in excitement_scores),
                "Overall Assessment":    fmt(overall_assessment_scores),
                "Overall List":          " / ".join(f"{s:.1f}" for s in overall_assessment_scores),
                "Meta Review Score":     meta_review_score,
                "Recommendation":        recommendation,
                "Presentation Mode":     presentation_mode,
                "Award":                 award,
            }
            self.papers_data.append(paper_data)

            if overall_assessment_scores:
                self.score_distributions['overall_assessment'].append(
                    sum(overall_assessment_scores) / len(overall_assessment_scores)
                )
            if meta_review_score:
                try:
                    self.score_distributions['meta_review'].append(float(meta_review_score))
                except Exception:
                    pass

        print(f"Done! Processed {len(self.papers_data)} papers.")

    def compute_ac_meta_data(self):
        self.ac_meta_data = []

    def process_comments_data(self):
        if not self.papers_data:
            return
        paper_numbers = {p["Paper #"] for p in self.papers_data}
        base_url = "https://openreview.net/forum"
        count = 0
        print("Processing comments...")
        for sub in tqdm(self.submissions, desc="Checking for comments"):
            if sub.number not in paper_numbers:
                continue
            for reply in (getattr(sub, 'details', None) or {}).get('replies', []):
                if self.is_relevant_comment(reply):
                    self.comments_data.append({
                        "Paper #":  sub.number, "Paper ID": sub.id,
                        "Type":     self.classify_comment_type(reply),
                        "Role":     self.infer_role_from_signature(getattr(reply, 'signatures', [])),
                        "Date":     self.format_timestamp(getattr(reply, 'tcdate', None)),
                        "Content":  self.extract_comment_text(reply),
                        "Link":     f"{base_url}?id={getattr(reply,'forum','')}&noteId={getattr(reply,'id','')}",
                        "ReplyTo":  getattr(reply, 'replyto', None),
                        "NoteId":   getattr(reply, 'id', ''),
                    })
                    count += 1
            paper_link   = sub.content.get('paper_link', {}).get('value', '')
            linked_forum = self._parse_linked_forum_id(paper_link)
            if linked_forum:
                for reply in self._get_linked_replies(linked_forum):
                    if self.is_relevant_comment(reply):
                        self.comments_data.append({
                            "Paper #":  sub.number, "Paper ID": sub.id,
                            "Type":     self.classify_comment_type(reply),
                            "Role":     self.infer_role_from_signature(getattr(reply, 'signatures', [])),
                            "Date":     self.format_timestamp(getattr(reply, 'tcdate', None)),
                            "Content":  self.extract_comment_text(reply),
                            "Link":     f"{base_url}?id={getattr(reply,'forum','')}&noteId={getattr(reply,'id','')}",
                            "ReplyTo":  getattr(reply, 'replyto', None),
                            "NoteId":   getattr(reply, 'id', ''),
                        })
                        count += 1
        print(f"Found {count} comments.")

    def process_data(self):
        try:
            self.process_papers_data()
            if self.papers_data:
                self.compute_correlation_data()
            self.process_comments_data()
        except Exception as e:
            print(f"Error in process_data: {e}")
            import traceback; traceback.print_exc()

    def generate_histogram_data(self):
        bins_overall = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                        4.0, 4.25, 4.5, 4.75, 5.0, 5.25]
        bins_meta    = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        hd = {
            'overall_assessment': {'counts': [], 'labels': [f"{x:.2f}" for x in bins_overall[:-1]]},
            'meta_review':        {'counts': [], 'labels': [f"{x:.2f}" for x in bins_meta[:-1]]},
        }
        if self.score_distributions['overall_assessment']:
            hist, _ = np.histogram(self.score_distributions['overall_assessment'], bins=bins_overall)
            hd['overall_assessment']['counts'] = hist.tolist()
        if self.score_distributions['meta_review']:
            hist, _ = np.histogram(self.score_distributions['meta_review'], bins=bins_meta)
            hd['meta_review']['counts'] = hist.tolist()
        return hd

    def generate_report(self, output_dir=".", filename="commitment_report.html"):
        os.makedirs(output_dir, exist_ok=True)

        if not self.submissions:
            return self._write_error_report(output_dir, filename, "No papers found",
                f"No papers found for role={self.role}, user={self.me}, venue={self.venue_id}.<br>"
                f"SAC role: ensure you appear in a Senior_Area_Chairs group.<br>"
                f"AC role: ensure you appear in an Area_Chairs group.")

        self.process_data()

        if not self.papers_data:
            return self._write_error_report(output_dir, filename, "No paper data generated",
                f"Found {len(self.submissions)} submissions but none could be processed.<br>"
                f"Papers may be withdrawn or linked ARR submissions may be inaccessible.")

        template_data = {
            "title":                   f"Commitment Phase Report: {self.venue_id}",
            "generated_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":                self.venue_id,
            "role":                    self.role,
            "papers":                  self.papers_data,
            "comments_count":          len(self.comments_data),
            "comments":                self.comments_data,
            "comment_trees":           self.organize_comments_by_paper(),
            "histogram_data":          self.generate_histogram_data(),
            "correlation_data":        self.correlation_data,
            "paper_type_distribution": self.generate_paper_type_distribution(),
            "score_scatter_data":      self.generate_score_scatter_data(),
        }

        template_dir = self._resolve_template_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        html = env.get_template("commitment_report.html").render(**template_data)
        output_path = Path(output_dir) / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _resolve_template_dir(self):
        for p in [Path(__file__).parent / "templates", Path("templates")]:
            if p.exists() and (p / "commitment_report.html").exists():
                return p
        raise FileNotFoundError("Cannot find templates/ directory with commitment_report.html")

    def _write_error_report(self, output_dir, filename, title, message):
        html = (f'<!DOCTYPE html><html><head><title>Error</title>'
                f'<style>body{{font-family:sans-serif;margin:2em}}'
                f'.e{{background:#fef2f2;border:1px solid #fca5a5;padding:1.5em;border-radius:8px}}'
                f'h1{{color:#991b1b}}</style></head><body>'
                f'<h1>⚠ {title}</h1>'
                f'<div class="e"><p>{message}</p>'
                f'<p><b>ID:</b> {self.me} &nbsp; <b>Venue:</b> {self.venue_id} &nbsp; <b>Role:</b> {self.role}</p>'
                f'</div></body></html>')
        output_path = Path(output_dir) / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path
