"""
Commitment Phase Report Generator for ACL Conferences

Extends ARRReportGenerator with commitment-specific logic:
- Fetches papers by checking per-paper AC group membership (user is AC, not SAC)
- Pre-fetches and caches all venue groups for speed (_build_group_index)
- Caches linked ARR notes to avoid repeated API calls
- Adds new data fields: Is Anonymous, Has Review Issue, Has Meta Review Issue,
  Has Low Confidence
"""

import argparse
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
from urllib.parse import urlparse, parse_qs
from arr_report_generator import ARRReportGenerator


class CommitmentReportGenerator(ARRReportGenerator):

    def __init__(self, username, password, venue_id, me):
        # ---- Direct init (do NOT call super().__init__ — different flow) ----
        self.username   = username
        self.password   = password
        self.venue_id   = venue_id
        self.me         = me
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=username,
            password=password
        )

        self.venue_group      = self.client.get_group(venue_id)
        self.submission_name  = self.venue_group.content['submission_name']['value']

        # Data containers
        self.papers_data        = []
        self.comments_data      = []
        self.ac_meta_data       = []
        self.correlation_data   = None
        self.score_distributions = {'overall_assessment': [], 'meta_review': []}

        # Caches
        self.ac_email_cache  = {}
        self.profile_cache   = {}
        self.linked_note_cache = {}   # forum_id -> Note object
        self.linked_replies_cache = {}  # forum_id -> list[Note]

        # Group index: bulk pre-fetch for speed
        self.group_index = {}
        self._build_group_index()

        # Find AC groups + assigned papers
        print(f"Checking groups for {me} in {venue_id}...")
        try:
            self.my_ac_groups = []
            try:
                ac_group = self.client.get_group(f"{venue_id}/Area_Chairs")
                if me in ac_group.members:
                    self.my_ac_groups.append(ac_group.id)
                    print("Found you in the main Area Chairs group")
            except:
                pass
            my_groups = self.client.get_all_groups(member=me)
            ac_related = [g for g in my_groups if venue_id in g.id and "Area_Chair" in g.id]
            self.my_ac_groups.extend([g.id for g in ac_related])
            print(f"You are a member of {len(self.my_ac_groups)} AC-related groups")
        except Exception as e:
            print(f"Error checking AC groups: {e}")
            self.my_ac_groups = []

        print(f"Retrieving submissions for {venue_id}...")
        try:
            all_submissions = self.client.get_all_notes(
                invitation=f'{venue_id}/-/{self.submission_name}',
                details='replies'
            )
            print(f"Retrieved {len(all_submissions)} total submissions")

            self.submissions = []
            for submission in tqdm(all_submissions, desc="Finding your assigned papers"):
                # Check AC assignment via cached group index first
                ac_group_id = f"{venue_id}/{self.submission_name}{submission.number}/Area_Chairs"
                ac_group = self.group_index.get(ac_group_id)
                if ac_group is None:
                    try:
                        ac_group = self.client.get_group(ac_group_id)
                        self.group_index[ac_group_id] = ac_group
                    except:
                        ac_group = None

                is_assigned = ac_group is not None and hasattr(ac_group, 'members') and me in ac_group.members

                # Only papers that have a paper_link (i.e. an ARR submission linked)
                link = submission.content.get("paper_link", {}).get("value", "")
                if is_assigned and link.strip():
                    print(f"✓ Paper {submission.number}. Link: {link}")
                    self.submissions.append(submission)

            print(f"Found {len(self.submissions)} papers assigned to you")
        except Exception as e:
            print(f"Error retrieving submissions: {e}")
            self.submissions = []

    def _build_group_index(self):
        """Pre-fetch all groups under venue/submission prefix into a dict."""
        prefix = f'{self.venue_id}/{self.submission_name}'
        try:
            print(f"Pre-fetching groups with prefix: {prefix}")
            groups = self.client.get_all_groups(prefix=prefix)
            self.group_index = {g.id: g for g in groups}
            print(f"Cached {len(self.group_index)} groups for fast lookup")
        except Exception as e:
            print(f"Warning: could not pre-fetch groups ({e}). Will fall back to per-group calls.")
            self.group_index = {}

    # -------------------------------------------------------------------------
    # Override predicates for Note objects (not dicts)
    # -------------------------------------------------------------------------

    def is_actual_review(self, reply):
        try:
            invs = reply.invitations if hasattr(reply, 'invitations') else reply.get('invitations', [])
            return any('/-/Official_Review' in inv for inv in invs)
        except Exception:
            return False

    def is_meta_review(self, reply):
        try:
            invs = reply.invitations if hasattr(reply, 'invitations') else reply.get('invitations', [])
            return any('/-/Meta_Review' in inv for inv in invs)
        except Exception:
            return False

    def is_relevant_comment(self, reply):
        try:
            invs = reply.invitations if hasattr(reply, 'invitations') else reply.get('invitations', [])
            return any(
                part in inv
                for inv in invs
                for part in ["/-/Author-Editor_Confidential_Comment", "/-/Comment",
                             "/-/Review_Issue_Report", "/-/Meta-Review_Issue_Report"]
            )
        except Exception:
            return False

    def classify_comment_type(self, reply):
        try:
            invs = reply.invitations if hasattr(reply, 'invitations') else reply.get('invitations', [])
            if any("/-/Review_Issue_Report" in inv for inv in invs):
                return "Review Issue"
            elif any("/-/Meta-Review_Issue_Report" in inv for inv in invs):
                return "Meta Review Issue"
            elif any("/-/Author-Editor_Confidential_Comment" in inv for inv in invs):
                return "Author-Editor Confidential"
            elif any("/-/Comment" in inv for inv in invs):
                return "Confidential Comment"
            else:
                return "Other"
        except Exception:
            return "Comment"

    def extract_comment_text(self, reply):
        try:
            content = reply.content if hasattr(reply, 'content') else reply.get("content", {})
            for key in ["comment", "justification", "text", "response", "value"]:
                if key in content:
                    val = content[key]
                    if isinstance(val, dict) and "value" in val:
                        return val["value"]
                    return val
            fallback = []
            for k, v in content.items():
                if isinstance(v, dict) and "value" in v:
                    fallback.append(f"{k}: {v['value']}")
            return "\n".join(fallback) if fallback else "(No comment text found)"
        except Exception as e:
            return f"(Error extracting comment text: {e})"

    def infer_role_from_signature(self, signatures):
        if not signatures:
            return "Unknown"
        try:
            sig = signatures[0]
            if "/Authors" in sig:           return "Author"
            elif "/Reviewer" in sig:        return "Reviewer"
            elif "/Area_Chair" in sig:      return "Area Chair"
            elif "/Senior_Area_Chairs" in sig: return "Senior Area Chair"
            elif "/Program_Chairs" in sig:  return "Program Chair"
            elif sig.startswith("~"):       return "User"
            else:                           return "Other"
        except Exception:
            return "Unknown"

    # -------------------------------------------------------------------------
    # Linked-forum helpers with caching
    # -------------------------------------------------------------------------

    def _parse_linked_forum_id(self, paper_link):
        """Extract the OpenReview forum ID from a paper_link URL."""
        if not paper_link:
            return None
        try:
            parsed = urlparse(paper_link)
            qs = parse_qs(parsed.query)
            return qs.get("id", qs.get("noteId", [None]))[0]
        except Exception:
            return None

    def _get_linked_note(self, forum_id):
        """Fetch (and cache) the root note for a linked ARR forum."""
        if forum_id in self.linked_note_cache:
            return self.linked_note_cache[forum_id]
        try:
            note = self.client.get_note(id=forum_id)
            self.linked_note_cache[forum_id] = note
            return note
        except Exception as e:
            print(f"  Warning: could not fetch linked note {forum_id}: {e}")
            self.linked_note_cache[forum_id] = None
            return None

    def _get_linked_replies(self, forum_id):
        """Fetch (and cache) all replies for a linked ARR forum."""
        if forum_id in self.linked_replies_cache:
            return self.linked_replies_cache[forum_id]
        try:
            replies = self.client.get_notes(forum=forum_id)
            self.linked_replies_cache[forum_id] = replies
            return replies
        except Exception as e:
            print(f"  Warning: could not fetch replies for forum {forum_id}: {e}")
            self.linked_replies_cache[forum_id] = []
            return []

    # -------------------------------------------------------------------------
    # Data processing
    # -------------------------------------------------------------------------

    def process_papers_data(self):
        """Process papers assigned to this AC in the commitment venue."""
        if not self.submissions:
            print("No submissions to process.")
            return

        print(f"Starting to process {len(self.submissions)} papers...")
        base_url = "https://openreview.net/forum?id="

        for submission in tqdm(self.submissions, desc="Processing your assigned papers"):
            # Skip withdrawn / desk-rejected
            try:
                if hasattr(submission, 'content'):
                    if submission.content.get('withdrawal_confirmation', {}).get('value', '').strip():
                        print(f"Skipping withdrawn paper {submission.number}")
                        continue
                    venue_val = submission.content.get('venue', {}).get('value', '').lower()
                    if "withdrawn" in venue_val or "desk rejected" in venue_val:
                        print(f"Skipping desk-rejected paper {submission.number}")
                        continue
            except Exception:
                pass

            # --- SAC meta-review (on the commitment submission) ---
            base_forum_prefix = f"{self.venue_id}/{self.submission_name}"
            recommendation    = ""
            presentation_mode = ""
            award             = ""
            try:
                metas = self.client.get_notes(
                    invitation=f"{base_forum_prefix}{submission.number}/-/Meta_Review"
                )
                if metas:
                    c = metas[0].content
                    recommendation    = c.get("recommendation",    {}).get("value", "")
                    presentation_mode = c.get("presentation_mode", {}).get("value", "")
                    award_field       = c.get("award",             {}).get("value", [])
                    award = ", ".join(award_field) if isinstance(award_field, list) else award_field
            except Exception as e:
                print(f"  Error fetching SAC meta-review for #{submission.number}: {e}")

            paper_title = submission.content.get('title', {}).get('value', 'Untitled')
            paper_type  = submission.content.get('paper_type', {}).get('value', '')
            response_to_meta = submission.content.get('response_to_metareview', {}).get('value', '')

            # --- Resolve linked ARR forum ---
            paper_link = submission.content.get('paper_link', {}).get('value', '')
            linked_forum = self._parse_linked_forum_id(paper_link)
            linked_note  = self._get_linked_note(linked_forum) if linked_forum else None
            replies      = self._get_linked_replies(linked_forum) if linked_forum else []

            # Fill paper_type from linked note if missing
            if linked_note and not paper_type:
                paper_type = linked_note.content.get('paper_type', {}).get('value', '')

            # Resubmission URL from linked note
            prev_url = ""
            if linked_note and 'previous_URL' in linked_note.content:
                prev_url = linked_note.content['previous_URL'].get('value', '').strip()

            # --- Anonymity ---
            # Debug on first paper so we can identify the real field name
            if not self.papers_data:
                for src_label, src_content in [("commitment", submission.content),
                                                ("linked ARR", linked_note.content if linked_note else {})]:
                    anon_rel = {k: v for k, v in src_content.items()
                                if any(t in k.lower() for t in ("anon", "preprint", "non_anon", "nonanon"))}
                    if anon_rel:
                        print(f"[DEBUG] {src_label} anon/preprint fields: {list(anon_rel.keys())}")
                        for k, v in anon_rel.items():
                            print(f"  {k!r}: {v!r}")
                    else:
                        print(f"[DEBUG] {src_label} — no anon/preprint fields. Keys: {sorted(src_content.keys())}")
            # Check commitment submission first, then linked ARR note
            is_anonymous = self._infer_anonymity(submission.content)
            if not is_anonymous and linked_note:
                is_anonymous = self._infer_anonymity(linked_note.content)

            # --- Scan replies ---
            completed_reviews  = 0
            expected_reviews   = 0
            meta_review_score  = ""
            confidence_scores  = []
            soundness_scores   = []
            excitement_scores  = []
            overall_assessment_scores = []
            ethics_flag        = ""
            preprint_flag      = ""
            has_review_issue        = False
            review_issue_link       = ""
            has_meta_review_issue   = False
            meta_review_issue_link  = ""
            has_confidential        = False
            reviewer_ethics  = []
            preprinters      = []

            for reply in replies:
                invs = reply.invitations if hasattr(reply, 'invitations') else []

                # Review Issue
                if any("/-/Review_Issue_Report" in inv for inv in invs):
                    has_review_issue = True
                    if not review_issue_link:
                        fid = reply.forum if hasattr(reply, 'forum') else ""
                        nid = reply.id    if hasattr(reply, 'id')    else ""
                        review_issue_link = f"https://openreview.net/forum?id={fid}&noteId={nid}"

                # Meta-Review Issue
                if any("/-/Meta-Review_Issue_Report" in inv for inv in invs):
                    has_meta_review_issue = True
                    if not meta_review_issue_link:
                        fid = reply.forum if hasattr(reply, 'forum') else ""
                        nid = reply.id    if hasattr(reply, 'id')    else ""
                        meta_review_issue_link = f"https://openreview.net/forum?id={fid}&noteId={nid}"

                # Confidential comment
                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = reply.content if hasattr(reply, 'content') else {}
                    try:
                        val = content.get('confidence', {}).get('value')
                        if val is not None:
                            confidence_scores.append(float(val))
                    except: pass
                    try:
                        val = content.get('soundness', {}).get('value')
                        if val is not None:
                            soundness_scores.append(float(val))
                    except: pass
                    try:
                        val = content.get('excitement', {}).get('value')
                        if val is not None:
                            excitement_scores.append(float(val))
                    except: pass
                    try:
                        val = content.get('overall_assessment', {}).get('value')
                        if val is not None:
                            overall_assessment_scores.append(float(val))
                    except: pass

                    # Ethics check
                    needs_ethics = content.get('needs_ethics_review', {}).get('value', '').strip().lower()
                    ethical_concerns = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if needs_ethics == "yes" or (ethical_concerns and "no" not in ethical_concerns):
                        if hasattr(reply, 'number'):
                            reviewer_ethics.append(str(reply.number))

                    # Preprint knowledge
                    src = content.get('Knowledge_of_paper_source', {}).get('value', [])
                    sources = src if isinstance(src, list) else [src]
                    for s in sources:
                        if isinstance(s, str) and any(tok in s.lower() for tok in ("arxiv", "preprint")):
                            if hasattr(reply, 'number'):
                                preprinters.append(f"R{reply.number}")
                            break

                if self.is_meta_review(reply):
                    content = reply.content if hasattr(reply, 'content') else {}
                    meta_review_score = (
                        content.get('overall_assessment', {}).get('value', '') or
                        content.get('overall_rating',    {}).get('value', '') or
                        content.get('score',             {}).get('value', '')
                    )
                    # Meta ethics
                    if 'ethical_concerns' in content:
                        ec = content['ethical_concerns'].get('value', '').strip().lower()
                        if ec and "no concerns" not in ec:
                            ethics_flag = "AC: yes"

            # Expected reviews
            if replies:
                try:
                    rg_id = f"{self.venue_id}/{self.submission_name}{submission.number}/Reviewers"
                    rg = self.group_index.get(rg_id)
                    if rg is None:
                        rg = self.client.get_group(rg_id)
                        self.group_index[rg_id] = rg
                    expected_reviews = len(rg.members) if rg and hasattr(rg, 'members') else max(3, completed_reviews)
                except:
                    expected_reviews = max(3, completed_reviews)

            # Assemble flag strings
            if reviewer_ethics:
                suffix = "Reviewer: " + ", ".join(reviewer_ethics)
                ethics_flag = suffix if not ethics_flag else f"{ethics_flag}; {suffix}"
            if preprinters:
                preprint_flag = "Yes (" + ", ".join(preprinters) + ")"

            # Fallback: if anonymity not determined from metadata, use reviewer-detected preprint
            # preprint_flag non-empty  → reviewers noticed a preprint → NOT anonymous
            # preprint_flag empty      → no preprint detected         → probably anonymous
            if not is_anonymous:
                is_anonymous = "No" if preprint_flag else "Yes"

            # Low confidence
            has_low_confidence   = any(s <= self.LOW_CONF_THRESHOLD for s in confidence_scores)
            low_conf_reviewers   = ", ".join(str(round(s)) for s in confidence_scores if s <= self.LOW_CONF_THRESHOLD)

            # Format scores
            reviewer_confidence = self.format_scores_as_avg_std(confidence_scores)
            reviewer_soundness  = self.format_scores_as_avg_std(soundness_scores)
            reviewer_excitement = self.format_scores_as_avg_std(excitement_scores)
            reviewer_overall    = self.format_scores_as_avg_std(overall_assessment_scores)
            confidence_list = " / ".join(f"{s:.1f}" for s in confidence_scores) if confidence_scores else ""
            soundness_list  = " / ".join(f"{s:.1f}" for s in soundness_scores)  if soundness_scores  else ""
            excitement_list = " / ".join(f"{s:.1f}" for s in excitement_scores) if excitement_scores else ""
            overall_list    = " / ".join(f"{s:.1f}" for s in overall_assessment_scores) if overall_assessment_scores else ""

            paper_data = {
                "Paper #":           submission.number,
                "Paper ID":          submission.id,
                "Title":             paper_title,
                "Submission Link":   f"{base_url}{submission.id}",
                "Linked Forum":      f"{base_url}{linked_forum}" if linked_forum else "",
                "Paper Type":        paper_type,
                "Completed Reviews": completed_reviews,
                "Expected Reviews":  expected_reviews,
                "Resubmission":      prev_url,
                "Response to Meta-review": response_to_meta,
                # Flags
                "Is Anonymous":          is_anonymous,
                "Has Review Issue":      has_review_issue,
                "Review Issue Link":     review_issue_link,
                "Has Meta Review Issue": has_meta_review_issue,
                "Meta Review Issue Link": meta_review_issue_link,
                "Has Confidential":      "✓" if has_confidential else "",
                "Has Low Confidence":    has_low_confidence,
                "Low Confidence Reviewers": low_conf_reviewers,
                "Ethics Flag":           ethics_flag,
                "Preprint Flag":         preprint_flag,
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
                # SAC decision
                "Recommendation":    recommendation,
                "Presentation Mode": presentation_mode,
                "Award":             award,
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
        print(f'Total processed papers: {len(self.papers_data)}')

    def compute_ac_meta_data(self):
        """Commitment phase doesn't expose AC identities."""
        self.ac_meta_data = []

    def process_comments_data(self):
        """Collect comments from both the commitment submission and its linked ARR forum."""
        if not self.papers_data:
            return
        paper_numbers = {p["Paper #"] for p in self.papers_data}
        base_url = "https://openreview.net/forum"
        processed_count = 0

        print("Processing comments...")
        for submission in tqdm(self.submissions, desc="Checking for comments"):
            if submission.number not in paper_numbers:
                continue

            # Comments on the commitment submission itself
            for reply in (submission.details or {}).get('replies', []):
                if self.is_relevant_comment(reply):
                    forum_id   = reply.forum if hasattr(reply, 'forum') else ""
                    note_id    = reply.id    if hasattr(reply, 'id')    else ""
                    replyto    = reply.replyto if hasattr(reply, 'replyto') else None
                    signatures = reply.signatures if hasattr(reply, 'signatures') else []
                    tcdate     = reply.tcdate if hasattr(reply, 'tcdate') else None
                    self.comments_data.append({
                        "Paper #":  submission.number,
                        "Paper ID": submission.id,
                        "Type":     self.classify_comment_type(reply),
                        "Role":     self.infer_role_from_signature(signatures),
                        "Date":     self.format_timestamp(tcdate),
                        "Content":  self.extract_comment_text(reply),
                        "Link":     f"{base_url}?id={forum_id}&noteId={note_id}",
                        "ReplyTo":  replyto,
                        "NoteId":   note_id,
                    })
                    processed_count += 1

            # Comments on the linked ARR forum (use cached replies)
            paper_link   = submission.content.get('paper_link', {}).get('value', '')
            linked_forum = self._parse_linked_forum_id(paper_link)
            if linked_forum:
                for reply in self._get_linked_replies(linked_forum):
                    if self.is_relevant_comment(reply):
                        forum_id   = reply.forum if hasattr(reply, 'forum') else ""
                        note_id    = reply.id    if hasattr(reply, 'id')    else ""
                        replyto    = reply.replyto if hasattr(reply, 'replyto') else None
                        signatures = reply.signatures if hasattr(reply, 'signatures') else []
                        tcdate     = reply.tcdate if hasattr(reply, 'tcdate') else None
                        self.comments_data.append({
                            "Paper #":  submission.number,
                            "Paper ID": submission.id,
                            "Type":     self.classify_comment_type(reply),
                            "Role":     self.infer_role_from_signature(signatures),
                            "Date":     self.format_timestamp(tcdate),
                            "Content":  self.extract_comment_text(reply),
                            "Link":     f"{base_url}?id={forum_id}&noteId={note_id}",
                            "ReplyTo":  replyto,
                            "NoteId":   note_id,
                        })
                        processed_count += 1

        print(f"Found {processed_count} comments across {len(self.papers_data)} papers.")

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
        try:
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
        except Exception as e:
            print(f"Error generating histogram data: {e}")
            return {'overall_assessment': {'counts': [], 'labels': []}, 'meta_review': {'counts': [], 'labels': []}}

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def generate_report(self, output_dir="."):
        os.makedirs(output_dir, exist_ok=True)

        if not self.submissions:
            return self._write_error_report(output_dir, "No papers found",
                f"No papers assigned to you ({self.me}) in {self.venue_id}.")

        self.process_data()

        if not self.papers_data:
            return self._write_error_report(output_dir, "No paper data generated",
                f"Found {len(self.submissions)} submissions but no paper data could be processed.")

        paper_type_distribution = self.generate_paper_type_distribution()
        score_scatter_data      = self.generate_score_scatter_data()

        template_data = {
            "title":                 f"Commitment Phase Report: {self.venue_id}",
            "generated_date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":              self.venue_id,
            "papers":                self.papers_data,
            "comments_count":        len(self.comments_data),
            "comments":              self.comments_data,
            "comment_trees":         self.organize_comments_by_paper(),
            "histogram_data":        self.generate_histogram_data(),
            "correlation_data":      self.correlation_data,
            "paper_type_distribution": paper_type_distribution,
            "score_scatter_data":    score_scatter_data,
        }

        template_dir = self._resolve_template_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        template     = env.get_template("commitment_report.html")
        html_content = template.render(**template_data)

        output_path = Path(output_dir) / "commitment_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    def _resolve_template_dir(self):
        """Find templates/ alongside this file or in CWD."""
        candidates = [
            Path(__file__).parent / "templates",
            Path("templates"),
        ]
        for p in candidates:
            if p.exists() and (p / "commitment_report.html").exists():
                return p
        raise FileNotFoundError(
            "Could not find 'templates/' with commitment_report.html. "
            "Make sure the templates directory is alongside arr_commitment_generator.py "
            "or in your current working directory."
        )

    def _write_error_report(self, output_dir, title, message):
        html = f"""<!DOCTYPE html>
<html><head><title>Error - {title}</title>
<style>body{{font-family:sans-serif;margin:2em}}.error{{color:red;padding:1em;border:1px solid #ccc}}</style>
</head><body>
<h1>{title}</h1>
<div class="error"><p>{message}</p>
<p>OpenReview ID: {self.me}</p><p>Venue: {self.venue_id}</p></div>
</body></html>"""
        output_path = Path(output_dir) / "commitment_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate Commitment Phase Report')
    parser.add_argument('--username', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--venue_id', required=True)
    parser.add_argument('--me', required=True)
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()
    generator = CommitmentReportGenerator(
        username=args.username, password=args.password,
        venue_id=args.venue_id, me=args.me
    )
    report_path = generator.generate_report(output_dir=args.output_dir)
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
