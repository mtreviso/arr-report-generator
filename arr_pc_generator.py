"""
Program Chair (PC) Report Generator for ARR Venues.

Fetches ALL active submissions for a venue (no SAC filtering),
computes SAC-level and track-level aggregations, and generates
a rich HTML dashboard designed for PC-scale data (thousands of papers).

Usage:
  python generate_pc_report.py --username ... --password ... --venue_id ... --me ...
"""

import openreview
import numpy as np
import pandas as pd
import os
import jinja2
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from arr_report_generator import ARRReportGenerator


class PCReportGenerator(ARRReportGenerator):
    """
    Extends ARRReportGenerator in PC mode:
    - No SAC-batch filtering: fetches all papers
    - Adds SAC-level aggregation (sac_meta_data)
    - Adds track-level aggregation (track_data)
    - Adds attention queue (papers needing PC action)
    """

    def __init__(self, username, password, venue_id, me):
        # Call base __init__ with role='pc' so it skips the SAC filter
        super().__init__(username=username, password=password,
                         venue_id=venue_id, me=me, role='pc')
        # Extra data containers
        self.sac_meta_data    = []
        self.track_data       = []
        self.attention_papers = []
        self.reviewer_load    = {}  # reviewer_id -> paper count

    # -----------------------------------------------------------------------
    # Override process_papers_data to also capture SAC name per paper
    # -----------------------------------------------------------------------

    def process_papers_data(self):
        """Process all papers, adding SAC column (PC can see SAC assignments)."""
        base_url = "https://openreview.net/forum?id="

        for submission in tqdm(self.submissions, desc="Processing papers"):
            if self.is_withdrawn(submission):
                continue
            try:
                if "venue" in submission.content and \
                   "desk rejected" in submission.content["venue"]["value"].lower():
                    continue
            except Exception:
                pass

            prefix = f'{self.venue_id}/{self.submission_name}{submission.number}'

            # Area Chair
            ac_group = self.group_index.get(f'{prefix}/Area_Chairs')
            if ac_group is None:
                try:
                    ac_group = self.client.get_group(f'{prefix}/Area_Chairs')
                    self.group_index[f'{prefix}/Area_Chairs'] = ac_group
                except Exception:
                    ac_group = None
            ac_entry  = (ac_group.members[0] if ac_group and getattr(ac_group,'members',None) else "")
            ac_uid    = self._resolve_ac_user_id(ac_entry)
            ac_name   = self.get_display_name_for_user(ac_uid) if ac_uid else ""
            ac_email  = self.get_email_for_user(ac_uid) if ac_uid else ""

            # Senior Area Chair
            sac_group = self.group_index.get(f'{prefix}/Senior_Area_Chairs')
            if sac_group is None:
                try:
                    sac_group = self.client.get_group(f'{prefix}/Senior_Area_Chairs')
                    self.group_index[f'{prefix}/Senior_Area_Chairs'] = sac_group
                except Exception:
                    sac_group = None
            sac_entry = (sac_group.members[0] if sac_group and getattr(sac_group,'members',None) else "")
            sac_uid   = self._resolve_ac_user_id(sac_entry)
            sac_name  = self.get_display_name_for_user(sac_uid) if sac_uid else ""
            sac_email = self.get_email_for_user(sac_uid) if sac_uid else ""

            paper_type = submission.content.get("paper_type", {}).get("value", "")
            # Try multiple field names ARR uses across rounds
            track = ""
            for _tf in ("primary_area", "track", "area", "subject_area",
                        "research_area", "track_name", "submission_track"):
                _tv = submission.content.get(_tf, {})
                if isinstance(_tv, dict):
                    _tv = _tv.get("value", "")
                if _tv and str(_tv).strip():
                    track = str(_tv).strip()
                    break

            # Reply scan
            completed_reviews         = 0
            meta_review_score         = ""
            confidence_scores         = []
            soundness_scores          = []
            excitement_scores         = []
            overall_assessment_scores = []
            has_review_issue          = False
            review_issue_link         = ""
            review_issue_count        = 0
            has_confidential          = False
            has_low_confidence        = False
            ethics_flag               = ""
            has_emergency_declaration = False
            emergency_declaration_link = ""
            emergency_declaration_count = 0

            for reply in submission.details.get("replies", []):
                invitations = reply.get("invitations", [])

                if any("/-/Review_Issue_Report" in inv for inv in invitations):
                    has_review_issue = True
                    review_issue_count += 1
                    if not review_issue_link:
                        review_issue_link = (
                            f"https://openreview.net/forum?id={reply.get('forum','')}"
                            f"&noteId={reply.get('id','')}"
                        )

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
                        emergency_declaration_link = (
                            f"https://openreview.net/forum?id={reply.get('forum','')}"
                            f"&noteId={reply.get('id','')}"
                        )

                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = reply.get("content", {})
                    for field, lst in [('confidence', confidence_scores),
                                       ('soundness', soundness_scores),
                                       ('excitement', excitement_scores),
                                       ('overall_assessment', overall_assessment_scores)]:
                        try:
                            v = content.get(field, {}).get("value")
                            if v is not None:
                                lst.append(float(v))
                        except Exception:
                            pass
                    ne = content.get('needs_ethics_review', {}).get('value', '').strip().lower()
                    ec = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if ne == "yes" or (ec and "no" not in ec):
                        ethics_flag = "yes"

                if self.is_meta_review(reply):
                    content = reply.get("content", {})
                    meta_review_score = (
                        content.get("overall_assessment", {}).get("value", "") or
                        content.get("overall_rating",    {}).get("value", "") or
                        content.get("score",             {}).get("value", "")
                    )

            # Expected reviews
            expected_reviews = 0
            rg = self.group_index.get(f'{prefix}/Reviewers')
            if rg is None:
                try:
                    rg = self.client.get_group(f'{prefix}/Reviewers')
                    self.group_index[f'{prefix}/Reviewers'] = rg
                except Exception:
                    rg = None
            if rg and getattr(rg, 'members', None):
                expected_reviews = len(rg.members)
                for rev_id in rg.members:
                    self.reviewer_load[rev_id] = self.reviewer_load.get(rev_id, 0) + 1

            # Emergency reviewer group
            has_emergency_reviewer = False
            emergency_reviewer_count = 0
            for suffix in ["/Emergency_Reviewers", "/Emergency_Reviewer", "/Emergency_Review_Assignees"]:
                erg_id = f'{prefix}{suffix}'
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

            has_low_confidence = any(s <= self.LOW_CONF_THRESHOLD for s in confidence_scores)
            low_conf_reviewers = ", ".join(
                str(round(s)) for s in confidence_scores if s <= self.LOW_CONF_THRESHOLD
            )
            is_anonymous = self._infer_anonymity(submission.content)

            fmt = self.format_scores_as_avg_std
            paper_data = {
                "Paper #":               submission.number,
                "Paper ID":              submission.id,
                "Title":                 submission.content.get("title", {}).get("value", "Untitled"),
                "Paper Type":            paper_type,
                "Track":                 track,
                "Area Chair":            ac_name,
                "Area Chair ID":         ac_uid,
                "Area Chair Email":      ac_email,
                "Senior Area Chair":     sac_name,
                "Senior Area Chair ID":  sac_uid,
                "Senior Area Chair Email": sac_email,
                "Completed Reviews":     completed_reviews,
                "Expected Reviews":      expected_reviews,
                "Ready for Rebuttal":    "✓" if completed_reviews >= 3 else "",
                "Is Anonymous":          is_anonymous,
                "Has Review Issue":      has_review_issue,
                "Review Issue Link":     review_issue_link,
                "Has Confidential":      has_confidential,
                "Has Low Confidence":    has_low_confidence,
                "Low Confidence Reviewers": low_conf_reviewers,
                "Ethics Flag":           ethics_flag,
                "Has Emergency Declaration": has_emergency_declaration,
                "Emergency Declaration Link": emergency_declaration_link,
                "Emergency Declaration Count": emergency_declaration_count,
                "Has Emergency Reviewer": has_emergency_reviewer,
                "Emergency Reviewer Count": emergency_reviewer_count,
                "Review Issue Count":    review_issue_count,
                "Reviewer Confidence":   fmt(confidence_scores),
                "Confidence List":       " / ".join(f"{s:.1f}" for s in confidence_scores),
                "Soundness Score":       fmt(soundness_scores),
                "Soundness List":        " / ".join(f"{s:.1f}" for s in soundness_scores),
                "Excitement Score":      fmt(excitement_scores),
                "Excitement List":       " / ".join(f"{s:.1f}" for s in excitement_scores),
                "Overall Assessment":    fmt(overall_assessment_scores),
                "Overall List":          " / ".join(f"{s:.1f}" for s in overall_assessment_scores),
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
                except Exception:
                    pass

        print(f"Done! Processed {len(self.papers_data)} papers.")

    # -----------------------------------------------------------------------
    # SAC-level aggregation
    # -----------------------------------------------------------------------

    def compute_sac_meta_data(self):
        """Aggregate per Senior Area Chair (analogous to AC dashboard for SAC level)."""
        if not self.papers_data:
            return
        df = pd.DataFrame(self.papers_data)
        if "Senior Area Chair" not in df.columns or df["Senior Area Chair"].eq("").all():
            self.sac_meta_data = []
            return

        rows = []
        for sac_name, group in df.groupby("Senior Area Chair"):
            if not sac_name:
                continue
            num_papers       = len(group)
            reviews_done     = int(group["Completed Reviews"].sum())
            reviews_expected = int(group["Expected Reviews"].sum())
            meta_done        = int((group["Meta Review Score"] != "").sum())
            issues           = int(group["Has Review Issue"].sum())
            low_conf         = int(group["Has Low Confidence"].sum())
            ethics           = int((group["Ethics Flag"] != "").sum())
            sac_id    = group["Senior Area Chair ID"].iloc[0]
            sac_email = group["Senior Area Chair Email"].iloc[0]
            emerg_decl   = int(group["Has Emergency Declaration"].sum()) if "Has Emergency Declaration" in group.columns else 0
            emerg_assign = int(group["Has Emergency Reviewer"].sum()) if "Has Emergency Reviewer" in group.columns else 0
            emerg_unmet  = max(0, emerg_decl - emerg_assign)
            late_papers  = int((group["Completed Reviews"] < group["Expected Reviews"]).sum())
            rows.append({
                "Senior Area Chair":       sac_name,
                "Senior Area Chair ID":    sac_id,
                "Senior Area Chair Email": sac_email,
                "Num_Papers":              num_papers,
                "Completed_Reviews":       reviews_done,
                "Expected_Reviews":        reviews_expected,
                "Review_Pct":             round(100 * reviews_done / reviews_expected, 1) if reviews_expected else 0,
                "Meta_Reviews_Done":       f"{meta_done} / {num_papers}",
                "All_Meta_Done":           "✓" if meta_done == num_papers else "",
                "Review_Issues":           issues,
                "Low_Conf_Papers":         low_conf,
                "Ethics_Papers":           ethics,
                "Late_Papers":             late_papers,
                "Emergency_Declared":      emerg_decl,
                "Emergency_Unassigned":    emerg_unmet,
            })
        rows.sort(key=lambda r: r["Num_Papers"], reverse=True)
        self.sac_meta_data = rows

    # -----------------------------------------------------------------------
    # Track-level aggregation
    # -----------------------------------------------------------------------

    def compute_track_data(self):
        """Aggregate per track/area."""
        if not self.papers_data:
            return
        df = pd.DataFrame(self.papers_data)
        if "Track" not in df.columns or df["Track"].eq("").all():
            self.track_data = []
            return
        rows = []
        for track, group in df.groupby("Track"):
            if not track:
                continue
            num   = len(group)
            done  = int((group["Meta Review Score"] != "").sum())
            issues= int(group["Has Review Issue"].sum())
            avg_overall = group["Overall Assessment"].apply(self.parse_avg)
            avg_meta    = group["Meta Review Score"].apply(
                lambda x: float(x) if isinstance(x, (int,float)) or (isinstance(x,str) and x.strip()) else float('nan')
            )
            rows.append({
                "Track":        track,
                "Papers":       num,
                "Meta Done":    f"{done} / {num}",
                "Meta Pct":     round(100 * done / num, 1) if num else 0,
                "Avg Overall":  round(avg_overall.mean(), 2) if not avg_overall.isna().all() else "",
                "Avg Meta":     round(avg_meta.mean(), 2) if not avg_meta.isna().all() else "",
                "Issues":       issues,
            })
        rows.sort(key=lambda r: r["Papers"], reverse=True)
        self.track_data = rows

    # -----------------------------------------------------------------------
    # Attention queue: papers needing PC action
    # -----------------------------------------------------------------------

    def compute_attention_papers(self):
        """Papers that may need PC attention: ethics flag, review issues, unfinished reviews."""
        if not self.papers_data:
            return
        self.attention_papers = [
            p for p in self.papers_data
            if p["Ethics Flag"] or p["Has Review Issue"] or
               (p["Completed Reviews"] < p["Expected Reviews"] and p["Expected Reviews"] > 0)
        ]

    # -----------------------------------------------------------------------
    # Overview statistics
    # -----------------------------------------------------------------------

    def compute_overview_stats(self):
        """High-level counters mirroring the PC console."""
        if not self.papers_data:
            return {}
        df = pd.DataFrame(self.papers_data)
        total          = len(df)
        reviews_done   = int(df["Completed Reviews"].sum())
        reviews_exp    = int(df["Expected Reviews"].sum())
        meta_done      = int((df["Meta Review Score"] != "").sum())
        issues         = int(df["Has Review Issue"].sum())
        ethics         = int((df["Ethics Flag"] != "").sum())
        low_conf       = int(df["Has Low Confidence"].sum())
        all_reviewed   = int((df["Completed Reviews"] >= df["Expected Reviews"]).sum())
        sac_count      = df["Senior Area Chair"].nunique() if "Senior Area Chair" in df.columns else 0
        ac_count       = df["Area Chair"].nunique()

        # Score averages
        overall_vals = df["Overall Assessment"].apply(self.parse_avg).dropna()
        meta_vals    = df["Meta Review Score"].apply(
            lambda x: float(x) if x and str(x).strip() else np.nan
        ).dropna()

        # Emergency stats
        emerg_decl  = int(df["Has Emergency Declaration"].sum()) if "Has Emergency Declaration" in df.columns else 0
        emerg_unmet = int((df["Has Emergency Declaration"] & ~df["Has Emergency Reviewer"]).sum()) if "Has Emergency Declaration" in df.columns else 0

        # Paper type breakdown
        type_counts = df["Paper Type"].value_counts().to_dict() if "Paper Type" in df.columns else {}

        return {
            "total_papers":            total,
            "reviews_done":            reviews_done,
            "reviews_expected":        reviews_exp,
            "review_pct":              round(100 * reviews_done / reviews_exp, 1) if reviews_exp else 0,
            "papers_all_reviewed":     all_reviewed,
            "papers_all_reviewed_pct": round(100 * all_reviewed / total, 1) if total else 0,
            "meta_done":               meta_done,
            "meta_pct":                round(100 * meta_done / total, 1) if total else 0,
            "review_issues":           issues,
            "ethics_papers":           ethics,
            "low_conf_papers":         low_conf,
            "sac_count":               sac_count,
            "ac_count":                ac_count,
            "emergency_declared":      emerg_decl,
            "emergency_unmet":         emerg_unmet,
            "avg_overall":             round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
            "avg_meta":                round(float(meta_vals.mean()),    2) if len(meta_vals)    else None,
            "type_counts":             type_counts,
        }

    # -----------------------------------------------------------------------
    # Score outliers: AC score diverges from reviewer consensus
    # -----------------------------------------------------------------------

    def compute_score_outliers(self, threshold=0.8, top_n=15):
        """Papers where |meta_score - avg_overall| >= threshold."""
        results = []
        for p in self.papers_data:
            meta = p.get("Meta Review Score", "")
            if not meta:
                continue
            avg = self.parse_avg(p.get("Overall Assessment", ""))
            if avg is None or np.isnan(avg):
                continue
            try:
                diff = abs(float(meta) - avg)
            except Exception:
                continue
            if diff >= threshold:
                results.append({
                    "Paper #":    p["Paper #"],
                    "Paper ID":   p["Paper ID"],
                    "Title":      p["Title"],
                    "SAC":        p.get("Senior Area Chair", ""),
                    "AC":         p.get("Area Chair", ""),
                    "Avg Review": round(avg, 2),
                    "AC Score":   float(meta),
                    "Diff":       round(float(meta) - avg, 2),
                    "Divergence": round(diff, 2),
                })
        results.sort(key=lambda r: r["Divergence"], reverse=True)
        return results[:top_n]

    # -----------------------------------------------------------------------
    # High-disagreement papers: high std dev among reviewer scores
    # -----------------------------------------------------------------------

    def compute_high_disagreement_papers(self, threshold=0.7, top_n=15):
        """Papers where std dev of reviewer overall scores >= threshold."""
        results = []
        for p in self.papers_data:
            scores_raw = p.get("Overall List", "")
            if not scores_raw:
                continue
            try:
                scores = [float(s) for s in scores_raw.split(" / ") if s.strip()]
            except Exception:
                continue
            if len(scores) < 2:
                continue
            std = float(np.std(scores))
            if std >= threshold:
                results.append({
                    "Paper #":  p["Paper #"],
                    "Paper ID": p["Paper ID"],
                    "Title":    p["Title"],
                    "SAC":      p.get("Senior Area Chair", ""),
                    "AC":       p.get("Area Chair", ""),
                    "Scores":   scores_raw,
                    "Std Dev":  round(std, 2),
                    "Avg":      round(float(np.mean(scores)), 2),
                })
        results.sort(key=lambda r: r["Std Dev"], reverse=True)
        return results[:top_n]

    # -----------------------------------------------------------------------
    # Reviewer load distribution histogram
    # -----------------------------------------------------------------------

    def compute_reviewer_load_histogram(self):
        """Histogram of how many reviewers have 1, 2, 3, ... papers assigned."""
        if not self.reviewer_load:
            return {"labels": [], "counts": [], "total_reviewers": 0}
        from collections import Counter
        freq = Counter(self.reviewer_load.values())
        max_load = max(freq.keys())
        labels = list(range(1, max_load + 1))
        counts = [freq.get(i, 0) for i in labels]
        zero_count = max(0, len(self.reviewer_load) - sum(c > 0 for c in self.reviewer_load.values()))
        total = len(self.reviewer_load)
        avg_load = round(sum(self.reviewer_load.values()) / total, 2) if total else 0
        return {
            "labels":          labels,
            "counts":          counts,
            "total_reviewers": total,
            "zero_reviews":    zero_count,
            "avg_load":        avg_load,
        }

    # -----------------------------------------------------------------------
    # Load histograms for AC and SAC
    # -----------------------------------------------------------------------

    def compute_ac_load_histogram(self):
        """Histogram of papers-per-AC from ac_meta_data."""
        if not self.ac_meta_data:
            return {"labels": [], "counts": [], "total": 0, "avg_load": 0}
        from collections import Counter
        loads = [r["Num_Papers"] for r in self.ac_meta_data]
        freq = Counter(loads)
        max_load = max(freq.keys())
        labels = list(range(1, max_load + 1))
        counts = [freq.get(i, 0) for i in labels]
        total = len(loads)
        return {
            "labels": labels,
            "counts": counts,
            "total": total,
            "avg_load": round(sum(loads) / total, 2) if total else 0,
        }

    def compute_sac_load_histogram(self):
        """Histogram of papers-per-SAC from sac_meta_data."""
        if not self.sac_meta_data:
            return {"labels": [], "counts": [], "total": 0, "avg_load": 0}
        from collections import Counter
        loads = [r["Num_Papers"] for r in self.sac_meta_data]
        freq = Counter(loads)
        max_load = max(freq.keys())
        labels = list(range(1, max_load + 1))
        counts = [freq.get(i, 0) for i in labels]
        total = len(loads)
        return {
            "labels": labels,
            "counts": counts,
            "total": total,
            "avg_load": round(sum(loads) / total, 2) if total else 0,
        }

    def compute_sac_scoring_data(self, top_n=15):
        """Top SACs by avg |AC score - reviewer avg|, aggregated from papers_data."""
        if not self.papers_data:
            return []
        df = pd.DataFrame(self.papers_data)
        if "Senior Area Chair" not in df.columns:
            return []
        results = []
        for sac_name, group in df.groupby("Senior Area Chair"):
            if not sac_name:
                continue
            overall_vals = group["Overall Assessment"].apply(self.parse_avg).dropna()
            meta_vals = group["Meta Review Score"].apply(
                lambda x: float(x) if x and str(x).strip() else np.nan
            ).dropna()
            if overall_vals.empty or meta_vals.empty:
                continue
            avg_review = round(float(overall_vals.mean()), 2)
            avg_meta   = round(float(meta_vals.mean()), 2)
            diff       = round(avg_meta - avg_review, 2)
            sac_row    = next((r for r in self.sac_meta_data if r["Senior Area Chair"] == sac_name), {})
            results.append({
                "name":          sac_name,
                "email":         sac_row.get("Senior Area Chair Email", ""),
                "avg_review":    avg_review,
                "avg_meta":      avg_meta,
                "difference":    diff,
                "abs_diff":      abs(diff),
                "review_count":  int(overall_vals.count()),
                "meta_count":    int(meta_vals.count()),
                "num_papers":    len(group),
            })
        results.sort(key=lambda x: x["abs_diff"], reverse=True)
        return results[:top_n]

    # -----------------------------------------------------------------------
    # process_data override
    # -----------------------------------------------------------------------

    def process_data(self):
        self.process_papers_data()
        if self.papers_data:
            self.compute_ac_meta_data()
            self.compute_sac_meta_data()
            self.compute_track_data()
            self.compute_attention_papers()
            self.compute_correlation_data()
        self.process_comments_data()

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def _resolve_template_dir(self):
        for p in [Path(__file__).parent / "templates", Path("templates")]:
            if p.exists() and (p / "pc_report.html").exists():
                return p
        raise FileNotFoundError("Cannot find templates/ with pc_report.html")

    def generate_report(self, output_dir="."):
        os.makedirs(output_dir, exist_ok=True)
        self.process_data()

        if not self.papers_data:
            html = ("<html><body><h1>No papers found</h1>"
                    f"<p>Venue: {self.venue_id}</p></body></html>")
            p = Path(output_dir) / "pc_report.html"
            p.write_text(html)
            return p

        template_data = {
            "title":                   f"PC Dashboard: {self.venue_id}",
            "generated_date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":                self.venue_id,
            "overview_stats":          self.compute_overview_stats(),
            "papers":                  self.papers_data,
            "ac_meta":                 self.ac_meta_data,
            "sac_meta":                self.sac_meta_data,
            "track_data":              self.track_data,
            "attention_papers":        self.attention_papers,
            "comments_count":          len(self.comments_data),
            "comments":                self.comments_data,
            "comment_trees":           self.organize_comments_by_paper(),
            "histogram_data":          self.generate_histogram_data(),
            "correlation_data":        self.correlation_data,
            "paper_type_distribution": self.generate_paper_type_distribution(),
            "review_completion_data":  self.generate_review_completion_data(),
            "score_scatter_data":      self.generate_score_scatter_data(),
            "ac_scoring_data":         self.generate_ac_scoring_data(),
            "score_outliers":          self.compute_score_outliers(),
            "high_disagreement":       self.compute_high_disagreement_papers(),
            "reviewer_load":           self.compute_reviewer_load_histogram(),
            "ac_load":                 self.compute_ac_load_histogram(),
            "sac_load":                self.compute_sac_load_histogram(),
            "ac_scoring_top":          self.generate_ac_scoring_data()[:15],
            "sac_scoring_top":         self.compute_sac_scoring_data(),
        }

        template_dir = self._resolve_template_dir()
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        html = env.get_template("pc_report.html").render(**template_data)
        output_path = Path(output_dir) / "pc_report.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path
