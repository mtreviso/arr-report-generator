"""
Program Chair (PC) Commitment Report Generator.

Extends CommitmentReportGenerator in PC mode:
- role='pc': fetches ALL commitment papers (no AC/SAC filter)
- Bulk pre-fetches linked ARR notes and replies upfront to avoid N per-paper API calls
- Augments each paper with SAC, AC, and Track info
- Adds SAC-level and track-level aggregations
- Adds commitment-specific overview stats (recommendations, awards)

Usage via generate_pc_report.py:
  python generate_pc_report.py --phase commitment --venue_id ... [--linked-venue-id ...]

Performance note
----------------
The commitment phase is inherently slower than the review phase because each paper
links to a *different* ARR forum.  Without optimisation this causes ~5 serial API
calls per paper (fetch linked note, fetch linked replies, fetch AC/SAC/Reviewer groups).

When --linked-venue-id is supplied (or can be auto-detected), this class bulk-fetches:
  1. All ARR submissions in one call  → fills linked_note_cache
  2. All reviews / meta-reviews / issue reports / emergency declarations in bulk
     (one get_all_notes call per invitation type)  → fills linked_replies_cache

This brings per-paper API overhead close to zero for the linked-data lookups.
The AC/SAC/Reviewer group lookups go through the commitment venue's group_index
which is already pre-fetched by CommitmentReportGenerator._build_group_index().
"""

import numpy as np
import os
import jinja2
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from arr_commitment_generator import CommitmentReportGenerator


class PCCommitmentGenerator(CommitmentReportGenerator):
    """PC-level dashboard for the commitment phase."""

    TEMPLATE_NAME = "pc_commitment_report.html"

    def __init__(self, username, password, venue_id, me,
                 comments_level='none', skip_api_init=False,
                 linked_venue_id=""):
        super().__init__(
            username=username,
            password=password,
            venue_id=venue_id,
            me=me,
            role='pc',
            comments_level=comments_level,
            skip_api_init=skip_api_init,
        )
        self.linked_venue_id = linked_venue_id.strip()
        self.sac_meta_data = []
        self.track_data    = []
        self.reviewer_load = {}   # reviewer_id -> paper count

    # -----------------------------------------------------------------------
    # Bulk pre-fetch: linked ARR notes + all reply types in one pass
    # -----------------------------------------------------------------------

    def _detect_linked_venue_id(self):
        """
        Auto-detect the ARR venue ID from the first paper_link if linked_venue_id
        was not provided.  Makes a single API call to fetch the linked note and
        reads its 'venueid' content field.
        Returns the detected ID or "" on failure.
        """
        for sub in self.submissions:
            link = sub.content.get('paper_link', {}).get('value', '').strip()
            forum_id = self._parse_linked_forum_id(link) if link else None
            if not forum_id:
                continue
            try:
                note = self.client.get_note(id=forum_id)
                # Try venueid field first (most reliable)
                venueid = note.content.get('venueid', {})
                if isinstance(venueid, dict):
                    venueid = venueid.get('value', '')
                if venueid and isinstance(venueid, str) and venueid.strip():
                    detected = venueid.strip()
                    print(f"[prefetch] Auto-detected ARR venue: {detected}")
                    return detected
                # Fall back: parse from the note's invitation
                for inv in getattr(note, 'invitations', []):
                    if '/-/' in inv:
                        candidate = inv.split('/-/')[0]
                        if candidate:
                            print(f"[prefetch] Auto-detected ARR venue (from invitation): {candidate}")
                            return candidate
            except Exception as e:
                print(f"[prefetch] Could not auto-detect ARR venue: {e}")
            break  # only try the first paper
        return ""

    def _bulk_prefetch_linked_data(self, linked_venue_id):
        """
        Pre-fill linked_note_cache and linked_replies_cache for all linked ARR
        forums using bulk API calls instead of per-paper fetches.

        Steps:
          1. Collect all unique linked forum IDs from self.submissions.
          2. Bulk-fetch the ARR submission notes (fills linked_note_cache).
          3. Fetch all reply types used in process_papers_data in bulk, index
             by forum (fills linked_replies_cache).
        """
        # --- 1. Collect linked forum IDs -----------------------------------
        forum_ids = set()
        for sub in self.submissions:
            link = sub.content.get('paper_link', {}).get('value', '').strip()
            fid  = self._parse_linked_forum_id(link) if link else None
            if fid:
                forum_ids.add(fid)

        if not forum_ids:
            print("[prefetch] No linked forums found — skipping bulk prefetch.")
            return

        print(f"[prefetch] Bulk pre-fetching data for {len(forum_ids)} linked ARR forums...")

        # --- 2. Get ARR submission name ------------------------------------
        arr_submission_name = 'Submission'
        try:
            arr_venue_group     = self.client.get_group(linked_venue_id)
            arr_submission_name = arr_venue_group.content['submission_name']['value']
        except Exception as e:
            print(f"[prefetch] Warning: could not fetch ARR venue group ({e}). "
                  f"Assuming submission_name='{arr_submission_name}'.")

        # --- 3. Bulk-fetch linked ARR notes --------------------------------
        already_cached = sum(1 for fid in forum_ids if fid in self.linked_note_cache)
        if already_cached < len(forum_ids):
            try:
                arr_notes = self.client.get_all_notes(
                    invitation=f'{linked_venue_id}/-/{arr_submission_name}',
                )
                fetched = 0
                for note in arr_notes:
                    if note.id in forum_ids:
                        self.linked_note_cache[note.id] = note
                        fetched += 1
                print(f"[prefetch] Cached {fetched} / {len(forum_ids)} linked ARR notes.")
            except Exception as e:
                print(f"[prefetch] Warning: could not bulk-fetch ARR notes ({e}). "
                      "Falling back to per-paper note fetches.")

        # --- 4. Bulk-fetch replies by invitation type ----------------------
        # These are the invitation suffixes that process_papers_data scans.
        # We use prefix matching so venue-round-specific names (e.g.
        # '…/2025/October/Submission1/-/Official_Review') are caught too.
        inv_suffixes = [
            'Official_Review',
            'Meta_Review',
            'Review_Issue_Report',
            'Meta-Review_Issue_Report',
            'Emergency_Declaration',
            'EmergencyDeclaration',
            'Author-Editor_Confidential_Comment',
            'Comment',
        ]

        replies_by_forum = defaultdict(list)
        already_cached_forums = {fid for fid in forum_ids if fid in self.linked_replies_cache}

        if len(already_cached_forums) < len(forum_ids):
            for suffix in inv_suffixes:
                inv = f'{linked_venue_id}/-/{suffix}'
                try:
                    notes = self.client.get_all_notes(invitation=inv)
                    count = 0
                    for note in notes:
                        if note.forum in forum_ids and note.forum not in already_cached_forums:
                            replies_by_forum[note.forum].append(note)
                            count += 1
                    if count:
                        print(f"[prefetch]   {suffix}: {count} notes")
                except Exception:
                    pass  # invitation may not exist for this venue — silently skip

            # Populate cache for all forums (empty list = no replies of these types)
            for fid in forum_ids:
                if fid not in already_cached_forums:
                    self.linked_replies_cache[fid] = replies_by_forum.get(fid, [])

            print(f"[prefetch] Reply cache populated for {len(forum_ids)} forums.")

    # -----------------------------------------------------------------------
    # Per-paper processing: commitment data + AC / SAC / Track augmentation
    # -----------------------------------------------------------------------

    def process_papers_data(self):
        """
        1. Optionally bulk-prefetch linked ARR data (fast path).
        2. Run CommitmentReportGenerator.process_papers_data() — now mostly
           cache hits instead of per-paper API calls.
        3. Post-augment each paper dict with SAC, AC, Track.
        """
        # Bulk prefetch if we have a client (i.e. not loaded from cache)
        if hasattr(self, 'client') and self.client is not None:
            arr_vid = self.linked_venue_id or self._detect_linked_venue_id()
            if arr_vid:
                self._bulk_prefetch_linked_data(arr_vid)
            else:
                print(
                    "[prefetch] No ARR venue ID provided or detected. "
                    "Pass --linked-venue-id for faster runs. "
                    "Falling back to per-paper fetches."
                )

        super().process_papers_data()

        if self.papers_data:
            self._add_ac_sac_track_info()

    # Columns produced by ARRReportGenerator.process_papers_data() that are
    # absent from CommitmentReportGenerator.process_papers_data().
    # Used by compute_ac_meta_data(), compute_sac_meta_data(), and
    # generate_score_by_type_data() which are inherited from the review path.
    _COMMITMENT_COLUMN_DEFAULTS = {
        "Has Compromised Review": False,  # absent from commitment; crashes compute_ac_meta_data
        "Knows Authors Count":    0,      # used in score scatter data
        "Ready for Rebuttal":     "",     # used in ac_dashboard
        "AC Score":               "",     # alias used in generate_score_scatter_data fallback
    }

    def _add_ac_sac_track_info(self):
        """
        Post-process self.papers_data to add AC, SAC, Track, reviewer load,
        and any review-path columns that are absent from commitment paper dicts
        so that all inherited aggregation methods work without modification.
        """
        sub_by_number = {s.number: s for s in self.submissions}

        for paper in tqdm(self.papers_data, desc="Augmenting PC commitment data"):
            num    = paper["Paper #"]
            prefix = f'{self.venue_id}/{self.submission_name}{num}'

            # ---- Area Chair ------------------------------------------------
            ac_group = self._get_group_cached(f'{prefix}/Area_Chairs')
            ac_entry = (ac_group.members[0]
                        if ac_group and getattr(ac_group, 'members', None) else "")
            ac_uid   = self._resolve_ac_user_id(ac_entry)
            paper["Area Chair"]             = self.get_display_name_for_user(ac_uid) if ac_uid else ""
            paper["Area Chair ID"]          = ac_uid or ""
            paper["Area Chair Email"]       = self.get_email_for_user(ac_uid) if ac_uid else ""
            paper["Area Chair Affiliation"] = self.get_affiliation_for_user(ac_uid) if ac_uid else ""

            # ---- Senior Area Chair -----------------------------------------
            sac_group = self._get_group_cached(f'{prefix}/Senior_Area_Chairs')
            sac_entry = (sac_group.members[0]
                         if sac_group and getattr(sac_group, 'members', None) else "")
            sac_uid   = self._resolve_ac_user_id(sac_entry)
            paper["Senior Area Chair"]             = self.get_display_name_for_user(sac_uid) if sac_uid else ""
            paper["Senior Area Chair ID"]          = sac_uid or ""
            paper["Senior Area Chair Email"]       = self.get_email_for_user(sac_uid) if sac_uid else ""
            paper["Senior Area Chair Affiliation"] = self.get_affiliation_for_user(sac_uid) if sac_uid else ""

            # ---- Track -------------------------------------------------------
            track = ""
            sub   = sub_by_number.get(num)
            if sub:
                for tf in ("primary_area", "track", "area", "subject_area",
                           "research_area", "track_name", "submission_track"):
                    tv = sub.content.get(tf, {})
                    if isinstance(tv, dict):
                        tv = tv.get("value", "")
                    if tv and str(tv).strip():
                        track = str(tv).strip()
                        break

            if not track and sub:
                paper_link = sub.content.get('paper_link', {}).get('value', '')
                forum_id   = self._parse_linked_forum_id(paper_link) if paper_link else None
                if forum_id and forum_id in self.linked_note_cache:
                    linked_note = self.linked_note_cache[forum_id]
                    if linked_note:
                        for tf in ("primary_area", "track", "area", "subject_area",
                                   "research_area", "track_name", "submission_track"):
                            tv = linked_note.content.get(tf, {})
                            if isinstance(tv, dict):
                                tv = tv.get("value", "")
                            if tv and str(tv).strip():
                                track = str(tv).strip()
                                break

            paper["Track"] = track

            # ---- Fill missing review-phase columns --------------------------
            for col, default in self._COMMITMENT_COLUMN_DEFAULTS.items():
                paper.setdefault(col, default)

            # ---- Reviewer load ----------------------------------------------
            rg_id = f"{self.venue_id}/{self.submission_name}{num}/Reviewers"
            rg    = self._get_group_cached(rg_id)
            if rg and getattr(rg, 'members', None):
                for rev_id in rg.members:
                    actual = self._resolve_reviewer_id(rev_id)
                    self.reviewer_load[actual] = self.reviewer_load.get(actual, 0) + 1

    # -----------------------------------------------------------------------
    # Override: restore AC meta aggregation (CommitmentReportGenerator blanks it)
    # -----------------------------------------------------------------------

    def compute_ac_meta_data(self):
        if not self.papers_data:
            return
        from arr_report_generator import ARRReportGenerator
        ARRReportGenerator.compute_ac_meta_data(self)

    # -----------------------------------------------------------------------
    # SAC-level aggregation
    # -----------------------------------------------------------------------

    def compute_sac_meta_data(self):
        if not self.papers_data:
            return
        df = self._papers_df()
        if "Senior Area Chair" not in df.columns or df["Senior Area Chair"].eq("").all():
            self.sac_meta_data = []
            return

        rows = []
        for sac_name, group in df.groupby("Senior Area Chair"):
            if not sac_name:
                continue
            num_papers           = len(group)
            reviews_done         = int(group["Completed Reviews"].sum())
            reviews_expected     = int(group["Expected Reviews"].sum())
            recommendations_done = int((group["Recommendation"] != "").sum()) if "Recommendation" in group.columns else 0
            issues               = int(group["Has Review Issue"].sum())
            meta_review_issues   = int(group["Has Meta Review Issue"].sum()) if "Has Meta Review Issue" in group.columns else 0
            low_conf             = int(group["Has Low Confidence"].sum())
            ethics               = int((group["Ethics Flag"] != "").sum())
            awards               = int((group["Award"] != "").sum()) if "Award" in group.columns else 0
            emerg_decl           = int(group["Has Emergency Declaration"].sum()) if "Has Emergency Declaration" in group.columns else 0
            emerg_assign         = int(group["Has Emergency Assigned"].sum()) if "Has Emergency Assigned" in group.columns else 0

            sac_id    = group["Senior Area Chair ID"].iloc[0]
            sac_email = group["Senior Area Chair Email"].iloc[0]
            sac_affil = group["Senior Area Chair Affiliation"].iloc[0] if "Senior Area Chair Affiliation" in group.columns else ""

            rows.append({
                "Senior Area Chair":             sac_name,
                "Senior Area Chair ID":          sac_id,
                "Senior Area Chair Email":       sac_email,
                "Senior Area Chair Affiliation": sac_affil,
                "Num_Papers":                    num_papers,
                "Completed_Reviews":             reviews_done,
                "Expected_Reviews":              reviews_expected,
                "Review_Pct":                    round(100 * reviews_done / reviews_expected, 1) if reviews_expected else 0,
                "Recommendations_Done":          f"{recommendations_done} / {num_papers}",
                "All_Recommendations_Done":      "✓" if recommendations_done == num_papers else "",
                "Review_Issues":                 issues,
                "Meta_Review_Issues":            meta_review_issues,
                "Low_Conf_Papers":               low_conf,
                "Ethics_Papers":                 ethics,
                "Awards":                        awards,
                "Emergency_Declared":            emerg_decl,
                "Emergency_Assigned":            emerg_assign,
                "Emergency_Unassigned":          max(0, emerg_decl - emerg_assign),
            })

        rows.sort(key=lambda r: r["Num_Papers"], reverse=True)
        self.sac_meta_data = rows

    # -----------------------------------------------------------------------
    # Track-level aggregation
    # -----------------------------------------------------------------------

    def compute_track_data(self):
        if not self.papers_data:
            return
        df = self._papers_df()
        if "Track" not in df.columns or df["Track"].eq("").all():
            self.track_data = []
            return

        rows = []
        for track, group in df.groupby("Track"):
            if not track:
                continue
            num             = len(group)
            recommendations = int((group["Recommendation"] != "").sum()) if "Recommendation" in group.columns else 0
            issues          = int(group["Has Review Issue"].sum())
            meta_issues     = int(group["Has Meta Review Issue"].sum()) if "Has Meta Review Issue" in group.columns else 0
            awards          = int((group["Award"] != "").sum()) if "Award" in group.columns else 0
            avg_overall     = group["Overall Assessment"].apply(self.parse_avg)
            avg_meta        = group["Meta Review Score"].apply(
                lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip()) else float('nan')
            )
            rows.append({
                "Track":           track,
                "Papers":          num,
                "Recommendations": f"{recommendations} / {num}",
                "Rec_Pct":         round(100 * recommendations / num, 1) if num else 0,
                "Avg Overall":     round(avg_overall.mean(), 2) if not avg_overall.isna().all() else "",
                "Avg Meta":        round(avg_meta.mean(), 2) if not avg_meta.isna().all() else "",
                "Issues":          issues,
                "Meta Issues":     meta_issues,
                "Awards":          awards,
            })

        rows.sort(key=lambda r: r["Papers"], reverse=True)
        self.track_data = rows

    # -----------------------------------------------------------------------
    # Overview statistics
    # -----------------------------------------------------------------------

    def compute_overview_stats(self):
        if not self.papers_data:
            return {}
        df = self._papers_df()
        total        = len(df)
        reviews_done = int(df["Completed Reviews"].sum())
        reviews_exp  = int(df["Expected Reviews"].sum())
        issues       = int(df["Has Review Issue"].sum())
        meta_issues  = int(df["Has Meta Review Issue"].sum()) if "Has Meta Review Issue" in df.columns else 0
        ethics       = int((df["Ethics Flag"] != "").sum())
        low_conf     = int(df["Has Low Confidence"].sum())
        emerg_decl   = int(df["Has Emergency Declaration"].sum()) if "Has Emergency Declaration" in df.columns else 0
        sac_count    = df["Senior Area Chair"].nunique() if "Senior Area Chair" in df.columns else 0
        ac_count     = df["Area Chair"].nunique() if "Area Chair" in df.columns else 0

        rec_done = 0
        rec_dist = {}
        if "Recommendation" in df.columns:
            rec_done = int((df["Recommendation"] != "").sum())
            rec_dist = df["Recommendation"].value_counts().to_dict()
            rec_dist.pop("", None)

        pres_dist = {}
        if "Presentation Mode" in df.columns:
            pres_dist = df["Presentation Mode"].value_counts().to_dict()
            pres_dist.pop("", None)

        awards_count = int((df["Award"] != "").sum()) if "Award" in df.columns else 0
        overall_vals = df["Overall Assessment"].apply(self.parse_avg).dropna()
        meta_vals    = df["Meta Review Score"].apply(
            lambda x: float(x) if x and str(x).strip() else np.nan
        ).dropna()
        type_counts  = df["Paper Type"].value_counts().to_dict() if "Paper Type" in df.columns else {}

        return {
            "total_papers":         total,
            "reviews_done":         reviews_done,
            "reviews_expected":     reviews_exp,
            "review_pct":           round(100 * reviews_done / reviews_exp, 1) if reviews_exp else 0,
            "review_issues":        issues,
            "meta_review_issues":   meta_issues,
            "ethics_papers":        ethics,
            "low_conf_papers":      low_conf,
            "emergency_declared":   emerg_decl,
            "sac_count":            sac_count,
            "ac_count":             ac_count,
            "recommendations_done": rec_done,
            "recommendations_pct":  round(100 * rec_done / total, 1) if total else 0,
            "recommendation_dist":  rec_dist,
            "presentation_dist":    pres_dist,
            "awards_count":         awards_count,
            "avg_overall":          round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
            "avg_meta":             round(float(meta_vals.mean()), 2) if len(meta_vals) else None,
            "type_counts":          type_counts,
        }

    # -----------------------------------------------------------------------
    # Score analysis
    # -----------------------------------------------------------------------

    def compute_score_outliers(self, threshold=0.8, top_n=15):
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
                    "SAC_ID":     p.get("Senior Area Chair ID", ""),
                    "AC":         p.get("Area Chair", ""),
                    "AC_ID":      p.get("Area Chair ID", ""),
                    "Avg Review": round(avg, 2),
                    "AC Score":   float(meta),
                    "Diff":       round(float(meta) - avg, 2),
                    "Divergence": round(diff, 2),
                })
        results.sort(key=lambda r: r["Divergence"], reverse=True)
        return results[:top_n]

    def compute_high_disagreement_papers(self, threshold=0.7, top_n=15):
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
                    "SAC_ID":   p.get("Senior Area Chair ID", ""),
                    "AC":       p.get("Area Chair", ""),
                    "AC_ID":    p.get("Area Chair ID", ""),
                    "Scores":   scores_raw,
                    "Std Dev":  round(std, 2),
                    "Avg":      round(float(np.mean(scores)), 2),
                })
        results.sort(key=lambda r: r["Std Dev"], reverse=True)
        return results[:top_n]

    # -----------------------------------------------------------------------
    # Load histograms
    # -----------------------------------------------------------------------

    def compute_reviewer_load_histogram(self):
        if not self.reviewer_load:
            return {"labels": [], "counts": [], "total": 0, "avg_load": 0}
        from collections import Counter
        freq     = Counter(self.reviewer_load.values())
        max_load = max(freq.keys())
        labels   = list(range(1, max_load + 1))
        counts   = [freq.get(i, 0) for i in labels]
        total    = len(self.reviewer_load)
        return {"labels": labels, "counts": counts, "total": total,
                "avg_load": round(sum(self.reviewer_load.values()) / total, 2) if total else 0}

    def compute_ac_load_histogram(self):
        if not self.ac_meta_data:
            return {"labels": [], "counts": [], "total": 0, "avg_load": 0}
        from collections import Counter
        loads    = [r["Num_Papers"] for r in self.ac_meta_data]
        freq     = Counter(loads)
        max_load = max(freq.keys())
        labels   = list(range(1, max_load + 1))
        total    = len(loads)
        return {"labels": labels, "counts": [freq.get(i, 0) for i in labels], "total": total,
                "avg_load": round(sum(loads) / total, 2) if total else 0}

    def compute_sac_load_histogram(self):
        if not self.sac_meta_data:
            return {"labels": [], "counts": [], "total": 0, "avg_load": 0}
        from collections import Counter
        loads    = [r["Num_Papers"] for r in self.sac_meta_data]
        freq     = Counter(loads)
        max_load = max(freq.keys())
        labels   = list(range(1, max_load + 1))
        total    = len(loads)
        return {"labels": labels, "counts": [freq.get(i, 0) for i in labels], "total": total,
                "avg_load": round(sum(loads) / total, 2) if total else 0}

    # -----------------------------------------------------------------------
    # process_data
    # -----------------------------------------------------------------------

    def process_data(self):
        self.process_papers_data()
        if self.papers_data:
            self.compute_ac_meta_data()
            self.compute_sac_meta_data()
            self.compute_track_data()
            self.compute_attention_papers()
            self.compute_correlation_data()
        if self.comments_level != 'none':
            self.process_comments_data()
        else:
            self.comments_data = []

    # -----------------------------------------------------------------------
    # Template data
    # -----------------------------------------------------------------------

    def report_title(self):
        return f"PC Commitment Dashboard: {self.venue_id}"

    def build_template_data(self):
        ac_scoring_data       = self.generate_ac_scoring_data()
        score_by_type_data    = self.generate_score_by_type_data()
        reviewer_load_quality = self.generate_reviewer_load_quality_data()
        return {
            "title":                          self.report_title(),
            "generated_date":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id":                       self.venue_id,
            "overview_stats":                 self.compute_overview_stats(),
            "papers":                         self.papers_data,
            "ac_meta":                        self.ac_meta_data,
            "sac_meta":                       self.sac_meta_data,
            "track_data":                     self.track_data,
            "attention_papers":               self.attention_papers,
            **self.attention_template_flags(),
            "comments_count":                 len(self.comments_data),
            "comments":                       self.comments_data,
            "comments_level":                 self.comments_level,
            "comments_enabled":               self.comments_level != 'none',
            "histogram_data":                 self.generate_histogram_data(),
            "correlation_data":               self.correlation_data,
            "paper_type_distribution":        self.generate_paper_type_distribution(),
            "contribution_type_distribution": self.generate_contribution_type_distribution(),
            "review_completion_data":         self.generate_review_completion_data(),
            "score_scatter_data":             self.generate_score_scatter_data(),
            "ac_scoring_data":                ac_scoring_data,
            "ac_scoring_top":                 ac_scoring_data[:15],
            "score_outliers":                 self.compute_score_outliers(),
            "high_disagreement":              self.compute_high_disagreement_papers(),
            "reviewer_load":                  self.compute_reviewer_load_histogram(),
            "ac_load":                        self.compute_ac_load_histogram(),
            "sac_load":                       self.compute_sac_load_histogram(),
            "score_by_type_data":             score_by_type_data,
            "reviewer_load_quality":          reviewer_load_quality,
        }

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def _resolve_template_dir(self):
        for p in [Path(__file__).parent / "templates", Path("templates")]:
            if p.exists() and (p / self.TEMPLATE_NAME).exists():
                return p
        raise FileNotFoundError(f"Cannot find templates/ with {self.TEMPLATE_NAME}")

    def generate_report(self, output_dir=".", filename="pc_commitment_report.html"):
        os.makedirs(output_dir, exist_ok=True)
        self.process_data()
        if not self.papers_data:
            html = ("<html><body><h1>No commitment papers found</h1>"
                    f"<p>Venue: {self.venue_id}</p></body></html>")
            p = Path(output_dir) / filename
            p.write_text(html)
            return p
        template_dir = self._resolve_template_dir()
        env  = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        html = env.get_template(self.TEMPLATE_NAME).render(**self.build_template_data())
        output_path = Path(output_dir) / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path
