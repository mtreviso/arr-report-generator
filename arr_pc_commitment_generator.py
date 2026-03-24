"""Program Chair commitment-phase dashboard generator.

Builds a PC-scale commitment report with the same overall structure as the
standard PC dashboard, while using commitment-specific paper fields such as
recommendation, presentation mode, awards, review issues, and meta-review
issues.

Key differences from the review-phase PC report:
  - No SAC progress tab (commitment phase only has ACs)
  - Adds recommendation, presentation mode, and award analytics
  - Overview includes decision breakdown charts
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd
from tqdm import tqdm

from arr_commitment_generator import CommitmentReportGenerator
from arr_report_generator import ARRReportGenerator


class PCCommitmentGenerator(CommitmentReportGenerator):
    def __init__(self, username, password, venue_id, me, comments_level='none', skip_api_init=False,
                 linked_venue_id=None):
        super().__init__(
            username=username,
            password=password,
            venue_id=venue_id,
            me=me,
            role='pc',
            comments_level=comments_level,
            skip_api_init=skip_api_init,
            linked_venue_id=linked_venue_id,
        )
        self.track_data = []
        self.attention_papers = []
        self.reviewer_load = {}

    def process_papers_data(self):
        if not self.submissions:
            print("No submissions to process.")
            return

        print(f"Processing {len(self.submissions)} papers for PC commitment dashboard...")
        base_url = "https://openreview.net/forum?id="
        self.reviewer_load = defaultdict(int)

        for sub in tqdm(self.submissions, desc="Processing papers"):
            recommendation, presentation_mode, award = self._extract_commitment_meta_info(sub)

            prefix = f'{self.venue_id}/{self.submission_name}{sub.number}'

            ac_group = self._get_group_cached(f'{prefix}/Area_Chairs')
            ac_entry = (ac_group.members[0] if ac_group and getattr(ac_group, 'members', None) else "")
            ac_uid = self._resolve_ac_user_id(ac_entry)
            ac_name = self.get_display_name_for_user(ac_uid) if ac_uid else ""
            ac_email = self.get_email_for_user(ac_uid) if ac_uid else ""
            ac_affiliation = self.get_affiliation_for_user(ac_uid) if ac_uid else ""

            sac_group = self._get_group_cached(f'{prefix}/Senior_Area_Chairs')
            sac_entry = (sac_group.members[0] if sac_group and getattr(sac_group, 'members', None) else "")
            sac_uid = self._resolve_ac_user_id(sac_entry)
            sac_name = self.get_display_name_for_user(sac_uid) if sac_uid else ""
            sac_email = self.get_email_for_user(sac_uid) if sac_uid else ""
            sac_affiliation = self.get_affiliation_for_user(sac_uid) if sac_uid else ""

            paper_title = sub.content.get('title', {}).get('value', 'Untitled')
            paper_type = self._get_content_value(sub.content, 'paper_type', '')
            contribution_types = self._extract_contribution_types(sub.content)
            response_to_meta = self._get_content_value(sub.content, 'response_to_metareview', '')

            track = ""
            for source_content in (sub.content,):
                for field in ('primary_area', 'track', 'area', 'subject_area', 'research_area', 'track_name', 'submission_track'):
                    value = source_content.get(field, {})
                    if isinstance(value, dict):
                        value = value.get('value', '')
                    if value and str(value).strip():
                        track = str(value).strip()
                        break
                if track:
                    break

            paper_link = sub.content.get('paper_link', {}).get('value', '')
            linked_forum = self._parse_linked_forum_id(paper_link)
            linked_note = self._get_linked_note(linked_forum) if linked_forum else None
            replies = self._get_linked_replies(linked_forum) if linked_forum else []

            if linked_note:
                if not paper_type:
                    paper_type = self._get_content_value(linked_note.content, 'paper_type', '')
                if not contribution_types:
                    contribution_types = self._extract_contribution_types(linked_note.content)
                if not track:
                    for field in ('primary_area', 'track', 'area', 'subject_area', 'research_area', 'track_name', 'submission_track'):
                        value = linked_note.content.get(field, {})
                        if isinstance(value, dict):
                            value = value.get('value', '')
                        if value and str(value).strip():
                            track = str(value).strip()
                            break

            prev_url = ""
            if linked_note and 'previous_URL' in linked_note.content:
                prev_url = linked_note.content['previous_URL'].get('value', '').strip()

            is_anonymous = self._infer_anonymity(sub.content)
            if not is_anonymous and linked_note:
                is_anonymous = self._infer_anonymity(linked_note.content)

            completed_reviews = 0
            expected_reviews = 0
            meta_review_score = ""
            confidence_scores = []
            soundness_scores = []
            excitement_scores = []
            overall_assessment_scores = []
            ethics_flag = ""
            preprint_flag = ""
            has_review_issue = False
            review_issue_link = ""
            review_issue_count = 0
            has_meta_review_issue = False
            meta_review_issue_link = ""
            meta_review_issue_count = 0
            has_confidential = False
            has_emergency_declaration = False
            emergency_declaration_link = ""
            emergency_declaration_count = 0
            reviewer_ethics = []
            preprinters = []

            for reply in replies:
                invs = getattr(reply, 'invitations', [])

                if any('/-/Review_Issue_Report' in i for i in invs):
                    has_review_issue = True
                    review_issue_count += 1
                    if not review_issue_link:
                        review_issue_link = f"https://openreview.net/forum?id={getattr(reply,'forum','')}&noteId={getattr(reply,'id','')}"

                if any('/-/Meta-Review_Issue_Report' in i for i in invs):
                    has_meta_review_issue = True
                    meta_review_issue_count += 1
                    if not meta_review_issue_link:
                        meta_review_issue_link = f"https://openreview.net/forum?id={getattr(reply,'forum','')}&noteId={getattr(reply,'id','')}"

                if self._is_emergency_declaration_reply(reply):
                    has_emergency_declaration = True
                    emergency_declaration_count += 1
                    if not emergency_declaration_link:
                        emergency_declaration_link = self._reply_link(reply)

                if self.is_relevant_comment(reply):
                    has_confidential = True

                if self.is_actual_review(reply):
                    completed_reviews += 1
                    content = self._reply_content(reply)
                    for field, lst in [('confidence', confidence_scores), ('soundness', soundness_scores), ('excitement', excitement_scores), ('overall_assessment', overall_assessment_scores)]:
                        v = self._extract_numeric_score(content, field)
                        if v is not None:
                            lst.append(v)

                    ne = content.get('needs_ethics_review', {}).get('value', '').strip().lower()
                    ec = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if ne == 'yes' or (ec and 'no' not in ec):
                        if hasattr(reply, 'number'):
                            reviewer_ethics.append(str(reply.number))

                    src = content.get('Knowledge_of_paper_source', {}).get('value', [])
                    for s in (src if isinstance(src, list) else [src]):
                        if isinstance(s, str) and any(t in s.lower() for t in ('arxiv', 'preprint')):
                            if hasattr(reply, 'number'):
                                preprinters.append(f'R{reply.number}')
                            break

                if self.is_meta_review(reply):
                    content = self._reply_content(reply)
                    meta_review_score = self._extract_meta_review_score(content)
                    ec2 = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                    if ec2 and 'no concerns' not in ec2:
                        ethics_flag = 'AC: yes'

            rg_id = f'{prefix}/Reviewers'
            rg = self._get_group_cached(rg_id)
            if rg and getattr(rg, 'members', None):
                expected_reviews = len(rg.members)
                for reviewer in rg.members:
                    if isinstance(reviewer, str):
                        self.reviewer_load[reviewer] += 1
            else:
                expected_reviews = max(3, completed_reviews)

            has_emergency_reviewer = False
            emergency_reviewer_count = 0
            for suffix in ['/Emergency_Reviewers', '/Emergency_Reviewer', '/Emergency_Review_Assignees']:
                erg = self._get_group_cached(f'{prefix}{suffix}')
                if erg and getattr(erg, 'members', None):
                    has_emergency_reviewer = True
                    emergency_reviewer_count = len(erg.members)
                    break

            if reviewer_ethics:
                suffix = 'Reviewer: ' + ', '.join(reviewer_ethics)
                ethics_flag = suffix if not ethics_flag else f'{ethics_flag}; {suffix}'
            if preprinters:
                preprint_flag = 'Yes (' + ', '.join(preprinters) + ')'
            if not is_anonymous:
                is_anonymous = 'No' if preprint_flag else 'Yes'

            has_low_confidence = any(s <= self.LOW_CONF_THRESHOLD for s in confidence_scores)
            low_conf_reviewers = ', '.join(str(round(s)) for s in confidence_scores if s <= self.LOW_CONF_THRESHOLD)
            fmt = self.format_scores_as_avg_std

            # Count flags for sorting
            flag_count = sum([
                bool(has_review_issue),
                bool(has_meta_review_issue),
                bool(has_low_confidence),
                bool(ethics_flag),
                bool(has_confidential),
                bool(has_emergency_declaration),
                1 if (isinstance(is_anonymous, str) and is_anonymous.lower() == 'no') else 0,
            ])

            paper_data = {
                'Paper #': sub.number,
                'Paper ID': sub.id,
                'Title': paper_title,
                'Submission Link': f'{base_url}{sub.id}',
                'Linked Forum': f'{base_url}{linked_forum}' if linked_forum else '',
                'Paper Type': paper_type,
                'Track': track,
                'Contribution Types': '; '.join(contribution_types),
                'Contribution Type List': contribution_types,
                'Senior Area Chair': sac_name,
                'Senior Area Chair ID': sac_uid or '',
                'Senior Area Chair Email': sac_email,
                'Senior Area Chair Affiliation': sac_affiliation,
                'Area Chair': ac_name,
                'Area Chair ID': ac_uid or '',
                'Area Chair Email': ac_email,
                'Area Chair Affiliation': ac_affiliation,
                'Completed Reviews': completed_reviews,
                'Expected Reviews': expected_reviews,
                'Resubmission': prev_url,
                'Response to Meta-review': response_to_meta,
                'Is Anonymous': is_anonymous,
                'Has Review Issue': has_review_issue,
                'Review Issue Link': review_issue_link,
                'Review Issue Count': review_issue_count,
                'Has Meta Review Issue': has_meta_review_issue,
                'Meta Review Issue Link': meta_review_issue_link,
                'Meta Review Issue Count': meta_review_issue_count,
                'Has Confidential': '✓' if has_confidential else '',
                'Has Low Confidence': has_low_confidence,
                'Low Confidence Reviewers': low_conf_reviewers,
                'Has Emergency Declaration': has_emergency_declaration,
                'Emergency Declaration Link': emergency_declaration_link,
                'Emergency Declaration Count': emergency_declaration_count,
                'Has Emergency Reviewer': has_emergency_reviewer,
                'Emergency Reviewer Count': emergency_reviewer_count,
                'Has Emergency Assigned': bool(has_emergency_declaration and has_emergency_reviewer),
                'Has Emergency Unmet': bool(has_emergency_declaration and not has_emergency_reviewer),
                'Ethics Flag': ethics_flag,
                'Preprint Flag': preprint_flag,
                'Has Compromised Review': bool(preprint_flag),
                'Flag Count': flag_count,
                'Reviewer Confidence': fmt(confidence_scores),
                'Confidence List': ' / '.join(f'{s:.1f}' for s in confidence_scores),
                'Soundness Score': fmt(soundness_scores),
                'Soundness List': ' / '.join(f'{s:.1f}' for s in soundness_scores),
                'Excitement Score': fmt(excitement_scores),
                'Excitement List': ' / '.join(f'{s:.1f}' for s in excitement_scores),
                'Overall Assessment': fmt(overall_assessment_scores),
                'Overall List': ' / '.join(f'{s:.1f}' for s in overall_assessment_scores),
                'Meta Review Score': meta_review_score,
                'Recommendation': recommendation,
                'Presentation Mode': presentation_mode,
                'Award': award,
            }
            self.papers_data.append(paper_data)

            if overall_assessment_scores:
                self.score_distributions['overall_assessment'].append(sum(overall_assessment_scores) / len(overall_assessment_scores))
            if meta_review_score:
                try:
                    self.score_distributions['meta_review'].append(float(meta_review_score))
                except Exception:
                    pass

        self.reviewer_load = dict(self.reviewer_load)
        print(f"Done! Processed {len(self.papers_data)} papers.")

    def compute_ac_meta_data(self):
        """Call ARRReportGenerator's version directly (skip CommitmentReportGenerator
        which sets ac_meta_data = [])."""
        ARRReportGenerator.compute_ac_meta_data(self)

    def compute_track_data(self):
        if not self.papers_data:
            self.track_data = []
            return
        df = self._papers_df()
        if 'Track' not in df.columns or df['Track'].eq('').all():
            self.track_data = []
            return
        rows = []
        for track, group in df.groupby('Track'):
            if not track:
                continue
            num = len(group)
            done = int((group['Meta Review Score'] != '').sum())
            issues = int(group['Has Review Issue'].sum())
            meta_issues = int(group['Has Meta Review Issue'].sum()) if 'Has Meta Review Issue' in group.columns else 0
            avg_overall = group['Overall Assessment'].apply(self.parse_avg)
            avg_meta = group['Meta Review Score'].apply(lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip()) else float('nan'))
            rec_filled = int((group['Recommendation'].astype(str).str.strip() != '').sum()) if 'Recommendation' in group.columns else 0
            rows.append({
                'Track': track,
                'Papers': num,
                'Meta Done': f'{done} / {num}',
                'Meta Pct': round(100 * done / num, 1) if num else 0,
                'Avg Overall': round(avg_overall.mean(), 2) if not avg_overall.isna().all() else '',
                'Avg Meta': round(avg_meta.mean(), 2) if not avg_meta.isna().all() else '',
                'Issues': issues,
                'Meta Issues': meta_issues,
                'Rec Filled': rec_filled,
                'Rec Pct': round(100 * rec_filled / num, 1) if num else 0,
            })
        rows.sort(key=lambda r: r['Papers'], reverse=True)
        self.track_data = rows

    def compute_attention_papers(self):
        if not self.papers_data:
            self.attention_papers = []
            return
        rows = []
        for paper in self.papers_data:
            missing_reviews = max(0, int(paper.get('Expected Reviews', 0) or 0) - int(paper.get('Completed Reviews', 0) or 0))
            has_issue = bool(paper.get('Has Review Issue'))
            has_meta_issue = bool(paper.get('Has Meta Review Issue'))
            has_ethics = bool(paper.get('Ethics Flag'))
            has_emergency = bool(paper.get('Has Emergency Declaration'))
            if not (has_issue or has_meta_issue or has_ethics or has_emergency or missing_reviews > 0):
                continue
            enriched = dict(paper)
            enriched['Missing Reviews'] = missing_reviews
            enriched['Has Missing Reviews'] = missing_reviews > 0
            rows.append(enriched)
        rows.sort(key=lambda p: (
            0 if p.get('Has Review Issue') else 1,
            0 if p.get('Has Meta Review Issue') else 1,
            0 if p.get('Missing Reviews', 0) else 1,
            0 if p.get('Has Emergency Declaration') else 1,
            -int(p.get('Missing Reviews', 0) or 0),
            p.get('Paper #', 0),
        ))
        self.attention_papers = rows

    def compute_overview_stats(self):
        if not self.papers_data:
            return {}
        df = self._papers_df()
        total = len(df)
        reviews_done = int(df['Completed Reviews'].sum())
        reviews_exp = int(df['Expected Reviews'].sum())
        meta_done = int((df['Meta Review Score'] != '').sum())
        issues = int(df['Has Review Issue'].sum())
        meta_issues = int(df['Has Meta Review Issue'].sum()) if 'Has Meta Review Issue' in df.columns else 0
        ethics = int((df['Ethics Flag'] != '').sum())
        low_conf = int(df['Has Low Confidence'].sum())
        sb_papers = int(df['Has Compromised Review'].sum()) if 'Has Compromised Review' in df.columns else 0
        all_reviewed = int((df['Completed Reviews'] >= df['Expected Reviews']).sum())
        ac_count = df['Area Chair'].nunique() if 'Area Chair' in df.columns else 0
        overall_vals = df['Overall Assessment'].apply(self.parse_avg).dropna()
        meta_vals = df['Meta Review Score'].apply(lambda x: float(x) if x and str(x).strip() else np.nan).dropna()
        emerg_decl = int((pd.to_numeric(df['Emergency Declaration Count'], errors='coerce').fillna(0) > 0).sum()) if 'Emergency Declaration Count' in df.columns else 0

        # Paper type counts — filter out empty types
        type_counts = {}
        if 'Paper Type' in df.columns:
            for ptype, cnt in df['Paper Type'].value_counts().items():
                label = str(ptype).strip()
                if label:
                    type_counts[label] = int(cnt)
            empty_type_count = int((df['Paper Type'].astype(str).str.strip() == '').sum())
            if empty_type_count > 0:
                type_counts['Unspecified'] = empty_type_count

        # Commitment-specific stats
        recommended = int((df['Recommendation'].astype(str).str.strip() != '').sum()) if 'Recommendation' in df.columns else 0
        has_presentation = int((df['Presentation Mode'].astype(str).str.strip() != '').sum()) if 'Presentation Mode' in df.columns else 0

        # Award: split comma-separated values, exclude "do not consider"
        has_award = 0
        award_breakdown = {}
        if 'Award' in df.columns:
            for raw_award in df['Award']:
                raw = str(raw_award).strip()
                if not raw:
                    continue
                individual = [a.strip() for a in raw.split(',') if a.strip()]
                real_awards = [a for a in individual
                               if 'do not consider' not in a.lower()
                               and 'none' not in a.lower()]
                if real_awards:
                    has_award += 1
                for a in real_awards:
                    award_breakdown[a] = award_breakdown.get(a, 0) + 1

        rec_breakdown = {}
        if 'Recommendation' in df.columns:
            for val, cnt in df['Recommendation'].value_counts().items():
                label = str(val).strip()
                if label:
                    rec_breakdown[label] = int(cnt)

        pres_breakdown = {}
        if 'Presentation Mode' in df.columns:
            for val, cnt in df['Presentation Mode'].value_counts().items():
                label = str(val).strip()
                if label:
                    pres_breakdown[label] = int(cnt)

        return {
            'total_papers': total,
            'reviews_done': reviews_done,
            'reviews_expected': reviews_exp,
            'review_pct': round(100 * reviews_done / reviews_exp, 1) if reviews_exp else 0,
            'papers_all_reviewed': all_reviewed,
            'papers_all_reviewed_pct': round(100 * all_reviewed / total, 1) if total else 0,
            'meta_done': meta_done,
            'meta_pct': round(100 * meta_done / total, 1) if total else 0,
            'review_issues': issues,
            'meta_review_issues': meta_issues,
            'ethics_papers': ethics,
            'low_conf_papers': low_conf,
            'sb_papers': sb_papers,
            'ac_count': ac_count,
            'emergency_declared': emerg_decl,
            'avg_overall': round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
            'avg_meta': round(float(meta_vals.mean()), 2) if len(meta_vals) else None,
            'type_counts': type_counts,
            'recommended_papers': recommended,
            'recommended_pct': round(100 * recommended / total, 1) if total else 0,
            'has_presentation': has_presentation,
            'has_award': has_award,
            'rec_breakdown': rec_breakdown,
            'pres_breakdown': pres_breakdown,
            'award_breakdown': award_breakdown,
        }

    def compute_score_outliers(self, threshold=0.8, top_n=15):
        results = []
        for p in self.papers_data:
            meta = p.get('Meta Review Score', '')
            if not meta:
                continue
            avg = self.parse_avg(p.get('Overall Assessment', ''))
            if avg is None or np.isnan(avg):
                continue
            try:
                diff = abs(float(meta) - avg)
            except Exception:
                continue
            if diff >= threshold:
                results.append({
                    'Paper #': p['Paper #'], 'Paper ID': p['Paper ID'], 'Title': p['Title'],
                    'AC': p.get('Area Chair', ''), 'AC_ID': p.get('Area Chair ID', ''),
                    'Avg Review': round(avg, 2), 'AC Score': float(meta),
                    'Diff': round(float(meta) - avg, 2), 'Divergence': round(diff, 2),
                })
        results.sort(key=lambda r: r['Divergence'], reverse=True)
        return results[:top_n]

    def compute_high_disagreement_papers(self, threshold=0.7, top_n=15):
        results = []
        for p in self.papers_data:
            scores_raw = p.get('Overall List', '')
            if not scores_raw:
                continue
            try:
                scores = [float(s) for s in scores_raw.split(' / ') if s.strip()]
            except Exception:
                continue
            if len(scores) < 2:
                continue
            std = float(np.std(scores))
            if std >= threshold:
                results.append({
                    'Paper #': p['Paper #'], 'Paper ID': p['Paper ID'], 'Title': p['Title'],
                    'AC': p.get('Area Chair', ''), 'AC_ID': p.get('Area Chair ID', ''),
                    'Scores': scores_raw, 'Std Dev': round(std, 2), 'Avg': round(float(np.mean(scores)), 2),
                })
        results.sort(key=lambda r: r['Std Dev'], reverse=True)
        return results[:top_n]

    def compute_reviewer_load_histogram(self):
        if not self.reviewer_load:
            return {'labels': [], 'counts': [], 'total': 0, 'avg_load': 0}
        freq = Counter(self.reviewer_load.values())
        max_load = max(freq.keys())
        labels = list(range(1, max_load + 1))
        counts = [freq.get(i, 0) for i in labels]
        total = len(self.reviewer_load)
        return {'labels': labels, 'counts': counts, 'total': total, 'avg_load': round(sum(self.reviewer_load.values()) / total, 2) if total else 0}

    def compute_ac_load_histogram(self):
        if not self.ac_meta_data:
            return {'labels': [], 'counts': [], 'total': 0, 'avg_load': 0}
        loads = [r['Num_Papers'] for r in self.ac_meta_data]
        freq = Counter(loads)
        max_load = max(freq.keys())
        labels = list(range(1, max_load + 1))
        counts = [freq.get(i, 0) for i in labels]
        total = len(loads)
        return {'labels': labels, 'counts': counts, 'total': total, 'avg_load': round(sum(loads) / total, 2) if total else 0}

    def compute_recommendation_vs_score_data(self):
        """For each recommendation category, compute avg overall and meta scores."""
        if not self.papers_data:
            return []
        df = self._papers_df()
        if 'Recommendation' not in df.columns:
            return []
        results = []
        for rec, group in df.groupby('Recommendation'):
            rec_label = str(rec).strip()
            if not rec_label:
                continue
            overall_vals = group['Overall Assessment'].apply(self.parse_avg).dropna()
            meta_vals = group['Meta Review Score'].apply(lambda x: float(x) if x and str(x).strip() else np.nan).dropna()
            results.append({
                'recommendation': rec_label, 'count': len(group),
                'avg_overall': round(float(overall_vals.mean()), 2) if len(overall_vals) else None,
                'avg_meta': round(float(meta_vals.mean()), 2) if len(meta_vals) else None,
            })
        results.sort(key=lambda r: r['count'], reverse=True)
        return results

    def process_data(self):
        self.process_papers_data()
        if self.papers_data:
            self.compute_ac_meta_data()
            self.compute_track_data()
            self.compute_attention_papers()
            self.compute_correlation_data()
        if self.comments_level != 'none':
            self.process_comments_data()
        else:
            self.comments_data = []

    def _resolve_template_dir(self):
        for p in [Path(__file__).parent / 'templates', Path('templates')]:
            if p.exists() and (p / 'pc_commitment_report.html').exists():
                return p
        raise FileNotFoundError('Cannot find templates/ with pc_commitment_report.html')

    def report_title(self):
        return f'PC Commitment Dashboard: {self.venue_id}'

    def build_template_data(self):
        ac_scoring_data = self.generate_ac_scoring_data()
        score_by_type_data = self.generate_score_by_type_data()
        reviewer_load_quality = self.generate_reviewer_load_quality_data()
        return {
            'title': self.report_title(),
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'venue_id': self.venue_id,
            'overview_stats': self.compute_overview_stats(),
            'papers': self.papers_data,
            'ac_meta': self.ac_meta_data,
            'track_data': self.track_data,
            'attention_papers': self.attention_papers,
            **self.attention_template_flags(),
            'comments_count': len(self.comments_data),
            'comments': self.comments_data,
            'comments_level': self.comments_level,
            'comments_enabled': self.comments_level != 'none',
            'histogram_data': self.generate_histogram_data(),
            'correlation_data': self.correlation_data,
            'paper_type_distribution': self.generate_paper_type_distribution(),
            'contribution_type_distribution': self.generate_contribution_type_distribution(),
            'review_completion_data': self.generate_review_completion_data(),
            'score_scatter_data': self.generate_score_scatter_data(),
            'ac_scoring_data': ac_scoring_data,
            'score_outliers': self.compute_score_outliers(),
            'high_disagreement': self.compute_high_disagreement_papers(),
            'reviewer_load': self.compute_reviewer_load_histogram(),
            'ac_load': self.compute_ac_load_histogram(),
            'ac_scoring_top': ac_scoring_data[:15],
            'score_by_type_data': score_by_type_data,
            'reviewer_load_quality': reviewer_load_quality,
            'rec_vs_score': self.compute_recommendation_vs_score_data(),
        }

    def generate_report(self, output_dir='.', filename='pc_commitment_report.html'):
        os.makedirs(output_dir, exist_ok=True)
        self.process_data()

        if not self.papers_data:
            html = ('<html><body><h1>No papers found</h1>'
                    f'<p>Venue: {self.venue_id}</p></body></html>')
            p = Path(output_dir) / filename
            p.write_text(html)
            return p

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._resolve_template_dir())),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        html = env.get_template('pc_commitment_report.html').render(**self.build_template_data())
        output_path = Path(output_dir) / filename
        output_path.write_text(html, encoding='utf-8')
        return output_path
