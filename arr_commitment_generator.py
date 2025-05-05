"""
Commitment Phase Report Generator for ACL Conferences

This module extends the ARRReportGenerator class to create a specialized
report generator for the commitment phase of ACL conferences.
"""

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
from urllib.parse import urlparse, parse_qs
from arr_report_generator import ARRReportGenerator


class CommitmentReportGenerator(ARRReportGenerator):
    """
    Report generator specifically for the commitment phase of ACL conferences.
    Inherits from ARRReportGenerator to reuse functionality while customizing
    the data collection logic and display for commitment papers.
    """
    
    def __init__(self, username, password, venue_id, me):
        """
        Initialize the CommitmentReportGenerator with more direct paper access.
        """
        # Initialize OpenReview client directly
        self.username = username
        self.password = password
        self.venue_id = venue_id
        self.me = me
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net', 
            username=username, 
            password=password
        )
        
        # Get venue information
        self.venue_group = self.client.get_group(venue_id)
        self.submission_name = self.venue_group.content['submission_name']['value']
        
        print(f"Checking if {me} is a Senior Area Chair for {venue_id}...")
        
        # First check if the user is a member of any AC groups
        try:
            self.my_ac_groups = []
            
            # Check if user is in main Area Chairs group
            try:
                ac_group = self.client.get_group(f"{venue_id}/Area_Chairs")
                if me in ac_group.members:
                    self.my_ac_groups.append(ac_group.id)
                    print(f"Found you in the main Area Chairs group")
            except:
                pass
                
            # Check for all groups where the user is a member
            my_groups = self.client.get_all_groups(member=me)
            
            # Specifically look for AC groups related to this venue
            ac_related_groups = [g for g in my_groups if venue_id in g.id and "Area_Chair" in g.id]
            self.my_ac_groups.extend([g.id for g in ac_related_groups])
            
            print(f"You are a member of {len(self.my_ac_groups)} AC-related groups")
        except Exception as e:
            print(f"Error checking AC groups: {e}")
            self.my_ac_groups = []

        # Get all submissions directly
        print(f"Retrieving submissions for {venue_id}...")
        try:
            # Get all submissions
            all_submissions = self.client.get_all_notes(
                invitation=f'{venue_id}/-/{self.submission_name}', 
                details='replies'
            )
            print(f"Retrieved {len(all_submissions)} total submissions")
            
            # First identify the papers that are definitely assigned to this AC
            self.submissions = []
            for submission in tqdm(all_submissions, desc="Finding your assigned papers"):
                is_assigned = False
                
                # Check method 1: Direct AC group membership
                try:
                    ac_group_id = f"{venue_id}/{self.submission_name}{submission.number}/Area_Chairs"
                    ac_group = self.client.get_group(ac_group_id)
                    if me in ac_group.members:
                        is_assigned = True
                except:
                    pass

                # Check for a link
                link = submission.content.get("paper_link", {}).get("value", "")
                if is_assigned and not link.strip():
                    is_assigned = False
                
                # Only consider assigned papers to the AC
                if is_assigned:
                    print(f"✓ Paper {submission.number}. Link: {link}")
                    self.submissions.append(submission)
                    
            print(f"Found {len(self.submissions)} papers assigned to you")
                
        except Exception as e:
            print(f"Error retrieving submissions: {e}")
            self.submissions = []
        
        # Data containers
        self.papers_data = []
        self.comments_data = []
        self.score_distributions = {
            'overall_assessment': [],
            'meta_review': []
        }
        self.correlation_data = None
        self.ac_meta_data = []  # Empty AC meta data


    def process_papers_data(self):
        """
        Process papers data ONLY for papers where you're assigned as SAC.
        Uses very strict filtering to ensure only your assigned papers appear.
        """
        if not self.submissions:
            print("No submissions to process.")
            return
            
        # Let's print information about the papers we have
        print(f"Starting to process {len(self.submissions)} papers...")
        
        base_url = "https://openreview.net/forum?id="
        processed_count = 0
        
        # Process ONLY your papers 
        for submission in tqdm(self.submissions, desc="Processing your assigned papers"):
            # Skip withdrawn/desk rejected
            try:
                if hasattr(submission, 'content'):
                    # Check withdrawal
                    if 'withdrawal_confirmation' in submission.content:
                        if submission.content['withdrawal_confirmation'].get('value', '').strip():
                            print(f"Skipping withdrawn paper {submission.number}")
                            continue
                            
                    # Check venue for "desk rejected"
                    if 'venue' in submission.content:
                        venue_val = submission.content['venue'].get('value', '').lower()
                        if "withdrawn" in venue_val or "desk rejected" in venue_val:
                            print(f"Skipping desk rejected paper {submission.number}")
                            continue
            except Exception:
                pass
            
            # Get paper title and basic info
            paper_title = submission.content.get('title', {}).get('value', 'Untitled')
            paper_type = submission.content.get('paper_type', {}).get('value', '')
            
            # Get linked forum
            linked_forum = None
            linked_note = None
            replies = []
            
            if hasattr(submission, 'content') and 'paper_link' in submission.content:
                paper_link = submission.content['paper_link'].get('value', '')
                
                if paper_link:
                    try:
                        parsed = urlparse(paper_link)
                        qs = parse_qs(parsed.query)
                        linked_forum = qs.get("id", qs.get("noteId", [None]))[0]
                        
                        if linked_forum:
                            try:
                                linked_note = self.client.get_note(id=linked_forum)
                                replies = self.client.get_notes(forum=linked_forum)
                            except Exception as e:
                                print(f"Error fetching linked forum {linked_forum}: {e}")
                    except Exception as e:
                        print(f"Error parsing paper_link: {e}")
                        
            # If we got linked note, possibly get paper type from there if not in submission
            if linked_note and not paper_type:
                if hasattr(linked_note, 'content') and 'paper_type' in linked_note.content:
                    paper_type = linked_note.content['paper_type'].get('value', '')
            
            # Extract paper fields
            response_to_meta = submission.content.get('response_to_metareview', {}).get('value', '')
            
            # Get previous URL if linked note is available
            prev_url = ""
            if linked_note and hasattr(linked_note, 'content') and 'previous_URL' in linked_note.content:
                prev_url = linked_note.content['previous_URL'].get('value', '').strip()
            
            # Collect review data
            completed_reviews = 0
            expected_reviews = 0
            meta_review_score = ""
            confidence_scores = []
            soundness_scores = []
            excitement_scores = []
            overall_assessment_scores = []
            has_confidential = False
            ethics_flag = ""
            preprint_flag = ""
            
            # Process reviews if we have replies
            if replies:
                # Count reviews
                completed_reviews = sum(1 for reply in replies if self.is_actual_review(reply))
                
                # Get meta-review
                recommendation = ""
                presentation_mode = ""
                award = ""
                for reply in replies:
                    if self.is_meta_review(reply):
                        if hasattr(reply, 'content'):
                            content = reply.content
                            meta_review_score = (
                                content.get('overall_assessment', {}).get('value', '') or
                                content.get('overall_rating', {}).get('value', '') or
                                content.get('score', {}).get('value', '')
                            )
                            
                            # Extract additional fields
                            recommendation = content.get('recommendation', {}).get('value', '')
                            presentation_mode = content.get('presentation_mode', {}).get('value', '')
                            award_field = content.get('award', {}).get('value', [])
                            
                            # Handle award field (might be a list)
                            if isinstance(award_field, list):
                                award = ", ".join(award_field) 
                            else:
                                award = award_field
                            
                            # Check for ethics concerns
                            if 'ethical_concerns' in content:
                                ethical_concerns = content['ethical_concerns'].get('value', '').strip().lower()
                                if ethical_concerns and "no concerns" not in ethical_concerns:
                                    ethics_flag = "AC: yes"
                        break
                
                # Process reviewer data
                reviewer_ethics = []
                preprinters = []
                
                for reply in replies:
                    if self.is_actual_review(reply):
                        if hasattr(reply, 'content'):
                            content = reply.content
                            
                            # Get review scores
                            try:
                                if 'confidence' in content:
                                    val = content['confidence'].get('value')
                                    if val is not None:
                                        confidence_scores.append(float(val))
                            except:
                                pass
                                
                            try:
                                if 'soundness' in content:
                                    val = content['soundness'].get('value')
                                    if val is not None:
                                        soundness_scores.append(float(val))
                            except:
                                pass
                                
                            try:
                                if 'excitement' in content:
                                    val = content['excitement'].get('value')
                                    if val is not None:
                                        excitement_scores.append(float(val))
                            except:
                                pass
                                
                            try:
                                if 'overall_assessment' in content:
                                    val = content['overall_assessment'].get('value')
                                    if val is not None:
                                        overall_assessment_scores.append(float(val))
                            except:
                                pass
                            
                            # Check for ethics flags
                            needs_ethics = content.get('needs_ethics_review', {}).get('value', '').strip().lower()
                            ethical_concerns = content.get('ethical_concerns', {}).get('value', '').strip().lower()
                            
                            if (needs_ethics == "yes") or (ethical_concerns and "no" not in ethical_concerns):
                                if hasattr(reply, 'number'):
                                    reviewer_ethics.append(str(reply.number))
                            
                            # Check for preprint knowledge
                            src = content.get('Knowledge_of_paper_source', {}).get('value', [])
                            sources = src if isinstance(src, list) else [src]
                            
                            for s in sources:
                                if isinstance(s, str) and any(tok in s.lower() for tok in ("arxiv", "preprint")):
                                    if hasattr(reply, 'number'):
                                        preprinters.append(f"R{reply.number}")
                                    break
                
                # Check for confidential comments
                has_confidential = any(self.is_relevant_comment(reply) for reply in replies)
                
                # Get expected reviews count
                try:
                    reviewers_group = self.client.get_group(f'{self.venue_id}/{self.submission_name}{submission.number}/Reviewers')
                    expected_reviews = len(reviewers_group.members)
                except:
                    # Fallback to assuming at least 3 reviews
                    expected_reviews = max(3, completed_reviews)
            
            # Assemble flags
            if reviewer_ethics:
                suffix = "Reviewer: " + ", ".join(reviewer_ethics)
                ethics_flag = suffix if not ethics_flag else f"{ethics_flag}; {suffix}"
                
            if preprinters:
                preprint_flag = "Yes (" + ", ".join(preprinters) + ")"
            
            # Format scores
            reviewer_confidence = self.format_scores_as_avg_std(confidence_scores)
            reviewer_soundness = self.format_scores_as_avg_std(soundness_scores)
            reviewer_excitement = self.format_scores_as_avg_std(excitement_scores)
            reviewer_overall = self.format_scores_as_avg_std(overall_assessment_scores)
            
            # Format tooltip lists
            confidence_list = " / ".join(f"{s:.1f}" for s in confidence_scores) if confidence_scores else ""
            soundness_list = " / ".join(f"{s:.1f}" for s in soundness_scores) if soundness_scores else ""
            excitement_list = " / ".join(f"{s:.1f}" for s in excitement_scores) if excitement_scores else ""
            overall_list = " / ".join(f"{s:.1f}" for s in overall_assessment_scores) if overall_assessment_scores else ""
            
            # Create paper data
            paper_data = {
                "Paper #": submission.number,
                "Paper ID": submission.id,
                "Title": paper_title,
                "Submission Link": f"{base_url}{submission.id}",
                "Linked Forum": f"{base_url}{linked_forum}" if linked_forum else "",
                "Paper Type": paper_type,
                "Completed Reviews": completed_reviews,
                "Expected Reviews": expected_reviews,
                "Reviewer Confidence": reviewer_confidence,
                "Confidence List": confidence_list,
                "Soundness Score": reviewer_soundness,
                "Soundness List": soundness_list,
                "Excitement Score": reviewer_excitement,
                "Excitement List": excitement_list,
                "Overall Assessment": reviewer_overall,
                "Overall List": overall_list,
                "Meta Review Score": meta_review_score,
                "Recommendation": recommendation,
                "Presentation Mode": presentation_mode,
                "Award": award,
                "Has Confidential": "✓" if has_confidential else "",
                "Ethics Flag": ethics_flag,
                "Preprint Flag": preprint_flag,
                "Response to Meta-review": response_to_meta,
                "Resubmission": prev_url
            }
            
            self.papers_data.append(paper_data)
            processed_count += 1
            
            # Add to score distributions
            if overall_assessment_scores:
                avg_overall = sum(overall_assessment_scores) / len(overall_assessment_scores)
                self.score_distributions['overall_assessment'].append(avg_overall)
                
            if meta_review_score:
                try:
                    meta_score = float(meta_review_score)
                    self.score_distributions['meta_review'].append(meta_score)
                except:
                    pass
        
        print('Done!')
        print(f'Total processed papers: {processed_count}')


    def is_actual_review(self, reply):
        """Check if a reply is an actual review with proper attribute access."""
        try:
            if hasattr(reply, 'invitations'):
                return any('/-/Official_Review' in invitation for invitation in reply.invitations)
            return False
        except Exception:
            return False

    def is_meta_review(self, reply):
        """Check if a reply is a meta-review with proper attribute access."""
        try:
            if hasattr(reply, 'invitations'):
                return any('/-/Meta_Review' in invitation for invitation in reply.invitations)
            return False
        except Exception:
            return False

    def format_scores_as_avg_std(self, scores):
        """Format scores as average with standard deviation."""
        if not scores:
            return ""
        try:
            avg = np.mean(scores)
            std = np.std(scores)
            return f"{avg:.1f} ± {std:.1f}"
        except Exception:
            return ""
        
    def classify_comment_type(self, reply):
        """Determine the type of comment with error handling."""
        try:
            invitations = reply.invitations if hasattr(reply, 'invitations') else []
            
            if any("/-/Review_Issue_Report" in inv for inv in invitations):
                return "Review Issue"
            elif any("/-/Meta-Review_Issue_Report" in inv for inv in invitations):
                return "Meta Review Issue"
            elif any("/-/Author-Editor_Confidential_Comment" in inv for inv in invitations):
                return "Author-Editor Confidential"
            elif any("/-/Comment" in inv for inv in invitations):
                return "Confidential Comment"
            else:
                return "Other"
        except Exception:
            return "Comment"
            
    def process_comments_data(self):
        """
        Process only comments for papers you're assigned as SAC.
        """
        if not self.papers_data:
            print("No papers to process comments for.")
            return
            
        # Build a set of paper numbers for quick lookup
        paper_numbers = {p["Paper #"] for p in self.papers_data}
        
        base_url = "https://openreview.net/forum"
        processed_count = 0

        print("Processing comments...")
        for submission in tqdm(self.submissions, desc="Checking for comments"):
            # We only have SAC papers loaded, but double check
            if submission.number not in paper_numbers:
                continue
                
            # Process comments in the submission replies
            if hasattr(submission, 'details') and 'replies' in submission.details:
                original_replies = submission.details['replies']
                for reply in original_replies:
                    if self.is_relevant_comment(reply):
                        # Safely access attributes
                        forum_id = reply.forum if hasattr(reply, 'forum') else ""
                        note_id = reply.id if hasattr(reply, 'id') else ""
                        replyto = reply.replyto if hasattr(reply, 'replyto') else None
                        signatures = reply.signatures if hasattr(reply, 'signatures') else []
                        tcdate = reply.tcdate if hasattr(reply, 'tcdate') else None
                        
                        link = f"{base_url}?id={forum_id}&noteId={note_id}"
                        role = self.infer_role_from_signature(signatures)
                        date_str = self.format_timestamp(tcdate)

                        comment_text = self.extract_comment_text(reply)
                        
                        self.comments_data.append({
                            "Paper #": submission.number,
                            "Paper ID": submission.id,
                            "Type": self.classify_comment_type(reply),
                            "Role": role,
                            "Date": date_str,
                            "Content": comment_text,
                            "Link": link,
                            "ReplyTo": replyto,
                            "NoteId": note_id,
                        })
                        processed_count += 1
            
            # Check linked forum comments
            if hasattr(submission, 'content') and 'paper_link' in submission.content:
                paper_link = submission.content['paper_link'].get('value', '')
                linked_forum = None
                
                if paper_link:
                    try:
                        parsed = urlparse(paper_link)
                        qs = parse_qs(parsed.query)
                        linked_forum = qs.get("id", qs.get("noteId", [None]))[0]
                        
                        if linked_forum:
                            try:
                                linked_replies = self.client.get_notes(forum=linked_forum)
                                
                                for reply in linked_replies:
                                    if self.is_relevant_comment(reply):
                                        # Safely access attributes 
                                        forum_id = reply.forum if hasattr(reply, 'forum') else ""
                                        note_id = reply.id if hasattr(reply, 'id') else ""
                                        replyto = reply.replyto if hasattr(reply, 'replyto') else None
                                        signatures = reply.signatures if hasattr(reply, 'signatures') else []
                                        tcdate = reply.tcdate if hasattr(reply, 'tcdate') else None
                                        
                                        link = f"{base_url}?id={forum_id}&noteId={note_id}"
                                        role = self.infer_role_from_signature(signatures)
                                        date_str = self.format_timestamp(tcdate)
                                        
                                        comment_text = self.extract_comment_text(reply)

                                        self.comments_data.append({
                                            "Paper #": submission.number,
                                            "Paper ID": submission.id,
                                            "Type": self.classify_comment_type(reply),
                                            "Role": role,
                                            "Date": date_str,
                                            "Content": comment_text,
                                            "Link": link,
                                            "ReplyTo": replyto,
                                            "NoteId": note_id,
                                        })
                                        processed_count += 1
                            except Exception as e:
                                print(f"Error processing comments for linked forum: {e}")
                    except Exception as e:
                        print(f"Error parsing paper_link: {e}")
        
        print(f"Found {processed_count} comments across {len(self.papers_data)} papers.")

    def is_relevant_comment(self, reply):
        """Check if a reply is a relevant comment with proper attribute access."""
        try:
            if hasattr(reply, 'invitations'):
                return any(
                    part in inv
                    for inv in reply.invitations
                    for part in ["/-/Author-Editor_Confidential_Comment", "/-/Comment", 
                                "/-/Review_Issue_Report", "/-/Meta-Review_Issue_Report"]
                )
            return False
        except Exception:
            return False

    def extract_comment_text(self, reply):
        """Extract comment text from a reply with proper error handling."""
        try:
            content = reply.content if hasattr(reply, 'content') else {}
            
            # Try common keys in order of likely importance
            for key in ["comment", "justification", "text", "response", "value"]:
                if key in content:
                    val = content[key]
                    if isinstance(val, dict) and "value" in val:
                        return val["value"]
                    return val
                    
            # If no known key, flatten all text fields into a fallback string
            fallback = []
            for k, v in content.items():
                if isinstance(v, dict) and "value" in v:
                    fallback.append(f"{k}: {v['value']}")
            
            return "\n".join(fallback) if fallback else "(No comment text found)"
        except Exception as e:
            return f"(Error extracting comment text: {e})"

    def infer_role_from_signature(self, signatures):
        """Infer the role from signature with proper error handling."""
        if not signatures:
            return "Unknown"
            
        try:
            sig = signatures[0]  # Use first signature
            
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
        except Exception:
            return "Unknown"
                    
    def compute_ac_meta_data(self):
        """
        Override the AC metadata computation - we don't expose AC identities
        in the commitment phase.
        """
        # We skip computing AC metadata
        self.ac_meta_data = []
                    
    def process_data(self):
        """Process all data needed for the report with better error handling."""
        try:
            self.process_papers_data()
            if hasattr(self, 'compute_correlation_data'):
                self.compute_correlation_data()
            self.process_comments_data()
            # We explicitly don't call compute_ac_meta_data() as we don't want to expose ACs
        except Exception as e:
            print(f"Error in process_data: {e}")
            import traceback
            traceback.print_exc()

    def _get_main_template(self):
        """
        Get the main report template optimized for commitment phase.
        Removes the Area Chair dashboard tab.
        """
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎛️</text></svg>">
    <title>{{ title }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Load jQuery first -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <!-- Then load DataTables -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/datatables/1.10.21/js/jquery.dataTables.min.js"></script>
    <!-- Then load Chart.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <style>
        /* Tab Content */
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        
        /* Scrollbar */
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Enhanced title column handling */
        #papers-table td:nth-child(2) {
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        #papers-table td:nth-child(2) div {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        
        /* Content cell */
        .content-cell {
            max-width: 300px;
            min-width: 200px;
            max-height: 150px;
            overflow-y: auto;
            overflow-x: hidden;
            word-wrap: break-word;
            white-space: normal;
        }
        
        /* DataTables styling */
        table.dataTable {
            width: 100% !important;
            margin: 0 !important;
        }

        table.dataTable thead th {
            position: relative;
            background-image: none !important;
            padding-right: 25px !important;
            cursor: pointer;
        }
        
        table.dataTable thead th.sorting:after,
        table.dataTable thead th.sorting_asc:after,
        table.dataTable thead th.sorting_desc:after {
            position: absolute;
            top: 12px;
            right: 8px;
            display: block;
            font-family: Arial, sans-serif;
            cursor: pointer;
        }
        
        table.dataTable thead th.sorting:after {
            content: "↕";
            color: #CCC;
            font-size: 0.8em;
        }

        
        table.dataTable thead th.sorting_asc:after {
            content: "↑";
            opacity: 1;
        }
        
        table.dataTable thead th.sorting_desc:after {
            content: "↓";
            opacity: 1;
        }

        
        table.dataTable th.overall-assessment {
            min-width: 200px;
        }
        
        table.dataTable th.meta-review {
            min-width: 100px;
        }
        
        .dataTables_wrapper .dataTables_length, 
        .dataTables_wrapper .dataTables_filter {
            margin-bottom: 10px;
        }
        
        .dataTables_wrapper .dataTables_length select {
            border: 1px solid #e5e7eb;
            padding: 0.375rem 2.25rem 0.375rem 0.75rem;
            font-size: 0.875rem;
            line-height: 1.25rem;
            border-radius: 0.375rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            background-color: #fff;
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
            background-position: right 0.5rem center;
            background-repeat: no-repeat;
            background-size: 1.5em 1.5em;
            appearance: none;
        }
        
        .dataTables_wrapper .dataTables_filter input {
            border: 1px solid #e5e7eb;
            border-radius: 0.375rem;
            padding: 0.375rem 0.75rem;
            font-size: 0.875rem;
            line-height: 1.25rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-left: 0.5rem;
        }
        
        .dataTables_wrapper .dataTables_paginate .paginate_button {
            padding: 0.25rem 0.5rem;
            margin: 0 0.25rem;
            border-radius: 0.375rem;
            color: #4f46e5 !important;
            border: 1px solid #e5e7eb;
            background: white;
            line-height: 1.25;
            cursor: pointer;
            font-size: small;
        }

        .dataTables_wrapper .dataTables_paginate .paginate_button.disabled {
            cursor: default;
        }
        
        .dataTables_wrapper .dataTables_paginate .paginate_button.current {
            background: #4f46e5 !important;
            color: white !important;
            border: 1px solid #4f46e5;
        }
        
        .dataTables_wrapper .dataTables_paginate .paginate_button:hover:not(.current) {
            background: #f3f4f6 !important;
            color: #4f46e5 !important;
            border: 1px solid #e5e7eb;
        }
        
        .dataTables_wrapper .dataTables_info {
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            font-size: 0.875rem;
            color: #999;
            text-align: center;
            border-top: 1px solid rgb(229, 231, 235);
            font-size: small;
        }

        .dataTables_wrapper {
            width: 100% !important;
            overflow-x: auto;
            overflow-y: hidden;
        }

        .dataTables_wrapper .dataTables_scroll {
            width: 100% !important;
        }

        .dataTables_wrapper .dataTables_scrollBody {
            width: 100% !important;
        }
        
        .dataTables_scrollBody {
            min-height: 300px; /* Minimum table body height */
        }
        
        /* Comment content */
        .comment-content {
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        #papers-table_paginate{
            margin-bottom: 10px;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="relative">
        <!-- Header -->
        <header class="bg-slate-600 bg-slate-600 shadow-md">
            <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
                <h1 class="text-3xl font-bold text-white">
                    🎛️ ARR Commitment Dashboard
                </h1>
                <div class="text-right">
                    <p class="text-white text-sm">{{ venue_id }}</p>
                    <p class="text-white text-sm opacity-80">Generated at {{ generated_date }}</p>
                </div>
            </div>
        </header>

        <!-- Navigation -->
        <nav class="bg-white shadow-md">
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div class="flex justify-center">
                    <ul class="flex space-x-8">
                        <li>
                            <button class="tab-button py-4 px-1 text-center border-b-2 border-transparent hover:border-indigo-500 font-medium text-gray-500 hover:text-gray-700 transition-colors duration-200 focus:outline-none active" data-tab="papers-tab">
                                Papers Overview
                            </button>
                        </li>
                        <li>
                            <button class="tab-button py-4 px-1 text-center border-b-2 border-transparent hover:border-indigo-500 font-medium text-gray-500 hover:text-gray-700 transition-colors duration-200 focus:outline-none" data-tab="comments-tab">
                                Comments ({{ comments_count }})
                            </button>
                        </li>
                        <li>
                            <button class="tab-button py-4 px-1 text-center border-b-2 border-transparent hover:border-indigo-500 font-medium text-gray-500 hover:text-gray-700 transition-colors duration-200 focus:outline-none" data-tab="analytics-tab">
                                Analytics
                            </button>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <!-- Papers Tab -->
            <div id="papers-tab" class="tab-content active">
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Papers Overview</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Complete status of all papers in your batch, including review scores and meta-review status.
                        </p>
                    </div>
                    {% include 'components/papers_table.html' %}
                </div>
            </div>

            <!-- Comments Tab -->
            <div id="comments-tab" class="tab-content">
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Comments & Review Issues</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            All confidential comments and review issue reports.
                        </p>
                    </div>
                    {% include 'components/comments.html' %}
                </div>
            </div>

            <!-- Analytics Tab -->
            <div id="analytics-tab" class="tab-content">
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Score Distribution</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Distribution of overall assessment and meta-review scores.
                        </p>
                    </div>
                    {% include 'components/score_distribution.html' %}
                </div>
                
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Paper Type Distribution</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Distribution of papers by type.
                        </p>
                    </div>
                    {% include 'components/paper_type_distribution.html' %}
                </div>
                
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Score Comparison: Overall vs Meta</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Comparison between average overall assessment scores and meta-review scores.
                        </p>
                    </div>
                    {% include 'components/score_scatter.html' %}
                </div>

                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 pt-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Score Correlation Matrix</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Correlation matrix between different types of scores.
                        </p>
                    </div>
                    {% include 'components/correlation_matrix.html' %}
                </div>

            </div>

        </main>
    </div>

    <footer>
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-4">
                <div class="flex justify-center">
                    <span class="text-sm text-gray-500">
                        Code at <a class="text-gray-500 hover:text-gray-700" href="https://github.com/mtreviso/arr-report-generator" target="_blank">github.com/mtreviso/arr-report-generator</a>
                    </span> 
                </div>
            </div>
        <div>
    </footer>

    <!-- JavaScript for Functionality -->
    <script>
        // Tab functionality
        document.addEventListener('DOMContentLoaded', function() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');

            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    // Remove active class from all buttons and contents
                    tabButtons.forEach(btn => btn.classList.remove('active', 'border-indigo-500', 'text-indigo-600'));
                    tabButtons.forEach(btn => btn.classList.add('border-transparent', 'text-gray-500'));
                    tabContents.forEach(content => content.classList.remove('active'));

                    // Add active class to the clicked button and corresponding content
                    button.classList.add('active', 'border-indigo-500', 'text-indigo-600');
                    button.classList.remove('border-transparent', 'text-gray-500');
                    const tabId = button.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                    
                    // If analytics tab is selected, make sure charts are properly rendered
                    if (tabId === 'analytics-tab') {
                        setTimeout(() => {
                            if (window.scoreChart) {
                                window.scoreChart.resize();
                            }
                        }, 10);
                    }
                });
            });

            // Initialize DataTables for papers table
            $(document).ready(function() {
                // Initialize DataTables for papers table
                $('#papers-table').DataTable({
                    pageLength: 100,
                    autoWidth: false,
                    scrollX: true,
                    order: [[0, 'asc']],
                    columnDefs: [
                        { type: 'num', targets: [0] },
                        { className: 'overall-assessment', targets: 7 },
                        { className: 'meta-review', targets: 8 }
                    ],
                    language: {
                        search: "_INPUT_",
                        searchPlaceholder: "Search papers...",
                        lengthMenu: "_MENU_"
                    },
                    initComplete: function() {
                        // Add custom filtering for Paper Type column
                        this.api().columns(3).every(function() {
                            const column = this;
                            const select = $('<select id="paper-type-filter" class="bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full"><option value="">All Types</option></select>')
                                .appendTo('#paper-type-filter')
                                .on('change', function() {
                                    const val = $.fn.dataTable.util.escapeRegex($(this).val());
                                    column.search(val ? '^' + val + '$' : '', true, false).draw();
                                });

                            column.data().unique().sort().each(function(d, j) {
                                select.append('<option value="' + d + '">' + d + '</option>');
                            });
                        });
                        
                        // Move the DataTables controls to our custom containers
                        $('#papers-table_length').detach().appendTo('#papers-length-container');
                        $('#papers-table_filter').detach().appendTo('#papers-search-container');
                        
                        // Style the moved controls
                        $('#papers-table_length select').addClass('bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full');
                        $('#papers-table_filter input').addClass('bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full');
                        
                        // Remove the label text and keep just the select element
                        $('#papers-table_length label').contents().filter(function() {
                            return this.nodeType === 3; // Text nodes
                        }).remove();
                    }
                });

                // Comments table
                $('#comments-table').DataTable({
                    pageLength: 100,
                    autoWidth: false,
                    scrollX: false,
                    order: [[0, 'asc'], [2, 'asc']],
                    columnDefs: [
                        { type: 'num', targets: [0] },
                        { 
                            targets: 4, 
                            className: 'content-cell custom-scrollbar',
                            width: '300px'
                        }
                    ],
                    language: {
                        search: "_INPUT_",
                        searchPlaceholder: "Search comments...",
                        lengthMenu: "Show _MENU_ comments"
                    }
                });

            });

            // Initialize Markdown rendering for comments
            document.querySelectorAll('.comment-content').forEach(function(element) {
                element.innerHTML = marked.parse(element.textContent);
            });

            // Initialize and render charts
            initAllAnalyticsCharts();
        });

        // Initialize all analytics charts when tab is clicked
        function initAllAnalyticsCharts() {
            // Score distribution chart
            renderScoreDistributionChart();
            
            // Paper type chart
            if (typeof initPaperTypeChart === 'function') {
                initPaperTypeChart();
            }
            
            // Score comparison charts
            if (typeof initScoreCharts === 'function') {
                initScoreCharts();
            }
        }

        // Chart rendering function
        function renderScoreDistributionChart() {
            const ctx = document.getElementById('score-distribution-chart').getContext('2d');

            // Data from Jinja template
            const overallCounts = {{ histogram_data.overall_assessment.counts | tojson }};
            const metaCounts = {{ histogram_data.meta_review.counts | tojson }};
            const overallLabels = {{ histogram_data.overall_assessment.labels | tojson }};
            const metaLabels = {{ histogram_data.meta_review.labels | tojson }};

            window.scoreChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: overallLabels, // Use overall labels for x-axis
                    datasets: [
                        {
                            label: 'Overall Assessment',
                            data: overallCounts,
                            borderColor: 'rgba(59, 130, 246, 1)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgba(59, 130, 246, 1)',
                            tension: 0.1,
                            fill: true
                        },
                        {
                            label: 'Meta Review Score',
                            data: metaLabels.map((label, i) => ({
                                x: label, // Use meta labels for x values
                                y: metaCounts[i] // Use meta counts for y values
                            })),
                            borderColor: 'rgba(220, 38, 38, 1)',
                            backgroundColor: 'rgba(220, 38, 38, 0.1)',
                            pointBackgroundColor: 'rgba(220, 38, 38, 1)',
                            pointBorderColor: '#fff',
                            pointHoverBackgroundColor: '#fff',
                            pointHoverBorderColor: 'rgba(220, 38, 38, 1)',
                            tension: 0.1,
                            fill: true,
                            parsing: {
                                xAxisKey: 'x',
                                yAxisKey: 'y'
                            }
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2.5,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                title: function(context) {
                                    // Handle both formats of data points
                                    const dataPoint = context[0].raw;
                                    const score = typeof dataPoint === 'object' ? dataPoint.x : overallLabels[context[0].dataIndex];
                                    return `Score: ${score}`;
                                },
                                label: function(context) {
                                    // Handle both formats of data points
                                    const value = typeof context.raw === 'object' ? context.raw.y : context.raw;
                                    return `${context.dataset.label}: ${value} papers`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Papers'
                            },
                            ticks: {
                                stepSize: 1
                            }
                        },
                        x: {
                            type: 'linear', // Use linear scale to position points correctly
                            title: {
                                display: true,
                                text: 'Score'
                            }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>'''

    def _get_papers_table_template(self):
        """
        Get the papers table template with all requested columns.
        """
        return '''<div class="px-4 py-5 sm:p-6">
    <div class="mb-6 bg-gray-50 p-4 rounded-lg" id="papers-filter-container">
        <h3 class="text-lg font-medium text-gray-900 mb-3">Filter Papers</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
                <label for="paper-type-filter" class="block text-sm font-medium text-gray-700 mb-1">Paper Type</label>
                <div id="paper-type-filter"></div>
            </div>
            <div id="papers-length-container">
                <label class="block text-sm font-medium text-gray-700 mb-1">Papers per page</label>
                <!-- DataTables length control will be moved here -->
            </div>
            <div id="papers-search-container">
                <label class="block text-sm font-medium text-gray-700 mb-1">Search</label>
                <!-- DataTables search control will be moved here -->
            </div>
        </div>
    </div>
    <div class="overflow-x-auto w-full" style="max-width: 100%;">
        <table id="papers-table" class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 40px; width: 40px;">
                        #
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 300px; width: 300px;">
                        Title
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 60px; width: 60px;">
                        Type
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 80px; width: 80px;">
                        <span title="Completed / Expected">Reviews</span>
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 90px; width: 90px;">
                        Confidence
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 90px; width: 90px;">
                        Soundness
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 90px; width: 90px;">
                        Excitement
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 90px; width: 90px;">
                        Overall
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 70px; width: 70px;">
                        Meta
                    </th>
                    <th scope="col" class="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 50px; width: 50px;">
                        Conf.
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 70px; width: 70px;">
                        Ethics
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 70px; width: 70px;">
                        Preprint
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 100px; width: 100px;">
                        Recommendation
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 120px; width: 120px;">
                        Presentation
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 90px; width: 90px;">
                        Award
                    </th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                {% for paper in papers %}
                <tr>
                    <td class="px-2 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                        {{ paper["Paper #"] }}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500" style="max-width: 300px;">
                        <div class="truncate" title="{{ paper.Title }}">
                            <a href="{{ paper['Submission Link'] }}" target="_blank" class="text-indigo-600 hover:text-indigo-900">{{ paper.Title }}</a>
                        </div>
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {{ paper["Paper Type"] }}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500 text-center">
                        {{ paper["Completed Reviews"] }}/{{ paper["Expected Reviews"] }}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500">
                        {% if paper["Reviewer Confidence"] %}
                        <span title="{{ paper["Confidence List"] }}">{{ paper["Reviewer Confidence"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500">
                        {% if paper["Soundness Score"] %}
                        <span title="{{ paper["Soundness List"] }}">{{ paper["Soundness Score"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500">
                        {% if paper["Excitement Score"] %}
                        <span title="{{ paper["Excitement List"] }}">{{ paper["Excitement Score"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500">
                        {% if paper["Overall Assessment"] %}
                        <span title="{{ paper["Overall List"] }}">{{ paper["Overall Assessment"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Meta Review Score"] %}
                        <span class="font-semibold">{{ paper["Meta Review Score"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500 text-center">
                        {% if paper["Has Confidential"] %}
                        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            ✓
                        </span>
                        {% else %}
                        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            -
                        </span>
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Ethics Flag"] %}
                        <span class="text-red-600">{{ paper["Ethics Flag"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Preprint Flag"] %}
                        <span class="text-amber-600">{{ paper["Preprint Flag"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Recommendation"] %}
                        <span class="font-semibold">{{ paper["Recommendation"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Presentation Mode"] %}
                        {{ paper["Presentation Mode"] }}
                        {% else %}
                        -
                        {% endif %}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {% if paper["Award"] %}
                        <span class="text-yellow-600 font-semibold">{{ paper["Award"] }}</span>
                        {% else %}
                        -
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
'''

    def _create_template_files(self, template_dir):
        """
        Create the template files for commitment phase report.
        Overriding the parent method to exclude AC dashboard template.
        """
        # Create main report template
        report_template = template_dir / "report.html"
        with open(report_template, "w", encoding="utf-8") as f:
            f.write(self._get_main_template())
                
        # Create components templates
        components_dir = template_dir / "components"
        if not components_dir.exists():
            components_dir.mkdir()
            
        # Create papers table component
        papers_table = components_dir / "papers_table.html"
        with open(papers_table, "w", encoding="utf-8") as f:
            f.write(self._get_papers_table_template())
                
        # Create comments component
        comments = components_dir / "comments.html"
        with open(comments, "w", encoding="utf-8") as f:
            f.write(self._get_comments_template())
                
        # Create score distribution component
        score_distribution = components_dir / "score_distribution.html"
        with open(score_distribution, "w", encoding="utf-8") as f:
            f.write(self._get_score_distribution_template())
                
        # Create correlation matrix component
        correlation_matrix = components_dir / "correlation_matrix.html"
        with open(correlation_matrix, "w", encoding="utf-8") as f:
            f.write(self._get_correlation_matrix_template())
                
        # Create paper type distribution component
        paper_type_distribution = components_dir / "paper_type_distribution.html"
        with open(paper_type_distribution, "w", encoding="utf-8") as f:
            f.write(self._get_paper_type_distribution_template())
                
        # Create score scatter component
        score_scatter = components_dir / "score_scatter.html"
        with open(score_scatter, "w", encoding="utf-8") as f:
            f.write(self._get_score_scatter_template())
        
        # Note: We deliberately skip creating ac_dashboard.html

    def generate_report(self, output_dir="."):
        """
        Generate the HTML report with proper error handling.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        # Process data with error handling
        try:
            # Basic data check
            if not self.submissions:
                print("No submissions found for your AC ID.")
                error_html = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>Error - No Papers Found</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 2em; }}
                        .error {{ color: red; padding: 1em; border: 1px solid #ccc; }}
                    </style>
                </head>
                <body>
                    <h1>No Papers Found</h1>
                    <div class="error">
                        <p>No papers were found where you are assigned as a Senior Area Chair.</p>
                        <p>Please check:</p>
                        <ul>
                            <li>Your OpenReview ID: {self.me}</li>
                            <li>Venue: {self.venue_id}</li>
                            <li>That you're assigned as an Area Chair for this venue</li>
                        </ul>
                        <p>This could also happen if you don't have access to view any papers yet.</p>
                    </div>
                </body>
                </html>"""
                
                output_path = Path(output_dir) / "commitment_report.html"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(error_html)
                    
                return output_path
            
            # Process data
            self.process_data()
            
            # Check again after processing
            if not self.papers_data:
                print("No paper data was generated during processing.")
                error_html = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>Error - No Paper Data Generated</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 2em; }}
                        .error {{ color: red; padding: 1em; border: 1px solid #ccc; }}
                    </style>
                </head>
                <body>
                    <h1>No Paper Data Generated</h1>
                    <div class="error">
                        <p>The system found {len(self.submissions)} papers assigned to you, but no paper data could be processed.</p>
                        <p>This might be due to:</p>
                        <ul>
                            <li>All papers being withdrawn or desk rejected</li>
                            <li>Issues accessing the paper content</li>
                        </ul>
                        <p>Please contact the conference organizers if this issue persists.</p>
                    </div>
                </body>
                </html>"""
                
                output_path = Path(output_dir) / "commitment_report.html"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(error_html)
                    
                return output_path
        
            # Generate analytics data
            paper_type_distribution = self.generate_paper_type_distribution()
            score_scatter_data = self.generate_score_scatter_data()
            
            # Prepare data for templates
            template_data = {
                "title": f"Commitment Phase Report: {self.venue_id}",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "venue_id": self.venue_id,
                "papers": self.papers_data,
                "comments_count": len(self.comments_data),
                "comments": self.comments_data,
                "comment_trees": self.organize_comments_by_paper(),
                "histogram_data": self.generate_histogram_data(),
                "correlation_data": self.correlation_data,
                "paper_type_distribution": paper_type_distribution,
                "score_scatter_data": score_scatter_data
            }
            
            # Configure Jinja2 environment
            template_dir = Path(__file__).parent / "templates"
            if not template_dir.exists():
                template_dir = Path("templates")
                if not template_dir.exists():
                    template_dir.mkdir()
                    
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
            
            # Create template files
            self._create_template_files(template_dir)
            
            # Render the main template
            template = env.get_template("report.html")
            html_content = template.render(**template_data)

            # Fix the pre/code issue in comments with a post-processing step
            html_content = self._fix_comment_formatting(html_content)
            
            # Write the HTML file
            output_path = Path(output_dir) / "commitment_report.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            return output_path
        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple error page
            error_html = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>Error Generating Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 2em; }}
                    .error {{ color: red; padding: 1em; border: 1px solid #ccc; }}
                    pre {{ background: #f8f8f8; padding: 1em; overflow: auto; }}
                </style>
            </head>
            <body>
                <h1>Error Generating Report</h1>
                <div class="error">
                    <p>An error occurred while generating the report:</p>
                    <pre>{e}</pre>
                    <p>Please check your OpenReview ID: {self.me}</p>
                    <p>Venue: {self.venue_id}</p>
                </div>
            </body>
            </html>"""
            
            output_path = Path(output_dir) / "commitment_report_error.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(error_html)
                
            return output_path

    def generate_histogram_data(self):
        """Generate histogram data for review scores with error handling."""
        try:
            # Define bins for scores
            bins_overall = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 
                            4.0, 4.25, 4.5, 4.75, 5.0, 5.25]
            bins_meta = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

            # Use bin starts as labels
            bin_labels_overall = [f"{x:.2f}" for x in bins_overall[:-1]]
            bin_labels_meta = [f"{x:.2f}" for x in bins_meta[:-1]]
            
            histogram_data = {
                'overall_assessment': {
                    'counts': [],
                    'labels': bin_labels_overall
                },
                'meta_review': {
                    'counts': [],
                    'labels': bin_labels_meta
                }
            }
            
            # Compute histogram for overall assessment
            if self.score_distributions['overall_assessment']:
                hist, _ = np.histogram(
                    self.score_distributions['overall_assessment'], 
                    bins=bins_overall
                )
                histogram_data['overall_assessment']['counts'] = hist.tolist()
                
            # Compute histogram for meta review
            if self.score_distributions['meta_review']:
                hist, _ = np.histogram(
                    self.score_distributions['meta_review'], 
                    bins=bins_meta
                )
                histogram_data['meta_review']['counts'] = hist.tolist()
                
            return histogram_data
        except Exception as e:
            print(f"Error generating histogram data: {e}")
            # Return empty histogram data to avoid breaking template
            return {
                'overall_assessment': {'counts': [], 'labels': []},
                'meta_review': {'counts': [], 'labels': []}
            }


def main():
    parser = argparse.ArgumentParser(description='Generate Commitment Phase Report')
    parser.add_argument('--username', required=True, help='OpenReview username')
    parser.add_argument('--password', required=True, help='OpenReview password')
    parser.add_argument('--venue_id', required=True, help='Venue ID (e.g., aclweb.org/ACL/2025/Conference)')
    parser.add_argument('--me', required=True, help='Your OpenReview ID (e.g., ~Your_Name1)')
    parser.add_argument('--output_dir', default='.', help='Output directory for the report')
    
    args = parser.parse_args()
    
    generator = CommitmentReportGenerator(
        username=args.username,
        password=args.password,
        venue_id=args.venue_id,
        me=args.me
    )
    
    report_path = generator.generate_report(output_dir=args.output_dir)
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()