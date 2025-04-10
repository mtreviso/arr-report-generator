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
    def __init__(self, username, password, venue_id, me):
        self.username = username
        self.password = password
        self.venue_id = venue_id
        self.me = me
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
                    # Check if the user is a member of this group
                    if hasattr(g, 'members') and me in g.members:
                        self.my_sac_groups.add(g.id)

        # Data containers
        self.papers_data = []
        self.ac_meta_data = None
        self.comments_data = []
        
        # Score distributions for visualization
        self.score_distributions = {
            'overall_assessment': [],
            'meta_review': []
        }
        
        # Correlation data
        self.correlation_data = None
        
    def is_actual_review(self, reply):
        """Count a reply as an actual review only if its invitations include "/-/Official_Review"."""
        return any('/-/Official_Review' in invitation for invitation in reply.get('invitations', []))

    def is_meta_review(self, reply):
        """Check if a reply is a meta-review."""
        return any('/-/Meta_Review' in invitation for invitation in reply.get('invitations', []))

    def is_withdrawn(self, submission):
        # Check if there's a non-empty withdrawal_confirmation field.
        withdrawal_conf = submission.content.get("withdrawal_confirmation", {}).get("value", "").strip()
        if withdrawal_conf:
            return True
        # Alternatively, check if the venue value contains "withdrawn" (case insensitive).
        venue_val = submission.content.get("venue", {}).get("value", "").lower()
        if "withdrawn" in venue_val:
            return True
        return False

    def format_scores_as_list(self, scores):
        """Format a list of scores into "avg_score (score1 / score2 / ...)" with 1 decimal precision."""
        if scores:
            avg = sum(scores) / len(scores)
            score_list = " / ".join(f"{s:.1f}" for s in scores)
            return f"{avg:.1f} ({score_list})"
        else:
            return ""

    def format_scores_as_avg_std(self, scores):
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            return f"{avg:.1f} ± {std:.1f}"
        else:
            return ""
        
    def parse_avg(self, s):
        """Extract the average score from a formatted string."""
        try:
            return float(s.split()[0])
        except Exception:
            return float('nan')

    def parse_meta_review(self, s):
        """Parse a meta review score into a float."""
        try:
            return float(s)
        except Exception:
            return float('nan')
        
    def is_relevant_comment(self, reply):
        """Check if a reply is a relevant comment type."""
        invitations = reply.get("invitations", [])
        return any(
            part in inv
            for inv in invitations
            for part in ["/-/Author-Editor_Confidential_Comment", "/-/Comment", "/-/Review_Issue_Report"]
        )

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
        """Extract comment text from a reply."""
        content = reply.get("content", {})
        # Try common keys in order of likely importance
        for key in ["comment", "justification", "text", "response", "value"]:
            if key in content:
                val = content[key]
                return val.get("value") if isinstance(val, dict) else val
        # If no known key, flatten all text fields into a fallback string
        fallback = []
        for k, v in content.items():
            if isinstance(v, dict) and "value" in v:
                fallback.append(f"{k}: {v['value']}")
        return "\n".join(fallback) if fallback else "(No comment text found)"

    def infer_role_from_signature(self, signatures):
        """Infer the role from signature."""
        if not signatures:
            return "Unknown"
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

    def format_timestamp(self, ms_since_epoch):
        """Format timestamp to readable date."""
        if not ms_since_epoch:
            return ""
        dt = datetime.fromtimestamp(ms_since_epoch / 1000)
        return dt.strftime("%Y-%m-%d")

    def process_papers_data(self):
        """Process all papers data."""
        for submission in tqdm(self.submissions):
            # Skip withdrawn or desk rejected papers.
            if self.is_withdrawn(submission):
                print(f"Skipping withdrawn paper: {submission.id}")
                continue
            if "venue" in submission.content and "desk rejected" in submission.content["venue"]["value"].lower():
                print(f"Skipping desk rejected paper: {submission.id}")
                continue

            prefix = f'{self.venue_id}/{self.submission_name}{submission.number}'
            # Process only submissions in your SAC batch
            if not (set(submission.readers) & self.my_sac_groups):
                continue

            # Retrieve the assigned Area Chair
            try:
                area_chairs_group = self.client.get_group(f'{prefix}/Area_Chairs')
                if area_chairs_group.members:
                    ac = area_chairs_group.members[0]  # Assuming one AC per paper
                else:
                    continue
            except Exception:
                continue

            # Extract Paper Type (e.g., "Long" or "Short")
            paper_type = submission.content.get("paper_type", {}).get("value", "")

            # Count the number of completed reviews
            completed_reviews = sum(1 for reply in submission.details["replies"] if self.is_actual_review(reply))

            # Determine the expected number of reviews from the Reviewers group
            expected_reviews = 0
            try:
                reviewers_group = self.client.get_group(f'{prefix}/Reviewers')
                expected_reviews = len(reviewers_group.members)
            except Exception:
                expected_reviews = 0

            # Set review status: Checkmark if the paper has three or more completed reviews
            status = "✓" if completed_reviews >= 3 else ""

            # Extract meta-review score from the meta-review reply, if available
            meta_review_score = ""
            for reply in submission.details["replies"]:
                if self.is_meta_review(reply):
                    content = reply.get("content", {})
                    if "overall_assessment" in content:
                        meta_review_score = content["overall_assessment"].get("value", "")
                    elif "overall_rating" in content:
                        meta_review_score = content["overall_rating"].get("value", "")
                    elif "score" in content:
                        meta_review_score = content["score"].get("value", "")
                    break

            # Initialize lists to collect reviewer scores
            confidence_scores = []
            soundness_scores = []
            excitement_scores = []
            overall_assessment_scores = []

            # Aggregate reviewer scores from review replies
            for reply in submission.details["replies"]:
                if self.is_actual_review(reply):
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

            # Store both the summary and the full list for tooltips
            reviewer_confidence = self.format_scores_as_avg_std(confidence_scores)
            reviewer_soundness = self.format_scores_as_avg_std(soundness_scores)
            reviewer_excitement = self.format_scores_as_avg_std(excitement_scores)
            reviewer_overall = self.format_scores_as_avg_std(overall_assessment_scores)
            
            # Format full lists for tooltip display
            confidence_list = " / ".join(f"{s:.1f}" for s in confidence_scores) if confidence_scores else ""
            soundness_list = " / ".join(f"{s:.1f}" for s in soundness_scores) if soundness_scores else ""
            excitement_list = " / ".join(f"{s:.1f}" for s in excitement_scores) if excitement_scores else ""
            overall_list = " / ".join(f"{s:.1f}" for s in overall_assessment_scores) if overall_assessment_scores else ""
            
            # Get paper title
            paper_title = submission.content.get("title", {}).get("value", "Untitled")

            paper_data = {
                "Paper #": submission.number,
                "Paper ID": submission.id,
                "Title": paper_title,
                "Paper Type": paper_type,
                "Area Chair": ac,
                "Completed Reviews": completed_reviews,
                "Expected Reviews": expected_reviews,
                "Ready for Rebuttal": status,
                "Reviewer Confidence": reviewer_confidence,
                "Confidence List": confidence_list,
                "Soundness Score": reviewer_soundness,
                "Soundness List": soundness_list,
                "Excitement Score": reviewer_excitement,
                "Excitement List": excitement_list,
                "Overall Assessment": reviewer_overall,
                "Overall List": overall_list,
                "Meta Review Score": meta_review_score
            }
            
            self.papers_data.append(paper_data)
            
            # Add to score distributions if valid scores exist
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
            Meta_Reviews_Num=("Meta Review Score", lambda x: (x != "").sum())
        ).reset_index()

        # Add new column to show whether all papers are done (i.e. ready)
        meta_df["All Reviews Ready"] = meta_df.apply(
            lambda row: "✓" if row["Papers_Ready"] == row["Num_Papers"] else "", axis=1
        )

        # Add new column to show whether all meta reviews are done
        meta_df["All Meta-reviews Ready"] = meta_df.apply(
            lambda row: "✓" if row["Meta_Reviews_Num"] == row["Num_Papers"] else "", axis=1
        )

        # Format Meta_Reviews_Done column as "x of y"
        meta_df["Meta_Reviews_Done"] = meta_df.apply(
            lambda row: f"{row['Meta_Reviews_Num']} of {row['Num_Papers']}", axis=1
        )

        # Optionally drop the temporary numeric column
        meta_df.drop(columns=["Meta_Reviews_Num"], inplace=True)

        # Convert to list of dictionaries for the template
        self.ac_meta_data = meta_df.to_dict(orient='records')
        
    def compute_correlation_data(self):
        """Compute correlation between different review scores."""
        if not self.papers_data:
            return
            
        df = pd.DataFrame(self.papers_data)
        
        # Build a temporary DataFrame with numeric values from the aggregated string columns
        corr_data = pd.DataFrame({
            "Overall_Assessment_Avg": df["Overall Assessment"].apply(self.parse_avg),
            "Reviewer_Confidence_Avg": df["Reviewer Confidence"].apply(self.parse_avg),
            "Soundness_Score_Avg": df["Soundness Score"].apply(self.parse_avg),
            "Excitement_Score_Avg": df["Excitement Score"].apply(self.parse_avg),
            # For Meta Review Score, try converting directly to float if possible
            "Meta_Review_Score": df["Meta Review Score"].apply(
                lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip() != "") else np.nan
            )
        })

        # Compute the correlation matrix
        corr_table = corr_data.corr().fillna(0).round(2)
        
        # Save the correlation data for additional analytics charts
        self.score_correlation_data = corr_data
        
        # Format the correlation matrix for the template
        correlation_matrix = []
        labels = corr_table.index.tolist()
        
        for i, row_label in enumerate(labels):
            row_data = []
            for j, col_label in enumerate(labels):
                value = corr_table.iloc[i, j]
                # Only include upper triangle
                if j > i:  
                    row_data.append({
                        'value': value,
                        'color': self._get_correlation_color(value)
                    })
                else:
                    row_data.append(None)  # Lower triangle
            correlation_matrix.append({
                'label': row_label.replace('_', ' '),
                'cells': row_data  # Changed from 'values' to 'cells'
            })
            
        self.correlation_data = {
            'labels': [label.replace('_', ' ') for label in labels],
            'matrix': correlation_matrix
        }

    def generate_paper_type_distribution(self):
        """Generate data for paper type distribution chart."""
        if not self.papers_data:
            return {}
        
        df = pd.DataFrame(self.papers_data)
        
        # Count papers by type
        type_counts = df["Paper Type"].value_counts().to_dict()
        
        return {
            'labels': list(type_counts.keys()),
            'counts': list(type_counts.values())
        }

    def generate_review_completion_data(self):
        """Generate data for review completion visualization."""
        if not self.ac_meta_data:
            return {}
        
        # Extract completion percentages from AC meta data
        ac_completion = []
        
        for ac in self.ac_meta_data:
            completed = ac.get("Completed_Reviews", 0)
            expected = ac.get("Expected_Reviews", 0)
            
            percentage = 0
            if expected > 0:
                percentage = round((completed / expected) * 100)
                
            ac_completion.append({
                'name': ac.get("Area Chair", "Unknown"),
                'percentage': percentage
            })
        
        # Sort by percentage descending
        ac_completion.sort(key=lambda x: x['percentage'], reverse=True)
        
        return ac_completion

    def generate_ac_scoring_data(self):
        """Generate data for AC scoring analysis."""
        if not self.papers_data:
            return {}
        
        df = pd.DataFrame(self.papers_data)
        
        # Group by Area Chair and compute average scores
        ac_scores = {}
        
        for ac_name in df['Area Chair'].unique():
            ac_papers = df[df['Area Chair'] == ac_name]
            
            # Parse scores for this AC's papers
            overall_scores = [self.parse_avg(score) for score in ac_papers['Overall Assessment'] if score]
            meta_scores = [self.parse_meta_review(score) for score in ac_papers['Meta Review Score'] if score]
            
            # Filter out NaN values
            overall_scores = [s for s in overall_scores if not np.isnan(s)]
            meta_scores = [s for s in meta_scores if not np.isnan(s)]
            
            if overall_scores or meta_scores:
                ac_scores[ac_name] = {
                    'overall_avg': np.mean(overall_scores) if overall_scores else None,
                    'meta_avg': np.mean(meta_scores) if meta_scores else None,
                    'overall_count': len(overall_scores),
                    'meta_count': len(meta_scores),
                    'difference': np.mean(meta_scores) - np.mean(overall_scores) if meta_scores and overall_scores else None
                }
        
        # Convert to list format for template
        result = []
        for ac_name, scores in ac_scores.items():
            result.append({
                'name': ac_name,
                'overall_avg': round(scores['overall_avg'], 2) if scores['overall_avg'] is not None else None,
                'meta_avg': round(scores['meta_avg'], 2) if scores['meta_avg'] is not None else None,
                'overall_count': scores['overall_count'],
                'meta_count': scores['meta_count'],
                'difference': round(scores['difference'], 2) if scores['difference'] is not None else None
            })
        
        # Sort by largest meta-review/overall difference
        result.sort(key=lambda x: abs(x['difference']) if x['difference'] is not None else 0, reverse=True)
        
        return result

    def generate_score_scatter_data(self):
        """Generate data for scatter plot between scores."""
        if not hasattr(self, 'score_correlation_data') or self.score_correlation_data is None:
            return {}
        
        df = self.score_correlation_data
        
        # For each paper, create a data point with Overall Assessment and Meta Review
        scatter_data = []
        
        for i, row in df.iterrows():
            if not np.isnan(row["Overall_Assessment_Avg"]) and not np.isnan(row["Meta_Review_Score"]):
                scatter_data.append({
                    'x': float(row["Overall_Assessment_Avg"]),
                    'y': float(row["Meta_Review_Score"]),
                    'paper': i  # This is the index for reference
                })
        
        # Create a histogram of score differences
        differences = []
        
        for point in scatter_data:
            diff = point['y'] - point['x']
            differences.append(diff)
        
        # Count score differences
        diff_counts = {}
        for diff in differences:
            # Round to nearest 0.5
            rounded_diff = round(diff * 2) / 2
            diff_counts[rounded_diff] = diff_counts.get(rounded_diff, 0) + 1
        
        # Convert to sorted list of tuples
        diff_data = sorted([(k, v) for k, v in diff_counts.items()])
        
        return {
            'scatter': scatter_data,
            'differences': {
                'labels': [str(diff[0]) for diff in diff_data],
                'counts': [diff[1] for diff in diff_data]
            }
        }
    
    def _get_correlation_color(self, value):
        """Get a color based on correlation value."""
        if abs(value) < 0.3:
            return 'bg-gray-100'
        elif abs(value) < 0.5:
            return 'bg-blue-100'
        elif abs(value) < 0.7:
            return 'bg-blue-200'
        elif abs(value) < 0.9:
            return 'bg-blue-300'
        else:
            return 'bg-blue-500'
            
    def process_comments_data(self):
        """Process all relevant comments."""
        base_url = "https://openreview.net/forum"

        for submission in self.submissions:
            submission_id = f"{self.venue_id}/{self.submission_name}{submission.number}"

            for reply in submission.details.get("replies", []):
                if self.is_relevant_comment(reply):
                    forum_id = reply.get("forum", "")
                    note_id = reply.get("id", "")
                    replyto = reply.get("replyto", None)
                    link = f"{base_url}?id={forum_id}&noteId={note_id}"
                    signatures = reply.get("signatures", [])
                    role = self.infer_role_from_signature(signatures)
                    tcdate = reply.get("tcdate", None)
                    date_str = self.format_timestamp(tcdate)

                    self.comments_data.append({
                        "Paper #": submission.number,
                        "Paper ID": submission.id,
                        "Type": self.classify_comment_type(reply),
                        "Role": role,
                        "Date": date_str,
                        "Content": self.extract_comment_text(reply),
                        "Link": link,
                        "ReplyTo": reply.get("replyto", None),
                        "NoteId": note_id,
                    })
    
    def process_data(self):
        """Process all data needed for the report."""
        self.process_papers_data()
        self.compute_ac_meta_data()
        self.compute_correlation_data()
        self.process_comments_data()
    
    def generate_histogram_data(self):
        """Generate histogram data for review scores."""
        # Define bins for scores: from 1 to 5 (0.25 interval)
        # bins_overall = np.arange(1, 5.26, 0.25)
        bins_overall = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 
                        4.0, 4.25, 4.5, 4.75, 5.0, 5.25]
        # For meta-reviews, use 0.5 interval
        # bins_meta = np.arange(1, 5.56, 0.5)
        bins_meta = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

        # Use bin starts as labels
        bin_labels_overall = [f"{x:.2f}" for x in bins_overall[:-1]]
        bin_labels_meta = [f"{x:.2f}" for x in bins_meta[:-1]]

        print(bins_overall, bin_labels_overall)
        print(bins_meta, bin_labels_meta)
        
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
        
    def organize_comments_by_paper(self):
        """Organize comments by paper for threaded display."""
        if not self.comments_data:
            return []
            
        # Group comments by paper number
        comments_by_paper = {}
        for comment in self.comments_data:
            paper_num = comment["Paper #"]
            if paper_num not in comments_by_paper:
                comments_by_paper[paper_num] = []
            comments_by_paper[paper_num].append(comment)
            
        # For each paper, build a tree structure
        comment_trees = []
        for paper_num, comments in comments_by_paper.items():
            # Build a map for fast access
            comment_map = {c["NoteId"]: c for c in comments}
            
            # Find roots (comments that don't reply to other comments)
            roots = []
            for comment in comments:
                replyto = comment["ReplyTo"]
                if not replyto or replyto not in comment_map:
                    roots.append(comment)
                    
            # For each root, build a tree
            paper_tree = {
                "paper_num": paper_num,
                "threads": []
            }
            
            for root in roots:
                thread = self._build_comment_tree(root, comments)
                paper_tree["threads"].append(thread)
                
            comment_trees.append(paper_tree)
            
        return comment_trees
        
    def _build_comment_tree(self, root_comment, all_comments):
        """Recursively build a tree of comments."""
        tree = {
            "comment": root_comment,
            "children": []
        }
        
        # Find all replies to this comment
        root_id = root_comment["NoteId"]
        replies = [c for c in all_comments if c["ReplyTo"] == root_id]
        
        # Build trees for each reply
        for reply in replies:
            child_tree = self._build_comment_tree(reply, all_comments)
            tree["children"].append(child_tree)
            
        return tree
        
    def generate_report(self, output_dir="."):
        """Generate the HTML report using Jinja2 templates."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Process all data
        self.process_data()
        
        # Generate additional analytics data
        paper_type_distribution = self.generate_paper_type_distribution()
        review_completion_data = self.generate_review_completion_data()
        score_scatter_data = self.generate_score_scatter_data()
        ac_scoring_data = self.generate_ac_scoring_data()  # Add this line
        
        # Prepare data for templates
        template_data = {
            "title": f"ARR Review Report: {self.venue_id}",
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "venue_id": self.venue_id,
            "papers": self.papers_data,
            "ac_meta": self.ac_meta_data,
            "comments_count": len(self.comments_data),
            "comments": self.comments_data,
            "comment_trees": self.organize_comments_by_paper(),
            "histogram_data": self.generate_histogram_data(),
            "correlation_data": self.correlation_data,
            "paper_type_distribution": paper_type_distribution,
            "review_completion_data": review_completion_data,
            "score_scatter_data": score_scatter_data,
            "ac_scoring_data": ac_scoring_data  # Add this line
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
        output_path = Path(output_dir) / "arr_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path

    def _fix_comment_formatting(self, html_content):
        """Fix markdown formatting issues in comment content post-processing."""
        # Pattern to find the first pre/code block in each comment-content div
        pattern = r'(<div class="mt-2 (?:markdown-content|prose|) comment-content">)\s*<pre><code>(.*?)</code></pre>'
        
        # Replace with properly formatted paragraph
        replacement = r'\1<p>\2</p>'
        
        # Apply the fix (with re.DOTALL to match across newlines)
        fixed_html = re.sub(pattern, replacement, html_content, flags=re.DOTALL)
        
        return fixed_html
        
    def _create_template_files(self, template_dir):
        """Create the template files."""
        
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
                
        # Create AC dashboard component
        ac_dashboard = components_dir / "ac_dashboard.html"
        with open(ac_dashboard, "w", encoding="utf-8") as f:
            f.write(self._get_ac_dashboard_template())
                
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
                
        # Create review completion component
        review_completion = components_dir / "review_completion.html"
        with open(review_completion, "w", encoding="utf-8") as f:
            f.write(self._get_review_completion_template())
                
        # Create score scatter component
        score_scatter = components_dir / "score_scatter.html"
        with open(score_scatter, "w", encoding="utf-8") as f:
            f.write(self._get_score_scatter_template())

        # Create AC scoring component
        ac_scoring = components_dir / "ac_scoring.html"
        with open(ac_scoring, "w", encoding="utf-8") as f:
            f.write(self._get_ac_scoring_template())
    
    def _get_main_template(self):
        """Get the main report template."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            font-size: 0.875rem;
            color: #6b7280;
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
        
        /* Filter dropdown */
        #ac-filter select {
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

        #papers-table_paginate{
            margin-bottom: 10px;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="relative">
        <!-- Header -->
        <header class="bg-indigo-600 shadow-md">
            <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
                <h1 class="text-3xl font-bold text-white">
                    🎛️ ARR Review Dashboard
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
                            <button class="tab-button py-4 px-1 text-center border-b-2 border-transparent hover:border-indigo-500 font-medium text-gray-500 hover:text-gray-700 transition-colors duration-200 focus:outline-none" data-tab="ac-dashboard-tab">
                                AC Dashboard
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
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Papers Overview</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Complete status of all papers in your batch, including review scores and meta-review status.
                        </p>
                    </div>
                    {% include 'components/papers_table.html' %}
                </div>
            </div>

            <!-- AC Dashboard Tab -->
            <div id="ac-dashboard-tab" class="tab-content">
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Area Chair Dashboard</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Performance overview of all Area Chairs in your batch.
                        </p>
                    </div>
                    {% include 'components/ac_dashboard.html' %}
                </div>
            </div>

            <!-- Comments Tab -->
            <div id="comments-tab" class="tab-content">
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
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
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Score Distribution</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Distribution of overall assessment and meta-review scores.
                        </p>
                    </div>
                    {% include 'components/score_distribution.html' %}
                </div>
                
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Paper Type Distribution</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Distribution of papers by type.
                        </p>
                    </div>
                    {% include 'components/paper_type_distribution.html' %}
                </div>
                
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Review Completion Status</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Percentage of reviews completed per Area Chair.
                        </p>
                    </div>
                    {% include 'components/review_completion.html' %}
                </div>
                
                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Score Comparison: Overall vs Meta</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Comparison between average overall assessment scores and meta-review scores.
                        </p>
                    </div>
                    {% include 'components/score_scatter.html' %}
                </div>

                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
                        <h2 class="text-xl font-semibold text-gray-800">Area Chair Scoring Analysis</h2>
                        <p class="mt-1 text-sm text-gray-500">
                            Comparison of average review scores and meta-review scores by Area Chair.
                        </p>
                    </div>
                    {% include 'components/ac_scoring.html' %}
                </div>


                <div class="bg-white shadow overflow-hidden rounded-lg mb-6">
                    <div class="px-4 py-5 sm:px-6">
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
                // Papers table
                $('#papers-table').DataTable({
                    pageLength: 100,
                    autoWidth: false,
                    scrollX: false,
                    order: [[0, 'asc']],
                    columnDefs: [
                        { type: 'num', targets: [0, 4, 5] },
                        { className: 'overall-assessment', targets: 9 },
                        { className: 'meta-review', targets: 10 }
                    ],
                    language: {
                        search: "_INPUT_",
                        searchPlaceholder: "Search papers...",
                        lengthMenu: "Show _MENU_ papers"
                    },
                    initComplete: function() {
                        // Add custom filtering for Area Chair column
                        this.api().columns(3).every(function() {
                            const column = this;
                            const select = $('<select class="ml-2"><option value="">All ACs</option></select>')
                                .appendTo('#ac-filter')
                                .on('change', function() {
                                    const val = $.fn.dataTable.util.escapeRegex($(this).val());
                                    column.search(val ? '^' + val + '$' : '', true, false).draw();
                                });

                            column.data().unique().sort().each(function(d, j) {
                                select.append('<option value="' + d + '">' + d + '</option>');
                            });
                        });
                    }
                });

                // AC Dashboard table
                $('#ac-dashboard-table').DataTable({
                    paging: false,
                    searching: false,
                    info: false,
                    autoWidth: false,
                    scrollX: false,
                    order: [[5, 'desc']]
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
            
            // Review completion chart
            if (typeof initReviewCompletionChart === 'function') {
                initReviewCompletionChart();
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
        """Get the papers table template."""
        return '''<div class="px-4 py-5 sm:p-6">
    <div class="mb-4 flex items-center">
        <div class="max-w-xl">
            <label for="ac-filter" class="text-sm font-medium text-gray-700">Filter by Area Chair:</label>
            <span id="ac-filter"></span>
        </div>
    </div>
    <div class="overflow-x-auto w-full" style="max-width: 100%;">
        <table id="papers-table" class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 40px; width: 40px;">
                        #
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 150px; width: 150px; max-width: 150px;">
                        Title
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 60px; width: 60px;">
                        Type
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 120px; width: 120px;">
                        Area Chair
                    </th>
                    <th scope="col" class="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 80px; width: 80px;">
                        <span title="Completed / Expected">Reviews</span>
                    </th>
                    <th scope="col" class="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style="min-width: 50px; width: 50px;">
                        Ready
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
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                {% for paper in papers %}
                <tr>
                    <td class="px-2 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                        {{ paper["Paper #"] }}
                    </td>
                    <td class="px-2 py-3 text-sm text-gray-500" style="max-width: 150px;">
                        <div class="truncate" title="{{ paper.Title }}">
                            <a href="https://openreview.net/forum?id={{ paper['Paper ID'] }}" target="_blank" class="text-indigo-600 hover:text-indigo-900">{{ paper.Title }}</a>
                        </div>
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {{ paper["Paper Type"] }}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500">
                        {{ paper["Area Chair"] }}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500 text-center">
                        {{ paper["Completed Reviews"] }}/{{ paper["Expected Reviews"] }}
                    </td>
                    <td class="px-2 py-3 whitespace-nowrap text-sm text-gray-500 text-center">
                        {% if paper["Ready for Rebuttal"] %}
                        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            ✓
                        </span>
                        {% else %}
                        <span class="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                            -
                        </span>
                        {% endif %}
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
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>'''

    def _get_ac_dashboard_template(self):
        """Get the AC dashboard template."""
        return '''<div class="px-4 py-5 sm:p-6">
    <div>
        <table id="ac-dashboard-table" class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Area Chair
                    </th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Completed Reviews
                    </th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Expected Reviews
                    </th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Papers Ready
                    </th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Total Papers
                    </th>
                    <th scope="col" class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        All Reviews Ready
                    </th>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Meta Reviews
                    </th>
                    <th scope="col" class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        All Meta-reviews Ready
                    </th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                {% for ac in ac_meta %}
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {{ ac["Area Chair"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {{ ac["Completed_Reviews"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {{ ac["Expected_Reviews"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {{ ac["Papers_Ready"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {{ ac["Num_Papers"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">
                        {% if ac["All Reviews Ready"] %}
                        <span class="inline-flex items-center justify-center px-2 py-1 text-xs font-bold rounded-full bg-green-100 text-green-800">
                            {{ ac["All Reviews Ready"] }}
                        </span>
                        {% else %}
                        <span class="inline-flex items-center justify-center px-2 py-1 text-xs font-bold rounded-full bg-gray-100 text-gray-800">
                            -
                        </span>
                        {% endif %}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {{ ac["Meta_Reviews_Done"] }}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">
                        {% if ac["All Meta-reviews Ready"] %}
                        <span class="inline-flex items-center justify-center px-2 py-1 text-xs font-bold rounded-full bg-green-100 text-green-800">
                            {{ ac["All Meta-reviews Ready"] }}
                        </span>
                        {% else %}
                        <span class="inline-flex items-center justify-center px-2 py-1 text-xs font-bold rounded-full bg-gray-100 text-gray-800">
                            -
                        </span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>'''

    def extract_comment_text(self, reply):
        """Extract and process comment text from a reply."""
        content = reply.get("content", {})
        # Try common keys in order of likely importance
        for key in ["comment", "justification", "text", "response", "value"]:
            if key in content:
                val = content[key]
                raw_text = val.get("value") if isinstance(val, dict) else val
                
                # Process the text to fix markdown formatting issues
                if raw_text:
                    return self.process_comment_text(raw_text)
                    
        # If no known key, flatten all text fields into a fallback string
        fallback = []
        for k, v in content.items():
            if isinstance(v, dict) and "value" in v:
                fallback.append(f"{k}: {v['value']}")
        return self.process_comment_text("\n".join(fallback)) if fallback else "(No comment text found)"

    def process_comment_text(self, text):
        """Process comment text to fix markdown rendering issues."""
        if not text:
            return text
            
        # Trim whitespace
        text = text.strip()
        
        # Fix the issue with first paragraph becoming code block
        # by normalizing indentation
        lines = text.split('\n')
        
        # Determine minimum indentation (ignoring empty lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, current_indent)
        
        if min_indent == float('inf'):
            min_indent = 0
        
        # Remove common indentation from all lines
        processed_lines = [line[min_indent:] if line.strip() else line for line in lines]
        
        # Join lines back together
        processed_text = '\n'.join(processed_lines)
        
        # Pre-process for better markdown rendering
        
        # Ensure code blocks have proper formatting
        processed_text = re.sub(r'```([\s\S]+?)```', 
                              lambda m: '\n```\n' + m.group(1).strip() + '\n```\n', 
                              processed_text)
        
        # Convert consecutive line breaks to paragraph markers
        processed_text = re.sub(r'\n\n+', '\n\n', processed_text)
        
        # Ensure lists are properly formatted
        processed_text = re.sub(r'^(\s*[-*+])\s+', r'\1 ', processed_text, flags=re.MULTILINE)
        
        return processed_text

    def _get_comments_template(self):
        """Get the comments template with fixed content rendering."""
        return '''
<!-- Template function for rendering comment threads -->
{% macro render_comment_thread(thread, level) %}
    {% set comment = thread.comment %}
    {% set bg_color = "bg-gray-50" if level % 2 == 0 else "bg-white" %}
    {% set indent = level * 20 %}
    
    <div class="comment-thread mb-3 ml-{{ indent }} {{ bg_color }} border-l-4 border-gray-200 rounded-lg shadow-sm overflow-hidden comment-item"
         data-paper="{{ comment['Paper #'] }}" 
         data-type="{{ comment.Type }}" 
         data-role="{{ comment.Role }}">
        <div class="px-4 py-3">
            <div class="flex justify-between items-start">
                <div>
                    <span class="inline-block mr-2 text-xs font-medium text-gray-500">Paper #{{ comment['Paper #'] }}</span>
                    {% if comment.Type == "Review Issue" %}
                        <span class="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800 mr-2">{{ comment.Type }}</span>
                    {% elif comment.Type == "Author-Editor Confidential" %}
                        <span class="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800 mr-2">{{ comment.Type }}</span>
                    {% else %}
                        <span class="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 mr-2">{{ comment.Type }}</span>
                    {% endif %}
                    <span class="font-medium">{{ comment.Role }}</span>
                    <span class="text-sm text-gray-500 ml-2">{{ comment.Date }}</span>
                </div>
                <a href="{{ comment.Link }}" target="_blank" class="text-indigo-600 hover:text-indigo-900 text-sm">View on OpenReview</a>
            </div>
            <div class="mt-2 markdown-content comment-content" id="comment-{{ comment.NoteId }}">
                <!-- Content will be rendered here -->
            </div>
        </div>
    </div>
    
    {% if thread.children %}
        {% for child in thread.children %}
            {{ render_comment_thread(child, level + 1) }}
        {% endfor %}
    {% endif %}
{% endmacro %}

<div class="px-4 py-5 sm:p-6">
    <!-- Additional styles for markdown content -->
    <style>
        .markdown-content {
            line-height: 1.6;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        .markdown-content p {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }
        
        .markdown-content h1,
        .markdown-content h2,
        .markdown-content h3,
        .markdown-content h4,
        .markdown-content h5,
        .markdown-content h6 {
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
            line-height: 1.25;
        }
        
        .markdown-content h1 { font-size: 1.5em; }
        .markdown-content h2 { font-size: 1.25em; }
        .markdown-content h3 { font-size: 1.125em; }
        
        .markdown-content ul, 
        .markdown-content ol {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
            padding-left: 1.5em;
        }
        
        .markdown-content ul { list-style-type: disc; }
        .markdown-content ol { list-style-type: decimal; }
        
        .markdown-content li {
            margin-top: 0.25em;
            margin-bottom: 0.25em;
        }
        
        .markdown-content code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            padding: 0.2em 0.4em;
            background-color: rgba(175, 184, 193, 0.2);
            border-radius: 0.25em;
            font-size: 0.9em;
            white-space: pre-wrap;
        }
        
        .markdown-content pre {
            padding: 1em;
            margin: 0.5em 0;
            background-color: #f6f8fa;
            border-radius: 0.375em;
            overflow-x: auto;
            white-space: pre;
        }
        
        .markdown-content pre code {
            background-color: transparent;
            padding: 0;
            font-size: 0.9em;
            white-space: pre;
        }
        
        .markdown-content blockquote {
            padding-left: 1em;
            border-left: 0.25em solid #d1d5db;
            color: #6b7280;
            margin: 0.5em 0;
        }
        
        .markdown-content a {
            color: #3b82f6;
            text-decoration: underline;
        }
        
        .markdown-content a:hover {
            color: #2563eb;
        }
        
        .markdown-content img {
            max-width: 100%;
            height: auto;
        }
        
        .markdown-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        
        .markdown-content table th,
        .markdown-content table td {
            border: 1px solid #d1d5db;
            padding: 0.5em;
        }
        
        .markdown-content table th {
            background-color: #f3f4f6;
        }
        
        .markdown-content hr {
            height: 0.25em;
            padding: 0;
            margin: 1.5em 0;
            background-color: #e5e7eb;
            border: 0;
        }
    </style>

    <!-- Filters for comments -->
    <div class="mb-6 bg-gray-50 p-4 rounded-lg">
        <h3 class="text-lg font-medium text-gray-900 mb-3">Filter Comments</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
                <label for="paper-filter" class="block text-sm font-medium text-gray-700 mb-1">Paper #</label>
                <select id="paper-filter" class="bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full">
                    <option value="all">All Papers</option>
                    <!-- Paper options will be populated by JavaScript -->
                </select>
            </div>
            <div>
                <label for="type-filter" class="block text-sm font-medium text-gray-700 mb-1">Comment Type</label>
                <select id="type-filter" class="bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full">
                    <option value="all">All Types</option>
                    <option value="Review Issue">Review Issues</option>
                    <option value="Author-Editor Confidential">Author-Editor Confidential</option>
                    <option value="Confidential Comment">Confidential Comments</option>
                </select>
            </div>
            <div>
                <label for="role-filter" class="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select id="role-filter" class="bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm w-full">
                    <option value="all">All Roles</option>
                    <option value="Author">Author</option>
                    <option value="Reviewer">Reviewer</option>
                    <option value="Area Chair">Area Chair</option>
                    <option value="Senior Area Chair">Senior Area Chair</option>
                    <option value="Program Chair">Program Chair</option>
                </select>
            </div>
        </div>
    </div>

    <!-- Threaded view -->
    <div id="comments-threaded">
        {% if comment_trees %}
            <div id="no-comments-message" class="hidden text-center py-8 text-gray-500">No comments match the selected filters.</div>
            
            {% for paper in comment_trees %}
                <div class="paper-section border-t border-gray-200 pt-4 mt-4 first:border-t-0 first:pt-0 first:mt-0" data-paper="{{ paper.paper_num }}">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Paper #{{ paper.paper_num }}</h3>
                    {% for thread in paper.threads %}
                        {{ render_comment_thread(thread, 0) }}
                    {% endfor %}
                </div>
            {% endfor %}
        {% else %}
            <div class="text-center py-8 text-gray-500">No comments found.</div>
        {% endif %}
    </div>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Populate paper filter dropdown
        const paperFilter = document.getElementById('paper-filter');
        const paperSections = document.querySelectorAll('.paper-section');
        const commentItems = document.querySelectorAll('.comment-item');
        const noCommentsMessage = document.getElementById('no-comments-message');
        
        // Build unique paper numbers set
        const papers = new Set();
        paperSections.forEach(section => {
            papers.add(section.getAttribute('data-paper'));
        });
        
        // Build unique role set
        const roles = new Set();
        commentItems.forEach(item => {
            roles.add(item.getAttribute('data-role'));
        });
        
        // Add paper options to dropdown
        const paperArray = Array.from(papers).sort((a, b) => parseInt(a) - parseInt(b));
        paperArray.forEach(paper => {
            const option = document.createElement('option');
            option.value = paper;
            option.textContent = `Paper #${paper}`;
            paperFilter.appendChild(option);
        });

        // Add role options to dropdown (if not already defined in template)
        const roleFilter = document.getElementById('role-filter');
        if (roleFilter.children.length <= 1) {
            const rolesArray = Array.from(roles).sort();
            rolesArray.forEach(role => {
                if (role && role !== 'all') {
                    const option = document.createElement('option');
                    option.value = role;
                    option.textContent = role;
                    roleFilter.appendChild(option);
                }
            });
        }

        // Filter function
        function filterComments() {
            const selectedPaper = paperFilter.value;
            const selectedType = document.getElementById('type-filter').value;
            const selectedRole = roleFilter.value;
            
            let visibleCount = 0;
            
            // Hide all paper sections initially
            paperSections.forEach(section => {
                section.classList.add('hidden');
            });
            
            // Filter comments
            commentItems.forEach(item => {
                const paperMatch = selectedPaper === 'all' || item.getAttribute('data-paper') === selectedPaper;
                const typeMatch = selectedType === 'all' || item.getAttribute('data-type') === selectedType;
                const roleMatch = selectedRole === 'all' || item.getAttribute('data-role') === selectedRole;
                
                if (paperMatch && typeMatch && roleMatch) {
                    item.classList.remove('hidden');
                    
                    // Show the parent paper section
                    const parentSection = item.closest('.paper-section');
                    if (parentSection) {
                        parentSection.classList.remove('hidden');
                    }
                    
                    visibleCount++;
                } else {
                    item.classList.add('hidden');
                }
            });
            
            // Show "no comments" message if no matches
            if (visibleCount === 0) {
                noCommentsMessage.classList.remove('hidden');
            } else {
                noCommentsMessage.classList.add('hidden');
            }
        }
        
        // Add event listeners to filters
        paperFilter.addEventListener('change', filterComments);
        document.getElementById('type-filter').addEventListener('change', filterComments);
        roleFilter.addEventListener('change', filterComments);
        
        // Render comment content - we'll use a simple approach to avoid markdown issues
        const commentData = {{ comments | tojson }};
        
        // For each comment, find its container element and render paragraphs
        commentData.forEach(comment => {
            const container = document.getElementById(`comment-${comment.NoteId}`);
            if (container) {
                // Split by double newlines to get paragraphs
                const paragraphs = comment.Content.split('/\\n\\s*\\n/').filter(p => p.trim());
                
                // Create HTML content
                let html = '';
                paragraphs.forEach(paragraph => {
                    // Replace single newlines with <br>
                    const formattedParagraph = paragraph.trim().replace('/\\n/g', '<br>');
                    html += `<p>${formattedParagraph}</p>`;
                });
                
                container.innerHTML = html;
            }
        });
    });
</script>'''

    def _get_score_distribution_template(self):
        """Get the score distribution template."""
        return '''<div id="score-distribution-container" class="px-4 py-5 sm:p-6">
    <canvas id="score-distribution-chart" style="max-height: 400px;"></canvas>
</div>'''

    def _get_correlation_matrix_template(self):
        """Get the correlation matrix template."""
        return '''<div class="px-4 py-5 sm:p-6">
    <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Score Type
                    </th>
                    {% for label in correlation_data.labels %}
                    <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {{ label }}
                    </th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                {% for row in correlation_data.matrix %}
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {{ row.label }}
                    </td>
                    {% for cell in row.cells %}
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 {{ cell.color if cell else '' }}">
                        {{ cell.value if cell else '' }}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>'''


    def _get_paper_type_distribution_template(self):
        """Get the paper type distribution template."""
        return '''<div class="px-4 py-5 sm:p-6">
        <div class="flex flex-col md:flex-row">
            <div class="w-full md:w-1/2">
                <canvas id="paper-type-chart" style="max-height: 300px;"></canvas>
            </div>
            <div class="w-full md:w-1/2 mt-4 md:mt-0">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Paper Type</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Percentage</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {% set total = paper_type_distribution.counts | sum %}
                        {% for i in range(paper_type_distribution.labels | length) %}
                        <tr>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                {{ paper_type_distribution.labels[i] }}
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {{ paper_type_distribution.counts[i] }}
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {{ "%.1f"|format(paper_type_distribution.counts[i] / total * 100) }}%
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const initPaperTypeChart = function() {
            const ctx = document.getElementById('paper-type-chart');
            if (!ctx) return;
            
            const labels = {{ paper_type_distribution.labels | tojson }};
            const counts = {{ paper_type_distribution.counts | tojson }};
            
            if (!labels.length || !counts.length) {
                ctx.parentNode.innerHTML = '<div class="text-center py-8 text-gray-500">No paper type data available.</div>';
                return;
            }
            
            const colors = [
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 99, 132, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(153, 102, 255, 0.7)',
                'rgba(255, 159, 64, 0.7)'
            ];
            
            const borderColors = [
                'rgba(54, 162, 235, 1)',
                'rgba(255, 99, 132, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ];
            
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: counts,
                        backgroundColor: colors.slice(0, labels.length),
                        borderColor: borderColors.slice(0, labels.length),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                boxWidth: 15,
                                padding: 15
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.formattedValue;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = Math.round((context.raw / total) * 100);
                                    return `${label}: ${value} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        };
        
        // Call function when analytics tab is shown
        const analyticsTab = document.querySelector('[data-tab="analytics-tab"]');
        if (analyticsTab) {
            analyticsTab.addEventListener('click', function() {
                setTimeout(initPaperTypeChart, 100);
            });
        }
        
        // Also initialize if analytics tab is active initially
        if (document.getElementById('analytics-tab').classList.contains('active')) {
            setTimeout(initPaperTypeChart, 100);
        }
    });
    </script>'''


    def _get_review_completion_template(self):
        """Get the review completion template."""
        return '''<div class="px-4 py-5 sm:p-6">
        <div id="review-completion-chart-container">
            <canvas id="review-completion-chart" style="max-height: 400px; height: 400px;"></canvas>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const initReviewCompletionChart = function() {
            const ctx = document.getElementById('review-completion-chart');
            if (!ctx) return;
            
            const reviewData = {{ review_completion_data | tojson }};
            
            if (!reviewData.length) {
                document.getElementById('review-completion-chart-container').innerHTML = 
                    '<div class="text-center py-8 text-gray-500">No review completion data available.</div>';
                return;
            }
            
            // Prepare data for horizontal bar chart
            const labels = reviewData.map(item => item.name);
            const data = reviewData.map(item => item.percentage);
            
            // Generate background colors based on completion percentage
            const backgroundColors = data.map(percentage => {
                if (percentage >= 100) return 'rgba(34, 197, 94, 0.7)';  // Green for 100%
                if (percentage >= 75) return 'rgba(59, 130, 246, 0.7)';  // Blue for 75%+
                if (percentage >= 50) return 'rgba(234, 179, 8, 0.7)';   // Yellow for 50%+
                return 'rgba(239, 68, 68, 0.7)';                         // Red for <50%
            });
            
            const borderColors = data.map(percentage => {
                if (percentage >= 100) return 'rgba(34, 197, 94, 1)';
                if (percentage >= 75) return 'rgba(59, 130, 246, 1)';
                if (percentage >= 50) return 'rgba(234, 179, 8, 1)';
                return 'rgba(239, 68, 68, 1)';
            });
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Review Completion (%)',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1,
                        barThickness: 'flex',
                        barPercentage: 0.8,
                        maxBarThickness: 30
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Completion Percentage'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Area Chair'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Completion: ${context.formattedValue}%`;
                                }
                            }
                        }
                    }
                }
            });
        };
        
        // Call function when analytics tab is shown
        const analyticsTab = document.querySelector('[data-tab="analytics-tab"]');
        if (analyticsTab) {
            analyticsTab.addEventListener('click', function() {
                setTimeout(initReviewCompletionChart, 100);
            });
        }
        
        // Also initialize if analytics tab is active initially
        if (document.getElementById('analytics-tab').classList.contains('active')) {
            setTimeout(initReviewCompletionChart, 100);
        }
    });
    </script>'''


    def _get_score_scatter_template(self):
        """Get the score scatter template."""
        return '''<div class="px-4 py-5 sm:p-6">
        <div class="flex flex-col md:flex-row">
            <div class="w-full md:w-1/2 mb-6 md:mb-0">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Scatter Plot</h3>
                <div class="relative">
                    <canvas id="score-scatter-chart" style="max-height: 350px; height: 350px;"></canvas>
                </div>
                <p class="text-sm text-gray-500 mt-2 text-center">
                    Each point represents a paper showing the relationship between average review score and meta-review score.
                </p>
            </div>
            <div class="w-full md:w-1/2 md:pl-6">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Score Difference Distribution</h3>
                <div class="relative">
                    <canvas id="score-difference-chart" style="max-height: 350px; height: 350px;"></canvas>
                </div>
                <p class="text-sm text-gray-500 mt-2 text-center">
                    Distribution of differences between meta-review score and average review score.
                    Positive values indicate meta-review scores higher than average review scores.
                </p>
            </div>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const initScoreCharts = function() {
            const scatterCtx = document.getElementById('score-scatter-chart');
            const diffCtx = document.getElementById('score-difference-chart');
            if (!scatterCtx || !diffCtx) return;
            
            const scatterData = {{ score_scatter_data.scatter | tojson }};
            const diffData = {{ score_scatter_data.differences | tojson }};
            
            if (!scatterData.length) {
                scatterCtx.parentNode.innerHTML = '<div class="text-center py-8 text-gray-500">No score comparison data available.</div>';
                diffCtx.parentNode.innerHTML = '<div class="text-center py-8 text-gray-500">No score difference data available.</div>';
                return;
            }
            
            // Create scatter plot
            new Chart(scatterCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Papers',
                        data: scatterData.map(item => ({x: item.x, y: item.y})),
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Average Overall Assessment'
                            },
                            min: 1,
                            max: 5,
                            ticks: {
                                stepSize: 0.5
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Meta-Review Score'
                            },
                            min: 1,
                            max: 5,
                            ticks: {
                                stepSize: 0.5
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = scatterData[context.dataIndex];
                                    return `Paper: ${point.paper}, Avg Review: ${point.x.toFixed(1)}, Meta: ${point.y.toFixed(1)}`;
                                }
                            }
                        }
                    }
                }
            });
            
            // Create difference histogram
            new Chart(diffCtx, {
                type: 'bar',
                data: {
                    labels: diffData.labels,
                    datasets: [{
                        label: 'Number of Papers',
                        data: diffData.counts,
                        backgroundColor: function(context) {
                            const value = parseFloat(context.chart.data.labels[context.dataIndex]);
                            if (value > 0) return 'rgba(34, 197, 94, 0.7)';  // Green for positive
                            if (value < 0) return 'rgba(239, 68, 68, 0.7)';  // Red for negative
                            return 'rgba(107, 114, 128, 0.7)';               // Gray for zero
                        },
                        borderColor: function(context) {
                            const value = parseFloat(context.chart.data.labels[context.dataIndex]);
                            if (value > 0) return 'rgba(34, 197, 94, 1)';
                            if (value < 0) return 'rgba(239, 68, 68, 1)';
                            return 'rgba(107, 114, 128, 1)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Score Difference (Meta - Avg)'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Papers'
                            },
                            ticks: {
                                stepSize: 1
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                title: function(tooltipItems) {
                                    const diff = parseFloat(tooltipItems[0].label);
                                    if (diff > 0) {
                                        return `Meta score is ${diff} points higher`;
                                    } else if (diff < 0) {
                                        return `Meta score is ${Math.abs(diff)} points lower`;
                                    } else {
                                        return 'Meta score equals average review score';
                                    }
                                }
                            }
                        }
                    }
                }
            });
        };
        
        // Call function when analytics tab is shown
        const analyticsTab = document.querySelector('[data-tab="analytics-tab"]');
        if (analyticsTab) {
            analyticsTab.addEventListener('click', function() {
                setTimeout(initScoreCharts, 100);
            });
        }
        
        // Also initialize if analytics tab is active initially
        if (document.getElementById('analytics-tab').classList.contains('active')) {
            setTimeout(initScoreCharts, 100);
        }
    });
    </script>'''


    def _get_ac_scoring_template(self):
        """Get the AC scoring analysis template."""
        return '''<div class="px-4 py-5 sm:p-6">
        <div id="ac-scoring-chart-container" class="mb-6">
            <canvas id="ac-scoring-chart" style="max-height: 400px; height: 400px;"></canvas>
        </div>
        
        <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Area Chair</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Review Score</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Meta Score</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Difference</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Review Count</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Meta Count</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% for ac in ac_scoring_data %}
                    <tr>
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {{ ac.name }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ ac.overall_avg if ac.overall_avg is not none else "-" }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ ac.meta_avg if ac.meta_avg is not none else "-" }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500
                            {% if ac.difference is not none %}
                                {% if ac.difference > 0.2 %}
                                    text-green-600 font-medium
                                {% elif ac.difference < -0.2 %}
                                    text-red-600 font-medium
                                {% endif %}
                            {% endif %}">
                            {{ ac.difference if ac.difference is not none else "-" }}
                            {% if ac.difference is not none %}
                                {% if ac.difference > 0 %}
                                    <span class="ml-1">↑</span>
                                {% elif ac.difference < 0 %}
                                    <span class="ml-1">↓</span>
                                {% endif %}
                            {% endif %}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ ac.overall_count }}
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ ac.meta_count }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const initACScoringChart = function() {
            const ctx = document.getElementById('ac-scoring-chart');
            if (!ctx) return;
            
            const scoringData = {{ ac_scoring_data | tojson }};
            
            if (!scoringData.length) {
                document.getElementById('ac-scoring-chart-container').innerHTML = 
                    '<div class="text-center py-8 text-gray-500">No AC scoring data available.</div>';
                return;
            }
            
            // Take only top 10 ACs for better readability
            const chartData = scoringData.slice(0, 10);
            
            // Prepare data for bar chart
            const labels = chartData.map(item => item.name);
            const overallData = chartData.map(item => item.overall_avg);
            const metaData = chartData.map(item => item.meta_avg);
            const diffData = chartData.map(item => item.difference);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Avg Review Score',
                            data: overallData,
                            backgroundColor: 'rgba(59, 130, 246, 0.7)',
                            borderColor: 'rgba(59, 130, 246, 1)',
                            borderWidth: 1,
                            order: 2
                        },
                        {
                            label: 'Avg Meta Score',
                            data: metaData,
                            backgroundColor: 'rgba(220, 38, 38, 0.7)',
                            borderColor: 'rgba(220, 38, 38, 1)',
                            borderWidth: 1,
                            order: 1
                        },
                    ]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 5,
                            title: {
                                display: true,
                                text: 'Average Score'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Area Chair'
                            }
                        },
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.dataset.label;
                                    const value = context.formattedValue;
                                    
                                    if (label === 'Difference') {
                                        const val = parseFloat(value);
                                        if (val > 0) {
                                            return `${label}: +${value} (Meta > Review)`;
                                        } else if (val < 0) {
                                            return `${label}: ${value} (Meta < Review)`;
                                        } else {
                                            return `${label}: ${value} (Equal)`;
                                        }
                                    }
                                    
                                    return `${label}: ${value}`;
                                }
                            }
                        }
                    }
                }
            });
        };
        
        // Call function when analytics tab is shown
        const analyticsTab = document.querySelector('[data-tab="analytics-tab"]');
        if (analyticsTab) {
            analyticsTab.addEventListener('click', function() {
                setTimeout(initACScoringChart, 100);
            });
        }
        
        // Also initialize if analytics tab is active initially
        if (document.getElementById('analytics-tab').classList.contains('active')) {
            setTimeout(initACScoringChart, 100);
        }
    });
    </script>'''

def main():
    parser = argparse.ArgumentParser(description='Generate ARR Review Report')
    parser.add_argument('--username', required=True, help='OpenReview username')
    parser.add_argument('--password', required=True, help='OpenReview password')
    parser.add_argument('--venue_id', required=True, help='Venue ID (e.g., aclweb.org/ACL/ARR/2025/February)')
    parser.add_argument('--me', required=True, help='Your OpenReview ID (e.g., ~Your_Name1)')
    parser.add_argument('--output_dir', default='.', help='Output directory for the report')
    
    args = parser.parse_args()
    
    generator = ARRReportGenerator(
        username=args.username,
        password=args.password,
        venue_id=args.venue_id,
        me=args.me
    )
    
    report_path = generator.generate_report(output_dir=args.output_dir)
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()