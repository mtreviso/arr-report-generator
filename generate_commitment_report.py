#!/usr/bin/env python3
"""
Commitment Phase Report Generator for ACL Conferences

This script generates an HTML report for the commitment phase of ACL conferences,
showing papers, reviews, meta-reviews, and comments with detailed analytics.
"""

import os
import sys
import argparse
from arr_commitment_generator import CommitmentReportGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate Commitment Phase Report for ACL Conferences')
    
    parser.add_argument('--username', 
                      default=os.environ.get('OPENREVIEW_USERNAME', ''), 
                      help='OpenReview username (can be set via OPENREVIEW_USERNAME env var)')
    
    parser.add_argument('--password', 
                      default=os.environ.get('OPENREVIEW_PASSWORD', ''),
                      help='OpenReview password (can be set via OPENREVIEW_PASSWORD env var)')
    
    parser.add_argument('--venue_id', 
                      default='aclweb.org/ACL/2025/Conference',
                      help='OpenReview venue ID')
    
    parser.add_argument('--me', 
                      default=os.environ.get('OPENREVIEW_ID', ''),
                      help='Your OpenReview ID (e.g., ~Your_Name1)')
    
    parser.add_argument('--output_dir', 
                      default='./reports',
                      help='Directory to save the generated report')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.username or not args.password or not args.venue_id or not args.me:
        parser.print_help()
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Generating commitment phase report for venue: {args.venue_id}")
    print(f"User: {args.username}")
    print(f"Senior AC ID: {args.me}")
    
    # Generate the report
    try:
        generator = CommitmentReportGenerator(
            username=args.username,
            password=args.password,
            venue_id=args.venue_id,
            me=args.me
        )
        
        report_path = generator.generate_report(output_dir=args.output_dir)
        print(f"Report generated successfully at: {report_path}")
        print(f"Open this file in a web browser to view the report")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()