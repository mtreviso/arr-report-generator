#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARR Report Generator Runner
===========================

A simple script to run the ARR report generator with your OpenReview credentials.

Author: Based on Yiming Cui's ARR Tool (https://ymcui.com/)
"""

import argparse
import os
from datetime import datetime
from arr_report_generator import ARRReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Generate beautiful HTML report for ARR review status',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--username', 
                      default=os.environ.get('OPENREVIEW_USERNAME', ''), 
                      help='OpenReview username (can be set via OPENREVIEW_USERNAME env var)')
    
    parser.add_argument('--password', 
                      default=os.environ.get('OPENREVIEW_PASSWORD', ''),
                      help='OpenReview password (can be set via OPENREVIEW_PASSWORD env var)')
    
    parser.add_argument('--venue_id', 
                      default='aclweb.org/ACL/ARR/2025/February',
                      help='OpenReview venue ID')
    
    parser.add_argument('--me', 
                      default=os.environ.get('OPENREVIEW_ID', ''),
                      help='Your OpenReview ID (e.g., ~Your_Name1)')
    
    parser.add_argument('--output_dir', 
                      default='./reports',
                      help='Directory to save the generated report')
    
    args = parser.parse_args()
    
    # Check if credentials are provided
    if not args.username:
        args.username = input("Enter your OpenReview username: ")
    
    if not args.password:
        import getpass
        args.password = getpass.getpass("Enter your OpenReview password: ")
    
    if not args.me:
        args.me = input("Enter your OpenReview ID (e.g., ~Your_Name1): ")
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating ARR report for venue: {args.venue_id}")
    print(f"Report will be saved in: {args.output_dir}")
    
    # Create and run the generator
    generator = ARRReportGenerator(
        username=args.username,
        password=args.password,
        venue_id=args.venue_id,
        me=args.me
    )
    
    # Generate the report
    report_path = generator.generate_report(output_dir=args.output_dir)
    
    print(f"\nReport successfully generated!")
    print(f"Report location: {report_path}")
    print(f"Open the HTML file in your browser to view the report.")

if __name__ == "__main__":
    main()
