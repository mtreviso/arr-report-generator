#!/usr/bin/env python3
"""
Generate Program Chair (PC) Dashboard for ARR Venues.

Fetches ALL active submissions (no SAC/AC filtering) and produces
an aggregate dashboard with overview stats, SAC-level progress,
track breakdown, attention queue, and full paper table.

Usage:
  python generate_pc_report.py \\
    --username your@email.com \\
    --password yourpassword \\
    --venue_id aclweb.org/ACL/ARR/2025/May \\
    --me ~Your_Name1

  # Or via env vars:
  export OPENREVIEW_USERNAME=your@email.com
  export OPENREVIEW_PASSWORD=yourpassword
  export OPENREVIEW_ID=~Your_Name1
  python generate_pc_report.py --venue_id aclweb.org/ACL/ARR/2025/May
"""
import os, sys, argparse, getpass
from arr_pc_generator import PCReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate PC Dashboard for ARR/ACL Conferences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--username',   default=os.environ.get('OPENREVIEW_USERNAME', ''))
    parser.add_argument('--password',   default=os.environ.get('OPENREVIEW_PASSWORD', ''))
    parser.add_argument('--venue_id',   required=True,
                        help='OpenReview venue ID, e.g. aclweb.org/ACL/ARR/2025/May')
    parser.add_argument('--me',         default=os.environ.get('OPENREVIEW_ID', ''),
                        help='Your OpenReview tilde ID, e.g. ~Your_Name1')
    parser.add_argument('--output_dir', default='./reports')
    args = parser.parse_args()

    if not args.username:
        args.username = input("OpenReview username: ")
    if not args.password:
        args.password = getpass.getpass("OpenReview password: ")
    if not args.me:
        args.me = input("Your OpenReview tilde ID (e.g. ~Your_Name1): ")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating PC dashboard | venue: {args.venue_id}")
    print("NOTE: This fetches ALL papers in the venue — may take several minutes for large venues.")

    try:
        gen = PCReportGenerator(
            username=args.username, password=args.password,
            venue_id=args.venue_id, me=args.me
        )
        path = gen.generate_report(output_dir=args.output_dir)
        print(f"\nReport generated: {path}")
        print("Open the HTML file in your browser to view the dashboard.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
