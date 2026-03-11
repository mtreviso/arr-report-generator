#!/usr/bin/env python3
"""
Generate ARR Review Phase Report.

--role sac (default): only papers in your SAC batch
--role pc:            all papers in the venue (Program Chair / testing mode)
"""
import os, sys, argparse, getpass
from arr_report_generator import ARRReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Generate ARR Review Phase Report',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--username',   default=os.environ.get('OPENREVIEW_USERNAME', ''))
    parser.add_argument('--password',   default=os.environ.get('OPENREVIEW_PASSWORD', ''))
    parser.add_argument('--venue_id',   default='aclweb.org/ACL/ARR/2025/February')
    parser.add_argument('--me',         default=os.environ.get('OPENREVIEW_ID', ''),
                        help='Your OpenReview tilde ID, e.g. ~Your_Name1')
    parser.add_argument('--role',       default='sac', choices=['sac', 'pc'],
                        help='sac=Senior AC (default), pc=all papers (PC/test mode)')
    parser.add_argument('--output_dir', default='./reports')
    args = parser.parse_args()

    if not args.username:
        args.username = input("OpenReview username: ")
    if not args.password:
        args.password = getpass.getpass("OpenReview password: ")
    if not args.me:
        args.me = input("Your OpenReview ID (e.g. ~Your_Name1): ")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating review report | venue: {args.venue_id} | role: {args.role}")

    gen = ARRReportGenerator(
        username=args.username, password=args.password,
        venue_id=args.venue_id, me=args.me, role=args.role
    )
    path = gen.generate_report(output_dir=args.output_dir)
    print(f"Report generated: {path}")

if __name__ == "__main__":
    main()
