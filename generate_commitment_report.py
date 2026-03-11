#!/usr/bin/env python3
"""
Generate Commitment Phase Report for ACL Conferences.

--role sac (default): papers where you are Senior Area Chair
--role ac:            papers where you are Area Chair
--role pc:            all papers (Program Chair / testing mode)
"""
import os, sys, argparse
from arr_commitment_generator import CommitmentReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Generate Commitment Phase Report for ACL Conferences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--username',   default=os.environ.get('OPENREVIEW_USERNAME', ''))
    parser.add_argument('--password',   default=os.environ.get('OPENREVIEW_PASSWORD', ''))
    parser.add_argument('--venue_id',   default='aclweb.org/ACL/2025/Conference')
    parser.add_argument('--me',         default=os.environ.get('OPENREVIEW_ID', ''),
                        help='Your OpenReview tilde ID, e.g. ~Your_Name1')
    parser.add_argument('--role',       default='sac', choices=['sac', 'ac', 'pc'],
                        help='sac=Senior AC (default), ac=Area Chair, pc=all papers (PC/test mode)')
    parser.add_argument('--output_dir', default='./reports')
    args = parser.parse_args()

    if not args.username or not args.password or not args.venue_id or not args.me:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating commitment report | venue: {args.venue_id} | role: {args.role} | me: {args.me}")

    try:
        gen = CommitmentReportGenerator(
            username=args.username, password=args.password,
            venue_id=args.venue_id, me=args.me, role=args.role
        )
        path = gen.generate_report(output_dir=args.output_dir)
        print(f"Report generated: {path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
