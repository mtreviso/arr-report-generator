export OPENREVIEW_USERNAME="your_username@domain.com"
export OPENREVIEW_PASSWORD="your_password"
export OPENREVIEW_ID="~your_openreview_id"

# generate commitment report in: reports/review_report.html
python3 generate_review_report.py --venue_id "aclweb.org/ACL/ARR/2025/February"

# generate commitment report in: reports/commitment_report.html
python3 generate_commitment_report.py --venue_id "aclweb.org/ACL/2025/Conference"
