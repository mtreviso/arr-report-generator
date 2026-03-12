# ARR Report Generator

HTML dashboards for OpenReview ARR/ACL venues for Senior Area Chairs, Area Chairs, and Program Chairs. Covers both the **review phase** and the **commitment phase**.

**Runs entirely on your machine.** Your OpenReview credentials are only used to fetch data and are never stored or transmitted elsewhere.

---

## Screenshots

| ![](ss/ss1.png) | ![](ss/ss2.png) | ![](ss/ss3.png) |
|:---:|:---:|:---:|
| ![](ss/ss4.png) | ![](ss/ss5.png) | ![](ss/ss6.png) |

---

## Installation

```bash
git clone https://github.com/mtreviso/arr-report-generator
cd arr-report-generator
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

### Review Report (ARR venue)

```bash
python generate_review_report.py \
  --username your@email.com \
  --password yourpassword \
  --venue_id aclweb.org/ACL/ARR/2025/February \
  --me "~Your_Name1"
```

`--role sac` (default): only papers in your SAC batch.  
`--role pc`: all papers in the venue (PC/test mode, may take several minutes).

### PC Dashboard (ARR venue)

Fetches **all** papers across all SAC batches. Intended for Program Chairs. May take several minutes for large venues (6000+ papers).

```bash
python generate_pc_report.py \
  --username "your@email.com" \
  --password "yourpassword" \
  --venue_id "aclweb.org/ACL/ARR/2025/February" \
  --me "~Your_Name1"
```

### Commitment Report (ACL/EMNLP/etc. venue)

```bash
python generate_commitment_report.py \
  --username "your@email.com" \
  --password "yourpassword" \
  --venue_id "aclweb.org/ACL/2025/Conference" \
  --me "~Your_Name1" \
  --role sac
```

`--role sac` (default): papers where you are SAC.  
`--role ac`: papers where you are AC.  
`--role pc`: all papers.

### Credentials via environment variables

All three scripts read these env vars if flags are not supplied:

```bash
export OPENREVIEW_USERNAME="your@email.com"
export OPENREVIEW_PASSWORD="yourpassword"
export OPENREVIEW_ID="~Your_Name1"
python generate_review_report.py --venue_id aclweb.org/ACL/ARR/2025/February
```

### Output files

| Script | Output |
|--------|--------|
| `generate_review_report.py` | `reports/review_report.html` |
| `generate_pc_report.py` | `reports/pc_report.html` |
| `generate_commitment_report.py` | `reports/commitment_report.html` |

Open in any browser — fully self-contained, no server needed.

---

## Developer Cache (skip the slow fetch)

For iterating on templates locally, you can cache all OpenReview data to disk after one full run and reload it instantly on subsequent runs.

```bash
# Step 1: full fetch + save cache (do this once)
python generate_review_report.py ... --save-cache

# Step 2: use cache for all subsequent runs (seconds, not minutes)
python generate_review_report.py ... --use-cache
```

`--cache-dir .dev_cache` (default) controls where the pickle files go. You can maintain multiple named caches, e.g. one per SAC:

```bash
python generate_review_report.py ... --impersonate "~SAC_Name1" --save-cache --cache-dir .cache_sac1
python generate_review_report.py ... --use-cache --cache-dir .cache_sac1
```

The cache stores three files: `submissions.pkl`, `group_index.pkl`, and `processed.pkl`. It is **not** venue-aware — if you switch venues, use a different `--cache-dir` or delete the existing one.

> `--save-cache` and `--use-cache` are mutually exclusive.

---

## Impersonation (view as a SAC)

The review and commitment reports support impersonating another OpenReview user before fetching data. This requires **PC-level permission** on the venue and uses the [OpenReview impersonate API](https://docs.openreview.net/reference/api-v2/openapi-definition#post-impersonate).

```bash

python generate_review_report.py ... --me "~Target_SAC_Name1" --impersonate
python generate_commitment_report.py ... --me "~Target_SAC_Name1" --impersonate
```

You can pass a `GROUP_ID` after the `--impersonate` flag, which corresponds to the authorization group, e.g., `aclweb.org/ACL/ARR/2025/October/Program_Chairs`. If `GROUP_ID` is empty, it will assume to be `<venue_id>/Program_Chairs`. Your own credentials are used to authenticate, and all data is then fetched as the target user. 

This is useful for verifying exactly what a SAC sees without asking them to run the script themselves. Can be combined with `--save-cache` / `--use-cache` for fast iteration.

> Impersonation is intentionally not available in `generate_pc_report.py`, which already fetches all papers by design.

---

## Comments levels

All report scripts support `--comments-level none|basic|full`.

- `none`: omit comments tab and skip comment processing
- `basic`: direct replies only (default)
- `full`: full reply threads --- slowest / largest output

---

## Report Contents

### Review Report (`review_report.html`)
- **Papers Overview** — filterable table with per-paper review scores, flags (ethics, low confidence, emergency, review issues), AC name linking to their OpenReview profile
- **AC Progress** — per-AC aggregation: completion rates, meta-review status, late reviews, emergency declarations, with email column
- **Comments** — threaded view of all confidential comments and review issue reports
- **Analytics** — score histograms, correlation matrix, overall-vs-AC-score density heatmap, score difference distribution

### PC Dashboard (`pc_report.html`)
Everything in the review report, plus:
- **Overview** — venue-wide counters (total papers, review %, meta %, ethics flags, emergency stats, paper type breakdown)
- **SAC Progress** — per-SAC aggregation with email column and OpenReview profile links
- **Track Breakdown** — per-track review and meta-review completion
- **Attention Queue** — papers needing PC action (ethics flags, review issues, missing reviews)
- **Extended Analytics** — reviewer load histogram, AC/SAC load histograms, score outliers, high-disagreement papers, SAC-level scoring divergence

### Commitment Report (`commitment_report.html`)
- **Papers Overview** — filterable table with recommendation, presentation type, award nomination, linked ARR submission, score summary
- **Comments** and **Analytics** tabs

---

## Project Structure

```
arr-report-generator/
├── generate_review_report.py       # Entry point — review phase
├── generate_commitment_report.py   # Entry point — commitment phase
├── generate_pc_report.py           # Entry point — PC dashboard
├── arr_report_generator.py         # Core logic (SAC/review phase)
├── arr_pc_generator.py             # Extends core for PC dashboard
├── arr_commitment_generator.py     # Extends core for commitment phase
├── dev_cache.py                    # Pickle cache + impersonation utilities
├── templates/
│   ├── review_report.html          # Full-page layout — review
│   ├── pc_report.html              # Full-page layout — PC dashboard
│   ├── commitment_report.html      # Full-page layout — commitment
│   └── components/
│       ├── _shared_styles.html     # CSS shared by all layouts
│       ├── _shared_tab_js.html     # Tab navigation + chart init JS
│       ├── papers_table_review.html
│       ├── papers_table_pc.html
│       ├── papers_table_commitment.html
│       ├── ac_dashboard.html       # AC Progress tab
│       ├── sac_dashboard.html      # SAC Progress tab (PC only)
│       ├── comments.html
│       ├── score_distribution.html
│       ├── score_scatter.html
│       ├── paper_type_distribution.html
│       ├── review_completion.html
│       ├── ac_scoring.html
│       └── correlation_matrix.html
└── requirements.txt
```

Templates are plain Jinja2 HTML files — edit them directly to customise the layout, add columns, or change styling without touching any Python.

---

## Configuring the low-confidence threshold

The low-confidence flag triggers when any reviewer on a paper has a confidence score at or below the threshold. The default is **2**. To change it, edit the class variable near the top of `arr_report_generator.py`:

```python
class ARRReportGenerator:
    LOW_CONF_THRESHOLD = 2   # ← change this
```

---

## AC Progress columns

| Column | Meaning |
|--------|---------|
| Late Rev. | Papers where completed reviews < expected (still missing reviews) |
| ⚡ Emerg. | Papers where an emergency review has been declared |
| ⚡ Unmet | Emergency declared but **no emergency reviewer assigned** yet |

The AC Progress tab is shared by all three report types. In the PC dashboard it gains a **SAC** column showing which Senior Area Chair is responsible for each AC.

---

## Emergency review detection

The generator checks reply invitations for patterns matching:
- `/-/Emergency_Review_Request`
- `/-/Emergency_Reviewer_Recruitment`
- `/-/Emergency_Reviewer_Request`
- `/-/Emergency_Review`

And checks for a group with suffix:
- `/Emergency_Reviewers`
- `/Emergency_Reviewer`
- `/Emergency_Review_Assignees`

On first run, invitation types found on the first paper are printed to stdout as `[DEBUG] Reply invitation types`. If emergencies are not being detected, check this output and file an issue with the actual invitation string.

---

## Credits

- Based on the ARR Tool by [Yiming Cui](https://ymcui.com/)

## License

MIT
