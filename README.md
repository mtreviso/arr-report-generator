# ARR Report Generator

HTML dashboards for OpenReview ARR/ACL venues for Senior Area Chairs, Area Chairs, and Program Chairs. It covers both the **review phase** and the **commitment phase**.

Important note: Runs entirely on your machine. Your OpenReview credentials are only used to fetch data and are never stored or transmitted elsewhere.

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

## Usage

### Review Phase (ARR venue)

```bash
python generate_review_report.py \
  --username your@email.com \
  --password yourpassword \
  --venue_id aclweb.org/ACL/ARR/2025/May \
  --me ~Your_Name1

# PC / test mode — all papers in the venue:
python generate_review_report.py ... --role pc
```

`--role` options: `sac` (default), `pc`

Note: `pc` role can take several minutes.


### Commitment Phase (ACL/EMNLP/etc venue)

```bash
python generate_commitment_report.py \
  --username your@email.com \
  --password yourpassword \
  --venue_id aclweb.org/ACL/2025/Conference \
  --me ~Your_Name1 \
  --role sac      # default; also: ac, pc
```

`--role` options: `sac` (default) — Senior AC papers, `ac` — Area Chair papers, `pc` — all papers

Note: `pc` role can take several minutes.


### PC Dashboard (ARR or ACL venue)

Fetches **all** papers. May take several minutes for large venues (6000+ papers).

```bash
python generate_pc_report.py \
  --username your@email.com \
  --password yourpassword \
  --venue_id aclweb.org/ACL/ARR/2025/May \
  --me ~Your_Name1
```


## Credentials via env vars

```bash
export OPENREVIEW_USERNAME=your@email.com
export OPENREVIEW_PASSWORD=yourpassword
export OPENREVIEW_ID=~Your_Name1
python generate_review_report.py --venue_id aclweb.org/ACL/ARR/2025/May
```

## Output files

| Script | Output |
|--------|--------|
| `generate_review_report.py` | `reports/review_report.html` |
| `generate_commitment_report.py` | `reports/commitment_report.html` |
| `generate_pc_report.py` | `reports/pc_report.html` |

Open in any browser — fully self-contained, no server needed.

---


## Project structure

```
arr-report-generator/
├── generate_review_report.py       # Entry point — review phase
├── generate_commitment_report.py   # Entry point — commitment phase
├── arr_report_generator.py         # Core logic (SAC, review phase)
├── arr_commitment_generator.py     # Extends core for commitment phase
├── templates/
│   ├── review_report.html          # Full-page layout — review
│   ├── commitment_report.html      # Full-page layout — commitment
│   └── components/
│       ├── _shared_styles.html     # CSS shared by both layouts
│       ├── _shared_tab_js.html     # Tab navigation + chart init JS
│       ├── papers_table_review.html
│       ├── papers_table_commitment.html
│       ├── ac_dashboard.html
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


## Configuring the low-confidence threshold

The low-confidence flag triggers when any reviewer on a paper has a confidence score at or below the threshold. The default is **2**. To change it, edit the class variable near the top of `arr_report_generator.py`:

```python
class ARRReportGenerator:
    LOW_CONF_THRESHOLD = 2   # ← change this
```


## AC Dashboard columns

| Column | Meaning |
|--------|---------|
| Late Rev. | Papers where completed reviews < expected (i.e. still missing reviews) |
| ⚡ Emerg. | Papers where an emergency review has been declared |
| ⚡ Unmet | Emergency declared but **no emergency reviewer assigned** yet |

The AC dashboard is shared by all three report types. In the PC dashboard it gains a **SAC** column showing which Senior Area Chair is responsible for each AC.

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