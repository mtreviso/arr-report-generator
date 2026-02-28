# ARR Report Generator

Interactive HTML dashboards for ACL Rolling Review Senior Area Chairs, covering both the **review phase** and the **commitment phase**.

> Runs entirely on your machine. Your OpenReview credentials are only used to fetch data and are never stored or transmitted elsewhere.

---

## Screenshots

| ![](ss/ss1.png) | ![](ss/ss2.png) | ![](ss/ss3.png) |
|:---:|:---:|:---:|
| ![](ss/ss4.png) | ![](ss/ss5.png) | ![](ss/ss6.png) |

---

## Features

**Both phases**
- Papers table with sortable columns (score, Ready status, flags)
- Anonymity indicator per paper (Anon / Non-anon badge, inferred from OpenReview metadata)
- Low-confidence reviewer flag — highlights papers where any reviewer confidence is ≤ 2
- Review Issue badge with a direct link to the issue note on OpenReview
- Threaded comment viewer with per-paper / per-type / per-role filtering
- Score distribution, paper-type breakdown, and score scatter charts

**Review phase**
- Area Chair dashboard (review + meta-review completion per AC)
- Area Chair filter on the papers table
- AC scoring analysis chart

**Commitment phase**
- Meta-review Issue badge with direct link to the issue note
- Linked ARR forum surfaced in the title column
- SAC recommendation, presentation mode, and award columns
- Resubmission link when present

---

## Installation

```bash
git clone https://github.com/mtreviso/arr-report-generator
cd arr-report-generator
pip install -r requirements.txt
```

---

## Usage

### Review phase

Run as a Senior Area Chair during the ARR review cycle.

```bash
python generate_review_report.py \
  --username "you@email.com" \
  --password "••••••••" \
  --me "~Your_Name1" \
  --venue_id "aclweb.org/ACL/ARR/2025/February"
```

Output: `./reports/review_report.html`

### Commitment phase

Run as an Area Chair after authors have committed their ARR submissions to a venue.

```bash
python generate_commitment_report.py \
  --username "you@email.com" \
  --password "••••••••" \
  --me "~Your_Name1" \
  --venue_id "aclweb.org/ACL/2025/Conference"
```

Output: `./reports/commitment_report.html`

Open either file in your browser — no server required.

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--username` | OpenReview username / email | env `OPENREVIEW_USERNAME` |
| `--password` | OpenReview password | env `OPENREVIEW_PASSWORD` |
| `--me` | Your OpenReview profile ID, e.g. `~Jane_Doe1` | env `OPENREVIEW_ID` |
| `--venue_id` | OpenReview venue ID | — |
| `--output_dir` | Directory for the generated report | `./reports` |

### Environment variables

To avoid repeating credentials on every run, export them once:

```bash
export OPENREVIEW_USERNAME="you@email.com"
export OPENREVIEW_PASSWORD="••••••••"
export OPENREVIEW_ID="~Your_Name1"

python generate_review_report.py --venue_id "aclweb.org/ACL/ARR/2025/February"
python generate_commitment_report.py --venue_id "aclweb.org/ACL/2025/Conference"
```

See `run.sh` for a ready-made example.

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

---

## Configuring the low-confidence threshold

The low-confidence flag triggers when any reviewer on a paper has a confidence score at or below the threshold. The default is **2**. To change it, edit the class variable near the top of `arr_report_generator.py`:

```python
class ARRReportGenerator:
    LOW_CONF_THRESHOLD = 2   # ← change this
```

---

## Credits

- Based on the ARR Tool by [Yiming Cui](https://ymcui.com/)

## License

MIT
