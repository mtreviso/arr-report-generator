# ARR Report Generator

HTML dashboards for OpenReview ARR/ACL venues for Senior Area Chairs, Area Chairs, and Program Chairs. Covers both the **review phase** and the **commitment phase**.

**Runs entirely on your machine.** Your OpenReview credentials are only used to fetch data and are never stored or transmitted elsewhere.

📄 See [the doc page](https://mtreviso.github.io/arr-report-generator/docs.html) for a simple walkthrough, or use the README below for the full reference.

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
  --username "your@email.com" \
  --password "yourpassword" \
  --venue-id "aclweb.org/ACL/ARR/2026/January" \
  --me "~Your_Name1"
```

`--role sac` (default): only papers in your SAC batch.  
`--role ac`: papers where you are AC.
`--role pc`: all papers in the venue (PC/test mode, may take several minutes).

### Commitment Report (ACL/EMNLP/etc. venue)

Add the `--phase commitment` flag to fetch the commitment phase data instead of the review phase.
Note that SACs are now ACs, so the `--role ac` flag is required to see any papers at all.

```bash
python generate_review_report.py \
  --username "your@email.com" \
  --password "yourpassword" \
  --venue-id "aclweb.org/ACL/2026/Conference" \
  --me "~Your_Name1" \
  --phase "commitment" \
  --role "ac"
```

If you know the ARR venue ID that commitment papers link back to, pass `--linked-venue-id` for significantly faster data fetching (bulk pre-fetches linked ARR data instead of per-paper lookups):

```bash
python generate_review_report.py \
  ... \
  --phase "commitment" \
  --linked-venue-id "aclweb.org/ACL/ARR/2026/February"
```

### PC Dashboard (ARR review phase)

Fetches **all** papers across all SAC batches. Intended for Program Chairs. May take several minutes for large venues (6000+ papers).

```bash
python generate_pc_report.py \
  --username "your@email.com" \
  --password "yourpassword" \
  --venue-id "aclweb.org/ACL/ARR/2026/January" \
  --me "~Your_Name1"
```

### PC Commitment Dashboard (ACL/EMNLP/etc. commitment phase)

`generate_pc_report.py` also supports the commitment phase via `--phase commitment`. This generates the full PC dashboard across all linked commitment submissions and writes `pc_commitment_report.html`.

```bash
python generate_pc_report.py \
  --username "your@email.com" \
  --password "yourpassword" \
  --venue-id "aclweb.org/ACL/2026/Conference" \
  --me "~Your_Name1" \
  --phase "commitment" \
  --linked-venue-id "aclweb.org/ACL/ARR/2026/January"
```

### Credentials via environment variables

All scripts read these env vars if flags are not supplied:

```bash
export OPENREVIEW_USERNAME="your@email.com"
export OPENREVIEW_PASSWORD="yourpassword"
export OPENREVIEW_ID="~Your_Name1"
python generate_review_report.py --venue-id aclweb.org/ACL/ARR/2025/February
```

---

## Common options

| Flag | Description |
|------|-------------|
| `--username` / `--password` | OpenReview credentials (or set via `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` env vars). |
| `--venue-id` | Full venue group ID, e.g. `aclweb.org/ACL/ARR/2025/February`. |
| `--me` | Your OpenReview tilde ID, e.g. `~Your_Name1` (or set via `OPENREVIEW_ID`). |
| `--role` | `sac` (default for review) / `ac` / `pc` — scope of papers to fetch. |
| `--phase` | `review` (default) / `commitment` — which phase to generate a report for. |
| `--linked-venue-id` | ARR venue ID for bulk pre-fetching linked submissions (commitment phase only). |
| `--comments-level` | `none` / `basic` / `full` (default) — higher detail means slower runs and larger files. |
| `--output-dir` | Output directory for generated reports. Default: `./reports`. |
| `--append-date` | Append today's date (YYYY-MM-DD) to the output filename to avoid overwriting. |
| `--save-cache` / `--use-cache` | Cache OpenReview data to disk; reload instantly on the next run. |
| `--cache-dir` | Custom directory for cache files. Auto-generated under `.dev_cache/` if omitted. |
| `--impersonate [GROUP_ID]` | Lets PCs inspect another SAC's view. Requires PC-level permission. |

## Comments levels

All report scripts support `--comments-level none|basic|full`.

- `none`: omit comments tab and skip comment processing
- `basic`: direct replies only 
- `full` (default): full reply threads — slowest / largest output


### Output files

| Script | Phase | Output |
|--------|-------|--------|
| `generate_review_report.py` | review | `reports/<venue>_review_report.html` |
| `generate_review_report.py` | commitment | `reports/<venue>_commitment_report.html` |
| `generate_pc_report.py` | review | `reports/<venue>_pc_report.html` |
| `generate_pc_report.py` | commitment | `reports/<venue>_pc_commitment_report.html` |

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

`--cache-dir` controls where the pickle files go. If you omit it, the tool auto-generates a phase- and report-aware cache path under `.dev_cache/` so review vs commitment and review vs PC runs do not collide. You can still maintain multiple named caches manually, e.g. one per SAC:

```bash
python generate_review_report.py ... --impersonate "~SAC_Name1" --save-cache --cache-dir .cache_sac1
python generate_review_report.py ... --use-cache --cache-dir .cache_sac1
```

The cache stores three files: `submissions.pkl`, `group_index.pkl`, and `processed.pkl`. By default, cache directories are automatically separated by report type, phase, venue, and role/impersonation where relevant.

> `--save-cache` and `--use-cache` are mutually exclusive.

## Impersonation (view as a SAC)

The review and commitment reports support impersonating another OpenReview user before fetching data. This requires **PC-level permission** on the venue and uses the [OpenReview impersonate API](https://docs.openreview.net/reference/api-v2/openapi-definition#post-impersonate).

```bash
python generate_review_report.py ... --phase "review" --me "~Target_SAC_Name1" --impersonate
python generate_review_report.py ... --phase "commitment" --me "~Target_SAC_Name1" --impersonate
```

You can pass a `GROUP_ID` after the `--impersonate` flag, which corresponds to the authorization group, e.g., `aclweb.org/ACL/ARR/2025/October/Program_Chairs`. If `GROUP_ID` is empty, it will assume to be `<venue-id>/Program_Chairs`. Your own credentials are used to authenticate, and all data is then fetched as the target user. 

This is useful for verifying exactly what a SAC sees without asking them to run the script themselves. Can be combined with `--save-cache` / `--use-cache` for fast iteration.

> Impersonation is intentionally not available in `generate_pc_report.py`, which already fetches all papers by design.


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
