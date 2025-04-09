# ARR Report Generator

This tool generates beautiful, interactive HTML reports for reviewing ARR submissions using OpenReview data. The report includes:

- **Papers Overview**: Complete status of all papers in your batch with interactive filtering/sorting
- **AC Dashboard**: Performance metrics for all Area Chairs
- **Comments & Issues**: Threaded view of all confidential comments and review issues
- **Analytics**: Score distributions and correlation analysis


## Screenshots

*Install and run the tool to see the beautiful report!*

## Installation

1. Clone this repository or download the source files:
   ```
   git clone <repository-url>
   cd arr-report-generator
   ```

2. Install required dependencies:
   ```
   pip install openreview-py pandas numpy jinja2 markdown
   ```

## Usage

### Basic Usage

Run the script with your OpenReview credentials:

```bash
python main.py --username "your_username" --password "your_password" --me "~Your_Name1" --venue_id "aclweb.org/ACL/ARR/2025/February"
```

The script will generate an HTML report in the `./reports` directory.

### Command-line Arguments

- `--username`: Your OpenReview username (can also be set via `OPENREVIEW_USERNAME` environment variable)
- `--password`: Your OpenReview password (can also be set via `OPENREVIEW_PASSWORD` environment variable)
- `--me`: Your OpenReview ID (e.g., `~Your_Name1`)
- `--venue_id`: The OpenReview venue ID (default: `aclweb.org/ACL/ARR/2025/February`)
- `--output_dir`: Directory to save the generated report (default: `./reports`)

### Environment Variables

Instead of providing credentials on the command line, you can set these environment variables:

```bash
export OPENREVIEW_USERNAME="your_username"
export OPENREVIEW_PASSWORD="your_password"
export OPENREVIEW_ID="~Your_Name1"
```

Then run the script without specifying these parameters:

```bash
python main.py --venue_id "aclweb.org/ACL/ARR/2025/February"
```

## Report Structure

The generated HTML report contains multiple sections accessible via tabs:

1. **Papers Overview**:
   - Interactive table with paper details
   - Filter by Area Chair
   - Sort by any column
   - Links to OpenReview forums

2. **AC Dashboard**:
   - Performance metrics for Area Chairs
   - Review completion status
   - Meta-review completion status

3. **Comments & Review Issues**:
   - Threaded view of all comments
   - Table view with filtering capabilities
   - Direct links to OpenReview

4. **Analytics**:
   - Score distribution chart
   - Correlation matrix of different scores

## Data Security Note

This tool runs entirely on your local machine. Your OpenReview credentials are only used to fetch data and are not stored or sent anywhere else.

## Credits

- Based on the ARR Tool by [Yiming Cui](https://ymcui.com/)

## License

MIT



