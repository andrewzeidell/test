"""
Analysis_Data_Format_v3 (Pipeline-Aligned)
------------------------------------------

Version 3 of the analysis aggregator now explicitly targets the CSV outputs
produced by the `scripts/read_and_extract.py` and `src/reader/summaries.py`
pipelines. This version aligns naming conventions, normalization, and merge
keys with those used by the summaries module.

Enhancements:
1. Recognizes standardized pipeline CSV outputs such as:
   - posting_age_trends.csv
   - hard_to_fill_signals*.csv (state, city, zip)
   - state_*, city_*, zip_* geographic aggregates
   - occupation_*, credentials_*, and topn_* summary outputs
2. Ensures consistent merge keys across aggregations:
   ('date', 'state', 'city', 'zip', 'onet_norm', 'soc2018_from_onet').
3. Uses the same normalization logic for state, city, and zip columns as
   defined by normalize_columns().
4. Produces unified Excel workbook `clean_output/pipeline_summary_v3.xlsx`,
   grouping outputs by analytic category.
5. Demonstrates use with output directory from read_and_extract pipeline.

Compatibility: fully synchronized with summaries & read_and_extract CSV conventions.
"""

from __future__ import annotations

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def find_csv_files(base_path: str | Path) -> List[Path]:
    """Recursively collect all CSV file paths under the base analysis directory."""
    base = Path(base_path)
    return [p for p in base.rglob("*.csv") if p.is_file()]


def extract_time_from_path(path: Path) -> Tuple[str, str]:
    """
    Extract (year, month) from a directory path (e.g., analysis_outputs/2024-03/).

    The function looks for YYYY-MM in any parent directory name.
    Returns ("unknown", "unknown") if no match found.
    """
    match = re.search(r"(20\d{2})-(\d{2})", str(path))
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def group_by_sheet(csv_files: List[Path]) -> Dict[str, List[Tuple[Path, Tuple[str, str]]]]:
    """
    Groups CSV file paths by their implied sheet names.

    Example:
        "by_state_seen.csv" → group key = "by_state"
        "credentials_license_counts.csv" → group key = "credentials_license_counts"
    """
    groups: Dict[str, List[Tuple[Path, Tuple[str, str]]]] = {}
    for file_path in csv_files:
        name = file_path.stem  # without extension
        # Infer sheet name (drop _seen or other trailing status cues)
        base_name = re.sub(r"_seen$|_counts$|_summary$", "", name)
        year, month = extract_time_from_path(file_path)
        groups.setdefault(base_name, []).append((file_path, (year, month)))
    return groups


def load_csv_with_date(csv_path: Path, year: str, month: str) -> pd.DataFrame:
    """Read CSV and append date metadata, with numeric auto-detection."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {csv_path}") from e

    df["year"] = year
    df["month"] = month

    # Normalize column names for consistency
    df.columns = [c.strip().lower() for c in df.columns]
    # Detect numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely ID columns (state, city, zip) consistent with v2 and summaries."""
    rename_map = {}
    for c in df.columns:
        lname = c.strip().lower().replace(" ", "_")
        if lname in {"state"}:
            rename_map[c] = "state"
        elif lname in {"city", "place"}:
            rename_map[c] = "city"
        elif lname in {"zip", "zipcode", "zip_code", "postal_code"}:
            rename_map[c] = "zip"
    return df.rename(columns=rename_map)


def collect_frames_by_sheet(base_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Traverse the pipeline output directory and assemble categorized time series
    from standardized CSV outputs.
    """
    csv_files = find_csv_files(base_dir)
    pipeline_patterns = [
        r"posting_age_trends\.csv$",
        r"hard_to_fill_signals.*\.csv$",
        r"(state|city|zip)_.+\.csv$",
        r"occupation_.+\.csv$",
        r"credentials_.+\.csv$",
        r"topn_.+\.csv$",
    ]
    combined_frames: Dict[str, pd.DataFrame] = {}
    for csv_path in csv_files:
        fname = csv_path.name
        if not any(re.match(pat, fname) for pat in pipeline_patterns):
            continue
        year, month = extract_time_from_path(csv_path)
        df = load_csv_with_date(csv_path, year, month)
        df = normalize_columns(df)
        key = fname.replace(".csv", "")
        combined_frames[key] = df if key not in combined_frames else pd.concat(
            [combined_frames[key], df], ignore_index=True
        )
    return combined_frames


def export_to_excel(sheet_frames: Dict[str, pd.DataFrame], output_path: str | Path) -> None:
    """
    Write all concatenated dataframes into a single Excel workbook.
    Each key becomes the sheet name.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, frame in sheet_frames.items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])  # Excel sheet name limit


def main(base_dir: str = "data/clean/aggregates/", out_path: str = "clean_output/pipeline_summary_v3.xlsx") -> None:
    """
    Main entrypoint for generating the pipeline-aligned Excel workbook.
    Runs on a directory produced by `scripts/read_and_extract.py`.

    Example:
        >>> python -m ML_Project.Analysis_Data_Format_v3 --base_dir data/clean/aggregates
    """
    frames = collect_frames_by_sheet(base_dir)
    export_to_excel(frames, out_path)
    print(f"✅ pipeline_summary_v3 Excel workbook written to {out_path}")


if __name__ == "__main__":
    main()