"""
Analysis_Data_Format_v3
-----------------------

Extends functionality of Analysis_Data_Format_v2 to handle cases where the
individual Excel "sheets" from v2 are now stored as separate CSV files
within monthly or dated subdirectories.

Key improvements:
1. Recursively searches a base analysis directory for per-sheet CSVs
   (e.g., "by_state_seen.csv", "education_counts.csv").
2. Groups files by former sheet name inferred from filename prefixes.
3. Extracts date metadata (year-month) from the folder hierarchy.
4. Concatenates data across months per sheet to assemble time series.
5. Writes a consolidated Excel workbook, each sheet representing an 
   original logical table from the legacy Excel workflows.

Output: `clean_output/timeseries_v3.xlsx`
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


def collect_frames_by_sheet(base_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Traverse directories and build concatenated time series per former sheet name.
    """
    csv_files = find_csv_files(base_dir)
    grouped = group_by_sheet(csv_files)

    result_frames: Dict[str, pd.DataFrame] = {}
    for sheet_name, entries in grouped.items():
        frames = []
        for file_path, (y, m) in entries:
            df = load_csv_with_date(file_path, y, m)
            frames.append(df)
        if frames:
            result_frames[sheet_name] = pd.concat(frames, ignore_index=True)
    return result_frames


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


def main(base_dir: str = "analysis_outputs/", out_path: str = "clean_output/timeseries_v3.xlsx") -> None:
    """
    Example main entrypoint for building the v3 timeseries workbook.

    Example:
        >>> python -m ML_Project.Analysis_Data_Format_v3
    """
    frames = collect_frames_by_sheet(base_dir)
    export_to_excel(frames, out_path)
    print(f"✅ v3 timeseries workbook written to {out_path}")


if __name__ == "__main__":
    main()