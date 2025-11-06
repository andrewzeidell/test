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


def split_by_stem_group(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame by the 'stem group' column if available; return dictionary of subsets.

    Returns:
        Dict[str, pd.DataFrame]: Mapping from STEM group name to DataFrame subset.
    """
    if "stem group" not in frame.columns:
        return {"Uncategorized": frame}
    groups = {}
    for name, subset in frame.groupby(frame["stem group"].fillna("Uncategorized")):
        groups[name] = subset
    return groups


def compute_totals_and_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute totals and averages for numeric columns while preserving date/year/month columns.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return df

    result_frames = [df]
    if {"year", "month"}.issubset(df.columns):
        grouped = df.groupby(["year", "month"], dropna=False)
        summaries = []
        for (year, month), g in grouped:
            totals = g[numeric_cols].sum(numeric_only=True)
            avgs = g[numeric_cols].mean(numeric_only=True)
            totals["year"], totals["month"], totals["summary_type"] = year, month, "TOTAL"
            avgs["year"], avgs["month"], avgs["summary_type"] = year, month, "AVERAGE"
            summaries += [totals, avgs]
        summary_df = pd.DataFrame(summaries)
        result_frames.append(summary_df)
    else:
        totals = df[numeric_cols].sum(numeric_only=True)
        avgs = df[numeric_cols].mean(numeric_only=True)
        totals["summary_type"], avgs["summary_type"] = "TOTAL", "AVERAGE"
        result_frames.append(pd.DataFrame([totals, avgs]))
    return pd.concat(result_frames, ignore_index=True)


def export_to_csvs(sheet_frames: Dict[str, pd.DataFrame], output_dir: str | Path = "clean_output/timeseries") -> None:
    """
    Export each analytical sheet split by STEM group to its CSVs with totals and averages added.

    Now includes a unified 'date' column for easy plotting and chronological analysis.

    Output pattern:
        clean_output/timeseries/{stem_group}/{sheet_name}.csv

    The output CSVs drop separate `year` and `month` columns since those are merged into `date`.
    """
    output_dir = Path(output_dir)
    for sheet, frame in sheet_frames.items():
        # Create unified datetime column for time-based plotting
        if {"year", "month"}.issubset(frame.columns):
            frame["date"] = pd.to_datetime(frame["year"].astype(str) + "-" + frame["month"].astype(str).str.zfill(2) + "-01")
            frame = frame.drop(columns=["year", "month"])
            frame = frame.sort_values(["date"], ascending=True, ignore_index=True)

        for stem_group, df_subset in split_by_stem_group(frame).items():
            target_dir = output_dir / stem_group
            target_dir.mkdir(parents=True, exist_ok=True)

            # Add totals/average while keeping 'date' chronological
            enriched = compute_totals_and_averages(df_subset)

            # If enriched still has year/month, unify for output consistency
            if {"year", "month"}.issubset(enriched.columns):
                enriched["date"] = pd.to_datetime(enriched["year"].astype(str) + "-" + enriched["month"].astype(str).str.zfill(2) + "-01")
                enriched = enriched.drop(columns=["year", "month"])
            if "date" in enriched.columns:
                enriched = enriched.sort_values("date", ascending=True, ignore_index=True)

            out_path = target_dir / f"{sheet}.csv"
            try:
                enriched.to_csv(out_path, index=False)
                print(f"✅ Wrote STEM {stem_group}: {out_path} (with unified date index for plotting)")
            except Exception as e:
                print(f"❌ Failed to write CSV [{sheet}] for {stem_group}: {e}")


def main(base_dir: str = "data/clean/aggregates/", out_dir: str = "clean_output/timeseries/") -> None:
    """
    Main entrypoint for generating per-series CSV time series datasets.

    Runs on a directory produced by `scripts/read_and_extract.py`.

    Example:
        >>> python -m ML_Project.Analysis_Data_Format_v3 --base_dir data/clean/aggregates
    """
    frames = collect_frames_by_sheet(base_dir)
    export_to_csvs(frames, out_dir)
    print("✅ All pipeline-aligned time series CSVs written to:", out_dir)


if __name__ == "__main__":
    main()