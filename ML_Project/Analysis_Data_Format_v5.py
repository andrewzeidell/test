"""
Analysis_Data_Format_v5 (Restored Breadth + Modern Aggregation)
---------------------------------------------------------------
Reintegrates the structured richness and dataset coverage of v2
with the aggregation and counts-awareness of later versions.

Outputs:
    data/results/timeseries_<analytic_class>.csv
    data/results/timeseries_all_states_monthly_totals.csv

Maintains:
- Multi-field structures (occupation, credential, pay, education, etc.)
- Automatic geography detection (state, city, zip)
- Time metadata extraction from filename or path
- Single-folder flat CSV outputs, periodized monthly

Dependencies: pandas, pathlib, re
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# ---------------------------------------------------------------------
# Helpers: discovery + normalization
# ---------------------------------------------------------------------
def find_csv_files(base_path: str | Path) -> List[Path]:
    base = Path(base_path)
    return [p for p in base.rglob("*.csv") if p.is_file()]


def extract_time_from_path(path: Path) -> Tuple[str, str]:
    """Extract (year, month) from path text (YYYY-MM)."""
    match = re.search(r"(20\d{2})-(\d{2})", str(path))
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def load_csv_with_date(csv_path: Path, year: str, month: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"], df["month"] = year, month
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"state"}:
            rename[c] = "state"
        elif lc in {"city", "place"}:
            rename[c] = "city"
        elif lc in {"zip", "zipcode", "zip_code", "postal_code"}:
            rename[c] = "zip"
    return df.rename(columns=rename)


# ---------------------------------------------------------------------
# Collection + merging logic (modeled after v2)
# ---------------------------------------------------------------------
def collect_frames_by_directory(base_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Collect and merge CSVs grouped by directory rather than inferred sheet names.
    Focuses on `by_state_*` files — one category per subfolder.
    Only processes state-level data by design for now.
    """
    all_csvs = find_csv_files(base_dir)
    collected: Dict[str, List[pd.DataFrame]] = {}

    for csv_path in all_csvs:
        name = csv_path.name.lower()
        if not name.startswith("by_state_"):
            continue  # skip non-state files
        # extract the "thing" (category) after by_state_
        match = re.match(r"by_state_(.+)\.csv$", name)
        if match:
            category = match.group(1).replace("_summary", "").replace("_seen", "").replace("_counts", "")
        else:
            category = "unknown"

        year, month = extract_time_from_path(csv_path)
        df = load_csv_with_date(csv_path, year, month)
        df = normalize_columns(df)
        collected.setdefault(category, []).append(df)

    # Merge all monthly frames per category
    merged: Dict[str, pd.DataFrame] = {}
    for key, dfs in collected.items():
        merged[key] = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(dfs)} CSV(s) for category [{key}] at state-level")

    return merged


# ---------------------------------------------------------------------
# Aggregation logic: generalized flexible grouping
# ---------------------------------------------------------------------
def aggregate_time_geography(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize by date + geography (state/city/zip) + optional STEM or category fields.
    Keeps all numeric aggregation columns as mean and retains total counts.
    """
    geo_cols = [c for c in ["state", "city", "zip"] if c in frame.columns]
    category_cols = [c for c in frame.columns if "stem" in c or "jobclass" in c or "onet" in c]
    time_cols = [c for c in ["year", "month"] if c in frame.columns]
    group_cols = time_cols + geo_cols + category_cols

    df = frame.copy()
    if {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

    num_cols = df.select_dtypes(include=["int", "float"]).columns
    agg_dict = {col: "mean" for col in num_cols if col not in group_cols}
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    grouped["postings_count"] = df.groupby(group_cols, dropna=False).size().values
    return grouped


# ---------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------
def export_timeseries(frames: Dict[str, pd.DataFrame], out_dir: str | Path = "data/results/") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    national_summaries = []

    for key, df in frames.items():
        print(f"🧮 Aggregating: {key}")
        aggregated = aggregate_time_geography(df)
        # Determine geography present
        geos = [g for g in ["state", "city", "zip"] if g in aggregated.columns]
        if not geos:
            continue
        for g in geos:
            subset = aggregated.dropna(subset=[g])

            # ensure date column exists and is singular (not year/month)
            if {"year", "month"}.issubset(subset.columns):
                subset["date"] = pd.to_datetime(
                    subset["year"].astype(str) + "-" + subset["month"].astype(str).str.zfill(2) + "-01"
                )
            subset = subset.drop(columns=[c for c in ["year", "month"] if c in subset.columns])
            subset = subset.sort_values("date", ascending=True, ignore_index=True)

            # clean up duplicate key suffixes like _by_state_by_state
            clean_key = re.sub(r"_by_(state|city|zip).*", "", key)
            file_path = out_path / f"timeseries_{clean_key}_by_{g}.csv"

            subset.to_csv(file_path, index=False)
            print(f"✅ Exported → {file_path}")

            if g == "state":
                national_summaries.append(subset)

    # national totals (sum across all state files)
    if national_summaries:
        combined = pd.concat(national_summaries, ignore_index=True)
        if "date" not in combined.columns and {"year", "month"}.issubset(combined.columns):
            combined["date"] = pd.to_datetime(
                combined["year"].astype(str) + "-" + combined["month"].astype(str).str.zfill(2) + "-01"
            )
        # drop groupings and sum postings_count across all STEM groupings
        totals = (
            combined.groupby("date", dropna=False)["postings_count"]
            .sum()
            .reset_index()
            .sort_values("date", ignore_index=True)
        )
        totals.to_csv(out_path / "timeseries_all_states_monthly_totals.csv", index=False)
        print("🌎 National monthly totals written (sum of all postings across states/STEM groups).")


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main(
    base_dir: str = "data/clean/aggregates/",
    out_dir: str = "data/results/",
):
    frames = collect_frames_by_directory(base_dir)
    export_timeseries(frames, out_dir)
    print("🏁 Analysis_Data_Format_v5 complete.")


if __name__ == "__main__":
    main()