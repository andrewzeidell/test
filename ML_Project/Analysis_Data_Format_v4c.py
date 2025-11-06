"""
Analysis_Data_Format_v4c (Counts Aggregation)
---------------------------------------------
Refined from v4 to produce compact, counts-based timeseries CSVs matching
data/raw/timeseries_* example outputs.

Differences from v4:
- No Excel exports.
- Aggregates numeric metrics into count-based summaries.
- Directly writes `timeseries_*_by_state.csv`, etc.
- Adds a national monthly totals file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def find_csv_files(base_path: str | Path) -> List[Path]:
    base = Path(base_path)
    return [p for p in base.rglob("*.csv") if p.is_file()]


def extract_time_from_path(path: Path) -> Tuple[str, str]:
    match = re.search(r"(20\\d{2})-(\\d{2})", str(path))
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def load_and_tag_csv(csv_path: Path, year: str, month: str) -> pd.DataFrame:
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


def collect_frames(base_dir: Path) -> Dict[str, pd.DataFrame]:
    files = find_csv_files(base_dir)
    groups: Dict[str, List[pd.DataFrame]] = {}

    for f in files:
        year, month = extract_time_from_path(f)
        df = load_and_tag_csv(f, year, month)
        df = normalize_columns(df)
        key = f.stem
        groups.setdefault(key, []).append(df)

    merged = {}
    for key, frames in groups.items():
        merged[key] = pd.concat(frames, ignore_index=True)
    return merged


def aggregate_counts(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Aggregate data to counts or numeric summaries per month × geography × STEM group.
    """
    dim_cols = ["year", "month", "state"]
    if level in {"city", "zip"} and level in df.columns:
        dim_cols.append(level)
    if "stem_group" in df.columns:
        dim_cols.append("stem_group")

    # Create date for consistent export
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    # Try to identify numeric metrics
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    agg_map = {c: "mean" for c in num_cols if c not in dim_cols}
    agg_map["date"] = "first"

    grouped = df.groupby(dim_cols, dropna=False).agg(agg_map).reset_index()
    grouped["postings_count"] = df.groupby(dim_cols, dropna=False).size().values
    grouped = grouped.sort_values(["date", *dim_cols], ignore_index=True)
    return grouped


def export_counts(frames: Dict[str, pd.DataFrame], out_dir: str | Path = "data/results") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_files = []

    for key, df in frames.items():
        level = "state" if "state" in key else "city" if "city" in key else "zip" if "zip" in key else None
        if level is None:
            continue

        agg_df = aggregate_counts(df, level)
        out_file = out_path / f"timeseries_STEMGroups_by_{level}.csv"
        agg_df.to_csv(out_file, index=False)
        output_files.append(out_file)
        print(f"✅ Exported counts-based file → {out_file}")

    # create national totals
    total_frames = []
    for key, df in frames.items():
        if "state" not in key:
            continue
        if {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
            )
            total = (
                df.groupby(["year", "month"], dropna=False)
                .size()
                .reset_index(name="postings_count")
            )
            total_frames.append(total)

    if total_frames:
        national = pd.concat(total_frames, ignore_index=True)
        national["date"] = pd.to_datetime(
            national["year"].astype(str) + "-" + national["month"].astype(str).str.zfill(2) + "-01"
        )
        totals_path = out_path / "timeseries_all_states_monthly_totals.csv"
        national.to_csv(totals_path, index=False)
        print(f"✅ Wrote national monthly totals → {totals_path}")


def main(
    base_dir: str = "data/clean/aggregates/",
    out_dir: str = "data/results/",
):
    frames = collect_frames(Path(base_dir))
    export_counts(frames, out_dir)
    print("🏁 v4c counts-based export complete →", out_dir)


if __name__ == "__main__":
    main()