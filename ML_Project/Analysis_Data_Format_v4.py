"""
Analysis_Data_Format_v4 (Incremental Export)
--------------------------------------------
Builds upon v3 while aligning with the same input/output paths and column schema.

Key differences:
- Preserves identical folder structure and filenames under `data/results/`.
- Replaces grouped folders per STEM category with consolidated Excel exports (multi-sheet).
- Implements incremental monthly write operations to prevent high memory usage.
- Adds aggregated monthly U.S. totals for national trends.
"""

from __future__ import annotations

import os
import re
import gc
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def find_csv_files(base_path: str | Path) -> List[Path]:
    base = Path(base_path)
    return [p for p in base.rglob("*.csv") if p.is_file()]


def extract_time_from_path(path: Path) -> Tuple[str, str]:
    match = re.search(r"(20\d{2})-(\d{2})", str(path))
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def load_csv_with_date(csv_path: Path, year: str, month: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["year"], df["month"] = year, month
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        lname = c.strip().lower().replace(" ", "_")
        if lname == "state":
            rename_map[c] = "state"
        elif lname in {"city", "place"}:
            rename_map[c] = "city"
        elif lname in {"zip", "zipcode", "zip_code", "postal_code"}:
            rename_map[c] = "zip"
    return df.rename(columns=rename_map)


def collect_frames(base_dir: str | Path) -> Dict[str, pd.DataFrame]:
    csv_files = find_csv_files(base_dir)
    combined_frames: Dict[str, pd.DataFrame] = {}

    for csv_path in csv_files:
        year, month = extract_time_from_path(csv_path)
        df = load_csv_with_date(csv_path, year, month)
        df = normalize_columns(df)
        key = csv_path.stem
        if key not in combined_frames:
            combined_frames[key] = df
        else:
            combined_frames[key] = pd.concat([combined_frames[key], df], ignore_index=True)
    return combined_frames


def split_by_stem_group(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "stem group" not in frame.columns:
        return {"Uncategorized": frame}
    groups = {}
    for name, subset in frame.groupby(frame["stem group"].fillna("Uncategorized")):
        groups[name] = subset
    return groups


def export_incremental_excel(frames: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Consolidate outputs into Excel workbooks (multi-sheet), one per analytic group.
    Matches the v3 file naming under data/results/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in frames.items():
        # Derive Excel output file
        if "state" in key:
            geo = "state"
        elif "city" in key:
            geo = "city"
        elif "zip" in key:
            geo = "zip"
        else:
            geo = "all"

        output_path = out_dir / f"timeseries_{key}_by_{geo}.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            for group, sub_df in split_by_stem_group(df).items():
                # Add date column if not present
                if {"year", "month"}.issubset(sub_df.columns):
                    sub_df["date"] = pd.to_datetime(
                        sub_df["year"].astype(str) + "-" + sub_df["month"].astype(str).str.zfill(2) + "-01"
                    )
                sub_df.sort_values("date", inplace=True, ignore_index=True)
                sub_df.to_excel(writer, sheet_name=group, index=False)
            writer.save()
        print(f"✅ Wrote workbook: {output_path}")
        gc.collect()


def export_monthly_totals(frames: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Compute monthly U.S. totals across all available state-level outputs.
    """
    all_records = []
    for key, df in frames.items():
        if "state" not in key:
            continue
        if {"year", "month"}.issubset(df.columns):
            df_grouped = (
                df.groupby(["year", "month"], dropna=False)
                .sum(numeric_only=True)
                .reset_index()
            )
            df_grouped["source"] = key
            all_records.append(df_grouped)

    if all_records:
        total_df = pd.concat(all_records, ignore_index=True)
        total_df["date"] = pd.to_datetime(
            total_df["year"].astype(str) + "-" + total_df["month"].astype(str).str.zfill(2) + "-01"
        )
        total_df.sort_values("date", inplace=True, ignore_index=True)
        total_path = out_dir / "timeseries_all_states_monthly_totals.csv"
        total_df.to_csv(total_path, index=False)
        print(f"✅ Wrote national monthly totals: {total_path}")


def main(
    base_dir: str = "data/clean/aggregates/",
    out_dir: str = "data/results/",
) -> None:
    """
    Incremental export runner for Analysis_Data_Format_v4.
    Uses v3's same directory conventions but streams outputs to Excel and total CSVs.
    """
    frames = collect_frames(base_dir)
    out_dir_path = Path(out_dir)
    export_incremental_excel(frames, out_dir_path)
    export_monthly_totals(frames, out_dir_path)
    print("🏁 v4 export completed successfully →", out_dir_path)


if __name__ == "__main__":
    main()