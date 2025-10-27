import os
import re
import pandas as pd
from pathlib import Path
import xlsxwriter
import openpyxl

# ==== EDIT THESE ====
BASE_PATH = r"analysis_outputs" # folder that contains dated subfolders
CSV_NAME = "by_state_seen.xlsx" # same filename present in every dated subfolder
OUT_BASENAME = "clean_output/timeseries" # output prefix (no extension)
# Optionally pin which value columns to include (leave [] to auto-detect numeric columns)
VALUE_COLUMNS = [] # e.g., ["Total", "Count", "Amount"]
# ====================

# Helper: standardize likely id column names
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lower + strip spaces/underscores for matching, but keep original names
    orig = df.columns
    low = [c.strip().lower().replace(" ", "_") for c in orig]

    rename = {}
    for c, lc in zip(orig, low):
        if lc in {"state"}:
            rename[c] = "state"
        elif lc in {"city", "place"}:
            rename[c] = "city"
        elif lc in {"zip", "zipcode", "zip_code", "postal_code"}:
            rename[c] = "zip"
    return df.rename(columns=rename)

# Parse dated folder names like YYYY_MMmm where first MM=start, last mm=end
# DATE_RE = re.compile(r"_(?P<year>\d{4})_(?P<start>\d{2})(?P<end>\d{2})$")
DATE_RE = re.compile(r'(?<!\d)\d{4}-(0[1-9]|1[0-2])(?!\d)')

# MODIFIED: Now returns a dictionary of dataframes, one for each sheet found.
# The keys are the sheet names.
def collect_frames_by_sheet(base: str, excel_name: str) -> dict[str, list[pd.DataFrame]]:
    # This dictionary will map sheet names to a list of dataframes (one per dated folder)
    # E.g., {"Sheet1": [df_date1, df_date2], "Sheet2": [df_date1, df_date2]}
    all_frames_by_sheet = {}

    for entry in os.listdir(base):
        match = DATE_RE.search(entry)

        if not match:
            # print(f"Skipping non-dated folder: {entry}") # uncomment for debugging
            continue
        year = entry.split("_")[-1].split("-")[0]
        start_mm = entry.split("_")[-1].split("-")[-1].split(".")[0]
        # year = match.group("year")
        # start_mm = match.group("start")
        # end_month = match.group("end") # not used, but kept for consistency

        excel_path = Path(base) / entry / excel_name
        if not excel_path.exists():
            # print(f"Skipping {entry}: {excel_name} not found.") # uncomment for debugging
            continue

        # Read ALL sheets from the Excel file into a dictionary {sheet_name: dataframe}
        try:
            # sheet_name=None reads all sheets
            dfs_from_excel = pd.read_excel(excel_path, sheet_name=None)
        except Exception as e:
            print(f"Error reading {excel_path}: {e}")
            continue

        for sheet_name, df_sheet in dfs_from_excel.items():
            # --- Per Sheet Processing ---
            # Normalize columns for consistency (state, city, zip)
            df_sheet = normalize_columns(df_sheet)

            # Add time metadata
            df_sheet["year"] = year
            df_sheet["month"] = start_mm
            # monthly timestamp using first of month (good index key)
            df_sheet["date"] = pd.to_datetime(df_sheet["year"].astype(str) + "-" + df_sheet["month"].astype(str).str.zfill(2) + "-01")

            # Store the dated dataframe into the list for its corresponding sheet
            if sheet_name not in all_frames_by_sheet:
                all_frames_by_sheet[sheet_name] = []
            all_frames_by_sheet[sheet_name].append(df_sheet)

    if not all_frames_by_sheet:
        raise FileNotFoundError("No matching dated folders/Excels found or no data sheets. Check BASE_PATH/CSV_NAME and folder format YYYY_MMmm.")

    # Now, combine the frames for each sheet and sort them
    combined_by_sheet = {}
    for sheet_name, rows in all_frames_by_sheet.items():
        combined = pd.concat(rows, ignore_index=True)
        combined = combined.sort_values(["date"]).reset_index(drop=True)
        combined_by_sheet[sheet_name] = combined

    return combined_by_sheet

def pick_value_columns(df: pd.DataFrame, explicit_cols=None):
    if explicit_cols:
        missing = [c for c in explicit_cols if c not in df.columns]
        if missing:
            # IMPORTANT: For multi-sheet, we should log a warning instead of raising an error
            # as VALUE_COLUMNS might be valid for *other* sheets.
            print(f"WARNING: VALUE_COLUMNS not found in current sheet's data: {missing}")
        return [c for c in explicit_cols if c in df.columns] # only return existing ones
    # Auto-pick numeric columns that are not id/time columns
    exclude = {"state", "city", "zip", "year", "month", "date"}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in num_cols if c not in exclude]


# ==============================================================================
# NEW ANALYSIS: Timeseries and Percentages for a Single Sheet (Finalized)
# ==============================================================================

# Helper to filter the combined dataframe for a specific sheet and column
def calculate_education_timeseries(
        combined_df_by_sheet: dict[str, pd.DataFrame],
        target_sheet: str,
        value_col: str,  # The column containing the counts (e.g., 'Count')
        group_col: str,  # The column containing the education categories (e.g., 'Education Type')
        unspecified_label: str = "Unspecified"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates two tables: one for raw counts and one for percentages for the specified sheet.
    It pivots the data by the grouping column to create the time series.
    """
    if target_sheet not in combined_df_by_sheet:
        print(f"Sheet '{target_sheet}' not found in data.")
        return pd.DataFrame(), pd.DataFrame()

    df = combined_df_by_sheet[target_sheet].copy()

    # Normalize columns to handle common name variations for the group/value columns
    # We must assume the user set the right names in main()
    if group_col not in df.columns or value_col not in df.columns:
        print(f"Required columns '{group_col}' or '{value_col}' missing in sheet '{target_sheet}'.")
        return pd.DataFrame(), pd.DataFrame()

    # --- FIX: Clean the text values in the grouping column (Education Type) ---
    # Convert the grouping column to string type (if not already)
    df[group_col] = df[group_col].astype(str)

    # Replace common non-standard apostrophes (e.g., typographic quote ’) with a straight quote (').
    # This ensures categories like "Master’s" and "Master's" are grouped together.
    df[group_col] = df[group_col].str.replace("’", "'").str.replace("‘", "'")

    # Optional: strip whitespace, convert to title case, or convert to lowercase for consistency
    df[group_col] = df[group_col].str.strip()
    # --------------------------------------------------------------------------

    # --- 1. Raw Counts Timeseries (Pivot Date vs. Category) ---
    # The 'date' column is in the index because it was added in collect_frames_by_sheet.
    counts_table = df.pivot_table(
        index="date",
        columns=group_col,
        values=value_col,
        aggfunc="sum",
        observed=True
    ).sort_index(axis=0)

    # --- 2. Percentage Calculation (Excluding 'Unspecified') ---

    # Filter out the 'Unspecified' rows/category for percentage basis
    df_for_pct = df[df[group_col].astype(str) != unspecified_label].copy()

    if df_for_pct.empty:
        print(f"No data remaining after excluding '{unspecified_label}' for percentage calculation.")
        return counts_table, pd.DataFrame()

    # Calculate the Total count PER DATE (the denominator)
    # This sums up the counts of ALL SPECIFIED categories for each date.
    date_totals = df_for_pct.groupby("date")[value_col].sum().rename("Total_Excl_Unspecified")

    # Pivot the specified counts for the numerator (Date, Category)
    numerator = df_for_pct.pivot_table(
        index="date",
        columns=group_col,
        values=value_col,
        aggfunc="sum",
        observed=True
    ).sort_index(axis=0)

    # Calculate percentage: (Category Count / Total Count for that Date)
    percentages = (numerator.div(date_totals, axis=0) * 100).round(2)

    # Rename columns for clarity in the output
    percentages.columns = [f"{col}_pct" for col in percentages.columns]

    return counts_table, percentages


# ==============================================================================

def build_table(df: pd.DataFrame, group_col: str, value_cols: list, include_average = True, average_label = "AVG") -> pd.DataFrame:
    # Keep rows where the grouping column exists (some files might be by other grains)
    # Also ensure there are value columns to aggregate
    if not value_cols:
         print(f"WARNING: No value columns to aggregate for grouping column '{group_col}'.")
         return pd.DataFrame() # no value columns

    keep = df.dropna(subset=[group_col]) if group_col in df.columns else pd.DataFrame(columns=df.columns)
    if keep.empty:
        return pd.DataFrame() # no data for this grain

    # Use a pivot to put dates on rows and (metric, location) as columns
    pivoted = keep.pivot_table(
        index="date",
        columns=group_col,
        values=value_cols,
        aggfunc="sum", # sum across any duplicates within a (date, location)
        observed=True
    ).sort_index()

    if pivoted.empty: # Check if pivot table is empty after aggregation
        return pd.DataFrame()


    if include_average:
        # 1. Calculate and append AVERAGE - appended after total
        avg = pivoted.groupby(level=0, axis=1).mean(numeric_only=True)
        avg.columns = pd.MultiIndex.from_product([avg.columns, [average_label]])


        # 2. Calculate and append TOTAL - This part is FIXED
        tot_label = "TOTAL"

        # Sum across the location level (level 1) for each metric (level 0).
        # This collapses the location columns into one 'TOTAL' column for each metric.
        tot = pivoted.groupby(level=0, axis=1).sum(numeric_only=True)

        # The result (tot) now has flat columns named after the metrics (Total, Count, etc.)
        # We need to turn this flat index back into a MultiIndex (Metric, TOTAL)
        tot.columns = pd.MultiIndex.from_product([tot.columns, [tot_label]])

        # Concatenate the final total back to the pivoted table
        pivoted = pd.concat([pivoted, tot], axis=1)
        pivoted = pd.concat([pivoted, avg], axis=1)


    # Make column order deterministic: sort by metric then location
    pivoted = pivoted.sort_index(axis=1, level=list(range(pivoted.columns.nlevels)))
    # Optional: flatten MultiIndex columns to "metric|location" for easy CSV/Excel viewing
    pivoted.columns = ["{}|{}".format(*col) if isinstance(col, tuple) else str(col) for col in pivoted.columns]
    return pivoted


# ==============================================================================
# NEW: Function to collect data from a single, separate CSV file
# ==============================================================================

def collect_special_csv(base: str, csv_name: str) -> pd.DataFrame:
    """Collects and combines data from a single CSV present in all dated subfolders."""
    rows = []

    for entry in os.listdir(base):
        match = DATE_RE.search(entry)

        if not match:
            continue

        year = match.group("year")
        start_mm = match.group("start")

        csv_path = Path(base) / entry / csv_name
        if not csv_path.exists():
            continue

        try:
            # Read CSV instead of Excel
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            continue

        # Normalize columns (in case there are 'state', 'city', 'zip' columns)
        df = normalize_columns(df)

        # Add time metadata
        df["year"] = year
        df["month"] = start_mm
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No matching dated folders or CSVs found for: {csv_name}")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["date"]).reset_index(drop=True)
    return combined


# ==============================================================================

def main():
    # MODIFIED: Get a dictionary of combined DataFrames, keyed by sheet name
    combined_by_sheet = collect_frames_by_sheet(BASE_PATH, CSV_NAME)

    # Initialize a dictionary to store all time-series tables for the final output Excel
    output_tables = {}

    for sheet_name, combined_df in combined_by_sheet.items():
        print(f"--- Processing sheet: {sheet_name} ---")

        # Pick value columns for the current sheet
        value_cols = pick_value_columns(combined_df, VALUE_COLUMNS)

        if not value_cols:
             print(f"Skipping {sheet_name}: No valid value columns found after filtering.")
             continue

        # Build three separate tables for the current sheet
        state_table = build_table(combined_df, "state", value_cols)
        city_table = build_table(combined_df, "city", value_cols)
        zip_table = build_table(combined_df, "zip", value_cols)

        # Store the tables, prefixing the sheet name to maintain unique keys in the output
        if not state_table.empty:
            output_tables[f"{sheet_name}_by_state"] = state_table
        if not city_table.empty:
            output_tables[f"{sheet_name}_by_city"] = city_table
        if not zip_table.empty:
            output_tables[f"{sheet_name}_by_zip"] = zip_table

        # Optional: write separate CSVs (optional based on your need)
        # These will be named like 'timeseries_SheetName_by_state.csv'
        csv_prefix = f"{OUT_BASENAME}_{sheet_name}"
        if not state_table.empty:
            state_table.T.to_csv(f"{csv_prefix}_by_state.csv", index=True, date_format="%Y-%m-%d")
        if not city_table.empty:
            city_table.T.to_csv(f"{csv_prefix}_by_city.csv", index=True, date_format="%Y-%m-%d")
        if not zip_table.empty:
            zip_table.T.to_csv(f"{csv_prefix}_by_zip.csv", index=True, date_format="%Y-%m-%d")

    # --------------------------------------------------------------------------
    # --- 2. Run New Specialized Analysis (Education Timeseries) ---
    # --------------------------------------------------------------------------

    # !!! EDIT THESE TO MATCH YOUR COLUMN NAMES IN THE SPECIAL CSV FILE !!!
    SPECIAL_CSV_NAME = "education_counts.csv"  # <--- NEW FILENAME VARIABLE
    TARGET_VALUE_COL = "postings"
    TARGET_GROUP_COL = "req_min_edu"
    # --------------------------------------------------------------------------

    print(f"\n--- Running Specialized Analysis for CSV: {SPECIAL_CSV_NAME} ---")

    try:
        # STEP 2a: Collect the data from the separate CSV
        special_df = collect_special_csv(BASE_PATH, SPECIAL_CSV_NAME)

        # STEP 2b: Pass the single DataFrame to the calculation function
        # Note: We wrap it in a dictionary to reuse the existing calculation function structure.
        special_combined_by_sheet = {SPECIAL_CSV_NAME: special_df}

        counts_ts, percentages_ts = calculate_education_timeseries(
            special_combined_by_sheet,
            target_sheet=SPECIAL_CSV_NAME,  # Use the filename as the sheet key
            value_col=TARGET_VALUE_COL,
            group_col=TARGET_GROUP_COL
        )

        # Store the specialized tables
        output_tables[f"{SPECIAL_CSV_NAME.replace('.csv', '')}_counts_timeseries"] = counts_ts
        output_tables[f"{SPECIAL_CSV_NAME.replace('.csv', '')}_percentages"] = percentages_ts

    except FileNotFoundError as e:
        print(f"Skipping specialized analysis: {e}")
    except Exception as e:
        print(f"An error occurred during specialized analysis: {e}")

    # Final Output: single Excel with sheets for all processed tables
    output_filename = f"{OUT_BASENAME}_all_sheets.xlsx" # Changed filename to reflect multi-sheet output
    if output_tables:
        with pd.ExcelWriter(output_filename, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
            for sheet_name, table in output_tables.items():
                # Ensure sheet names don't exceed Excel's 31 character limit
                truncated_sheet_name = sheet_name[:31]
                table.T.to_excel(xw, sheet_name=truncated_sheet_name)
        print("\n==============================")
        print(f"Successfully wrote all time series tables to: {output_filename}")
        print("Sheets created:", list(output_tables.keys()))
    else:
        print("\n==============================")
        print("No time series tables were generated. Check input files and column names.")
    print("==============================")


if __name__ == "__main__":
    main()