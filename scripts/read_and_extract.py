"""
CLI script for reading, transforming, and extracting cleaned job posting data.

This script parses command-line arguments for input parquet directory, date range or pattern,
output directory, and output format (parquet or feather). It reads parquet files with minimal
columns, applies transformations, merges lookup tables, and saves the cleaned data.

Usage:
    python scripts/read_and_extract.py --input_dir data/raw --date_pattern 2022-06 --output_dir data/clean --format parquet
"""
import argparse
import logging
import os
import sys
from typing import Optional

import pandas as pd

from src.reader.etl import read_parquet_files
from src.reader.transforms import transformations
from src.reader.transforms import lookup_utils


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Read and extract cleaned job posting data.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing parquet files.",
    )
    parser.add_argument(
        "--date_pattern",
        type=str,
        required=False,
        default=None,
        help="Date pattern to filter parquet files by (e.g., '2022-06').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save cleaned files.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "feather"],
        default="parquet",
        help="Output file format (parquet or feather). Default is parquet.",
    )
    return parser.parse_args()


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations to the DataFrame.

    Note: Pay normalization transformation is removed to improve speed.
    """
    # Example: Add posting age days column
    df["posting_age_days"] = df.apply(
        lambda row: transformations.calculate_posting_age_days(
            row.get("date_acquired"), row.get("expired_date")
        ),
        axis=1,
    )

    # Removed pay normalization to improve speed

    # Normalize O*NET code
    df["onet_code_normalized"] = df["classifications_onet_code"].apply(
        transformations.normalize_onet_code
    )

    # Clean education, experience, license fields
    df["education_clean"] = df["requirements_min_education"].apply(
        transformations.clean_education_field
    )
    df["experience_clean"] = df["requirements_experience"].apply(
        transformations.clean_experience_field
    )
    df["license_clean"] = df["requirements_license"].apply(
        transformations.clean_license_field
    )

    # Determine ghost job flag
    df["is_ghost_job"] = df["ghostjob"].apply(transformations.is_ghost_job)

    # Determine remote job flag
    if "remote_flag" in df.columns:
        df["is_remote_job"] = df["remote_flag"].apply(transformations.is_remote_job)
    else:
        df["is_remote_job"] = False

    return df


from typing import List, Optional
import glob

def find_parquet_files(input_dir: str, date_pattern: Optional[str] = None) -> List[str]:
    """Recursively find parquet files in input_dir matching the optional date_pattern.

    Args:
        input_dir: Directory to search for parquet files.
        date_pattern: Optional substring to filter files by date pattern.

    Returns:
        List of matching parquet file paths.
    """
    pattern = "**/*.parquet"
    all_files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
    if date_pattern:
        filtered_files = [f for f in all_files if date_pattern in os.path.basename(f)]
    else:
        filtered_files = all_files
    return filtered_files


def process_file(file_path: str, output_dir: str, output_format: str) -> None:
    """Process a single parquet file: read, transform, merge lookups, and save.

    Args:
        file_path: Path to the parquet file.
        output_dir: Directory to save the cleaned output.
        output_format: Output file format ('parquet' or 'feather').
    """
    logging.info("Processing file: %s", file_path)
    try:
        columns = [
            "date_acquired",
            "expired_date",
            "parameters_salary_min",
            "parameters_salary_max",
            "parameters_salary_unit",
            "classifications_onet_code",
            "requirements_min_education",
            "requirements_experience",
            "requirements_license",
            "ghostjob",
            "remote_flag",
        ]
        df = read_parquet_files(file_path, None, columns)
        logging.info("Read %d rows from %s", len(df), file_path)

        df = apply_transformations(df)
        logging.info("Applied transformations.")

        # Load original lookup CSV files
        stem_groups = lookup_utils.load_lookup_csv("STEM Groups in the BLS Projections.csv")
        job_zones = lookup_utils.load_lookup_csv("ONET_Job_Zones.csv")
        # Optionally load SOC codes if needed
        # soc_codes = lookup_utils.load_lookup_csv("SOC_Codes.csv")

        # Normalize and merge STEM groups
        stem_groups = stem_groups.rename(columns={'SOC Code': 'soc2018_from_onet'})
        df['onet_raw'] = df['classifications_onet_code_25'].fillna(df['classifications_onet_code'])
        df['onet_norm'] = df['onet_raw'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()
        df['soc2018_from_onet'] = df['onet_norm'].astype(str).str.split('.').str[0]

        df = lookup_utils.merge_lookup_pandas(df, stem_groups, on='soc2018_from_onet')

        # Normalize and merge ONET job zones
        job_zones = job_zones.rename(columns={'Code': 'onet_norm'})
        df = lookup_utils.merge_lookup_pandas(df, job_zones, on='onet_norm')

        logging.info("Merged original lookup CSV files.")

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_cleaned.{output_format}")
        if output_format == "parquet":
            df.to_parquet(output_file, index=False)
        else:
            df.to_feather(output_file)
        logging.info("Saved cleaned data to %s", output_file)

    except Exception as e:
        logging.error("Error processing file %s: %s", file_path, e, exc_info=True)


def main() -> None:
    """Main function to run the CLI script or automatic directory scanning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if len(sys.argv) > 1:
        # Run with CLI arguments
        args = parse_args()
        logging.info("Starting read_and_extract with args: %s", args)
        try:
            columns = [
                "date_acquired",
                "expired_date",
                "parameters_salary_min",
                "parameters_salary_max",
                "parameters_salary_unit",
                "classifications_onet_code",
                "requirements_min_education",
                "requirements_experience",
                "requirements_license",
                "ghostjob",
                "remote_flag",
            ]
            df = read_parquet_files(args.input_dir, args.date_pattern, columns)
            logging.info("Read %d rows from parquet files.", len(df))

            df = apply_transformations(df)
            logging.info("Applied transformations.")

            # Load original lookup CSV files
            stem_groups = lookup_utils.load_lookup_csv("STEM Groups in the BLS Projections.csv")
            job_zones = lookup_utils.load_lookup_csv("ONET_Job_Zones.csv")
            # Optionally load SOC codes if needed
            # soc_codes = lookup_utils.load_lookup_csv("SOC_Codes.csv")

            # Normalize and merge STEM groups
            stem_groups = stem_groups.rename(columns={'SOC Code': 'soc2018_from_onet'})
            df['onet_raw'] = df['classifications_onet_code_25'].fillna(df['classifications_onet_code'])
            df['onet_norm'] = df['onet_raw'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()
            df['soc2018_from_onet'] = df['onet_norm'].astype(str).str.split('.').str[0]

            df = lookup_utils.merge_lookup_pandas(df, stem_groups, on='soc2018_from_onet')

            # Normalize and merge ONET job zones
            job_zones = job_zones.rename(columns={'Code': 'onet_norm'})
            df = lookup_utils.merge_lookup_pandas(df, job_zones, on='onet_norm')

            logging.info("Merged original lookup CSV files.")

            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"cleaned_data.{args.format}")
            if args.format == "parquet":
                df.to_parquet(output_file, index=False)
            else:
                df.to_feather(output_file)
            logging.info("Saved cleaned data to %s", output_file)

        except Exception as e:
            logging.error("Error during processing: %s", e, exc_info=True)
            sys.exit(1)
    else:
        # No CLI args: automatic directory scanning and processing
        input_dir = "data/raw"
        output_dir = "data/clean"
        output_format = "parquet"
        date_pattern = None

        logging.info("No CLI arguments provided, running automatic directory scan.")
        files = find_parquet_files(input_dir, date_pattern)
        logging.info("Found %d parquet files to process.", len(files))
        for file_path in files:
            process_file(file_path, output_dir, output_format)


if __name__ == "__main__":
    main()