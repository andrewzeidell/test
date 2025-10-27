from typing import List, Optional
import os
import re
import pandas as pd
import duckdb

# Minimal columns needed for the 10 analytical goals
MINIMAL_COLUMNS = [
    "job_id",
    "date_compiled",
    "date_acquired",
    "expired",
    "expired_date",
    "state",
    "city",
    "zipcode",
    "classifications_onet_code",
    "classifications_onet_code_25",
    "title",
    "description",
    "ghostjob",
    "jobclass",
    "parameters_salary_min",
    "parameters_salary_max",
    "parameters_salary_unit",
    "requirements_min_education",
    "requirements_experience",
    "requirements_license",
    "application_company",
]

def list_parquet_files(
    directory: str, date_pattern: Optional[str] = None
) -> List[str]:
    """
    List parquet files in a directory optionally filtered by a date pattern in the filename.

    Args:
        directory (str): Path to the directory containing parquet files.
        date_pattern (Optional[str]): Regex pattern to filter files by date in filename.

    Returns:
        List[str]: List of parquet file paths matching the criteria.
    """
    files = []
    pattern = re.compile(date_pattern) if date_pattern else None
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            if pattern:
                if pattern.search(filename):
                    files.append(os.path.join(directory, filename))
            else:
                files.append(os.path.join(directory, filename))
    return sorted(files)


def read_parquet_files(
    file_paths: List[str], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read parquet files using DuckDB, selecting only specified columns.

    Args:
        file_paths (List[str]): List of parquet file paths to read.
        columns (Optional[List[str]]): List of columns to select. If None, select all.

    Returns:
        pd.DataFrame: Combined DataFrame with selected columns.
    """
    if not file_paths:
        return pd.DataFrame()

    selected_columns = columns if columns is not None else "*"
    query = f"""
        SELECT {', '.join(selected_columns) if selected_columns != '*' else '*'}
        FROM parquet_scan('{file_paths[0]}')
    """
    # Read first file
    df = duckdb.query(query).to_df()

    # Append remaining files
    for file_path in file_paths[1:]:
        query = f"""
            SELECT {', '.join(selected_columns) if selected_columns != '*' else '*'}
            FROM parquet_scan('{file_path}')
        """
        df_next = duckdb.query(query).to_df()
        df = pd.concat([df, df_next], ignore_index=True)

    return df


def load_parquet_data(
    directory: str, date_pattern: Optional[str] = None, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load parquet files from a directory filtered by an optional date pattern,
    selecting only specified columns, and return a combined DataFrame.

    Args:
        directory (str): Directory containing parquet files.
        date_pattern (Optional[str]): Regex pattern to filter files by date in filename.
        columns (Optional[List[str]]): List of columns to select. If None, use minimal columns.

    Returns:
        pd.DataFrame: Combined DataFrame with selected columns.
    """
    if columns is None:
        columns = MINIMAL_COLUMNS

    file_paths = list_parquet_files(directory, date_pattern)
    if not file_paths:
        return pd.DataFrame()

    df = read_parquet_files(file_paths, columns)
    return df