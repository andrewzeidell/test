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


def read_single_parquet_file(
    file_path: str, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read a single parquet file using DuckDB, selecting only specified columns.

    Args:
        file_path (str): Parquet file path to read.
        columns (Optional[List[str]]): List of columns to select. If None, select all.

    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """
    selected_columns = columns if columns is not None else "*"
    query = f"""
        SELECT {', '.join(selected_columns) if selected_columns != '*' else '*'}
        FROM parquet_scan('{file_path}')
    """
    df = duckdb.query(query).to_df()
    return df


def read_parquet_files(
    file_paths: List[str], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read multiple parquet files using DuckDB, selecting only specified columns,
    and concatenate them into a single DataFrame.

    Args:
        file_paths (List[str]): List of parquet file paths to read.
        columns (Optional[List[str]]): List of columns to select. If None, select all.

    Returns:
        pd.DataFrame: Combined DataFrame with selected columns.
    """
    if not file_paths:
        return pd.DataFrame()

    dfs = []
    for file_path in file_paths:
        df = read_single_parquet_file(file_path, columns)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined_df


from src.reader.transforms.transformations import clean_education_column

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

    # For backward compatibility, still read all files combined
    df = read_parquet_files(file_paths, columns)

    # Clean education field column if present
    if 'requirements_min_education' in df.columns:
        df['requirements_min_education'] = clean_education_column(df['requirements_min_education'])

    return df


def load_stem_lookup(stem_csv_path: str) -> pd.DataFrame:
    """
    Load the STEM groups lookup CSV and clean SOC codes.

    Args:
        stem_csv_path (str): Path to the STEM groups CSV file.

    Returns:
        pd.DataFrame: DataFrame with cleaned SOC codes and STEM group info.
    """
    stem_df = pd.read_csv(stem_csv_path, dtype=str)
    stem_df.columns = stem_df.columns.str.strip()
    stem_df['SOC Code'] = stem_df['SOC Code'].astype(str).str.replace(r'[^0-9-]', '', regex=True)
    return stem_df


def filter_stem_postings(df: pd.DataFrame, stem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the job postings DataFrame to retain only STEM group postings based on SOC codes.

    Args:
        df (pd.DataFrame): Job postings DataFrame with 'classifications_onet_code_25' or 'classifications_onet_code'.
        stem_df (pd.DataFrame): STEM groups DataFrame with cleaned SOC codes.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only STEM postings.
    """
    # Normalize ONET code to extract SOC 2018 code prefix
    df = df.copy()
    df['onet_raw'] = df['classifications_onet_code_25'].fillna(df['classifications_onet_code'])
    df['onet_norm'] = df['onet_raw'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()
    df['soc2018_from_onet'] = df['onet_norm'].astype(str).str.split('.').str[0]

    # Determine allowed SOC codes for STEM groups (exclude non-STEM)
    keep_groups = [g for g in stem_df['STEM Group'].unique() if g.lower() not in ['non-stem', 'non stem']]
    allowed_soc = stem_df[stem_df['STEM Group'].isin(keep_groups)]['SOC Code'].unique()

    # Filter dataframe by allowed SOC codes
    filtered_df = df[df['soc2018_from_onet'].isin(allowed_soc)]

    return filtered_df


def load_parquet_data_with_stem_filter(
    directory: str,
    stem_csv_path: str,
    date_pattern: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load parquet files from a directory, apply STEM filtering based on lookup CSV,
    and return a filtered DataFrame.

    Args:
        directory (str): Directory containing parquet files.
        stem_csv_path (str): Path to the STEM groups CSV file.
        date_pattern (Optional[str]): Regex pattern to filter files by date in filename.
        columns (Optional[List[str]]): List of columns to select. If None, use minimal columns.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only STEM postings.
    """
    df = load_parquet_data(directory, date_pattern, columns)
    if df.empty:
        return df

    stem_df = load_stem_lookup(stem_csv_path)
    filtered_df = filter_stem_postings(df, stem_df)
    return filtered_df