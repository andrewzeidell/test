"""
Placeholder for lookup merge utilities.

This module will provide helper functions to merge lookup tables
(e.g., O*NET and STEM lookups) with the main dataset.
"""

import os
from typing import Optional, Union, Tuple
import pandas as pd
import duckdb

LOOKUP_BASE_DIR = os.path.join("data", "lookups")

def load_lookup_csv(filename: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV lookup table from the configured lookups directory.

    Args:
        filename (str): The CSV filename to load.
        base_dir (Optional[str]): Base directory for lookups. Defaults to data/lookups/.

    Returns:
        pd.DataFrame: Loaded lookup table as a DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    directory = base_dir if base_dir is not None else LOOKUP_BASE_DIR
    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Lookup CSV file not found: {filepath}")
    return pd.read_csv(filepath)

def merge_lookup_pandas(
    main_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    on: Union[str, list[str]],
    how: str = "left",
    suffixes: Tuple[str, str] = ("", "_lookup"),
) -> pd.DataFrame:
    """
    Merge a lookup DataFrame into the main DataFrame using Pandas merge.

    Args:
        main_df (pd.DataFrame): The main job postings DataFrame.
        lookup_df (pd.DataFrame): The lookup DataFrame to merge.
        on (Union[str, list[str]]): Column name(s) to join on.
        how (str): Type of merge to perform. Defaults to 'left'.
        suffixes (Tuple[str, str]): Suffixes to apply to overlapping columns.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    return main_df.merge(lookup_df, on=on, how=how, suffixes=suffixes)

def merge_lookup_duckdb(
    main_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    on: Union[str, list[str]],
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge a lookup DataFrame into the main DataFrame using DuckDB SQL JOIN.

    Args:
        main_df (pd.DataFrame): The main job postings DataFrame.
        lookup_df (pd.DataFrame): The lookup DataFrame to merge.
        on (Union[str, list[str]]): Column name(s) to join on.
        how (str): Type of join to perform. Defaults to 'left'.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    con = duckdb.connect()
    con.register("main_df", main_df)
    con.register("lookup_df", lookup_df)

    if isinstance(on, str):
        on = [on]

    join_condition = " AND ".join(
        [f"main_df.{col} = lookup_df.{col}" for col in on]
    )

    query = f"""
    SELECT main_df.*, lookup_df.*
    FROM main_df
    {how.upper()} JOIN lookup_df
    ON {join_condition}
    """

    result_df = con.execute(query).df()

    # Drop duplicate join columns from lookup_df to avoid redundancy
    for col in on:
        if col in result_df.columns and f"{col}_1" in result_df.columns:
            result_df.drop(columns=[f"{col}_1"], inplace=True)

    con.unregister("main_df")
    con.unregister("lookup_df")
    con.close()

    return result_df