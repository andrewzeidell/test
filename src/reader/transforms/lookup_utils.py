import pandas as pd
from typing import Optional


def add_fips_code_from_csv(
    df: pd.DataFrame,
    csv_path: str,
    state_abbr_col: str,
    fips_code_col: str = "fips_code",
    csv_state_abbr_col: str = "state_abbr",
    csv_fips_code_col: str = "fips_code",
) -> pd.DataFrame:
    """
    Add a FIPS code column to a DataFrame by merging with a CSV file containing state abbreviation to FIPS code mappings.

    This function reads the CSV file at `csv_path` which should contain at least two columns:
    one for state abbreviations and one for corresponding FIPS codes. It merges this mapping
    with the input DataFrame `df` on the specified state abbreviation columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column with state abbreviations.
    csv_path : str
        Path to the CSV file containing state abbreviation to FIPS code mappings.
    state_abbr_col : str
        Name of the column in `df` containing state abbreviations.
    fips_code_col : str, optional
        Name of the new column to add to `df` for FIPS codes, by default "fips_code".
    csv_state_abbr_col : str, optional
        Name of the state abbreviation column in the CSV file, by default "state_abbr".
    csv_fips_code_col : str, optional
        Name of the FIPS code column in the CSV file, by default "fips_code".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the FIPS code column added. Rows with unmatched or missing state abbreviations
        will have NaN in the FIPS code column.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"state": ["CA", "NY", "TX", "ZZ"]})
    >>> df_with_fips = add_fips_code_from_csv(df, "state_fips.csv", "state")
    >>> print(df_with_fips)
      state fips_code
    0    CA       06
    1    NY       36
    2    TX       48
    3    ZZ      NaN
    """
    # Read the CSV mapping file
    mapping_df = pd.read_csv(csv_path, dtype={csv_state_abbr_col: str, csv_fips_code_col: str})

    # Merge the input DataFrame with the mapping DataFrame on state abbreviation columns
    merged_df = df.merge(
        mapping_df[[csv_state_abbr_col, csv_fips_code_col]],
        how="left",
        left_on=state_abbr_col,
        right_on=csv_state_abbr_col,
    )

    # Rename the FIPS code column to the desired output column name
    merged_df = merged_df.rename(columns={csv_fips_code_col: fips_code_col})

    # Drop the extra state abbreviation column from the CSV mapping
    merged_df = merged_df.drop(columns=[csv_state_abbr_col])

    return merged_df