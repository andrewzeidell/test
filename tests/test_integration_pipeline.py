import pytest
import pandas as pd
from src.reader import etl
from src.reader.transforms import lookup_utils as lu
from src.reader.transforms import transformations as tf

def test_full_pipeline(mock_parquet_dir, mock_lookup_csv):
    # Load parquet data
    df = etl.load_parquet_data(str(mock_parquet_dir))
    assert not df.empty

    # Load lookup tables
    stem_groups = lu.load_lookup_csv("STEM Groups in the BLS Projections.csv", base_dir=str(mock_lookup_csv))
    job_zones = lu.load_lookup_csv("ONET_Job_Zones.csv", base_dir=str(mock_lookup_csv))
    # Optionally load SOC codes if needed
    # soc_codes = lu.load_lookup_csv("SOC_Codes.csv", base_dir=str(mock_lookup_csv))

    # Normalize O*NET codes in main df
    df["onet_raw"] = df["classifications_onet_code_25"].fillna(df["classifications_onet_code"])
    df["onet_norm"] = df["onet_raw"].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()
    df["soc2018_from_onet"] = df["onet_norm"].astype(str).str.split('.').str[0]

    # Merge STEM groups
    stem_groups = stem_groups.rename(columns={'SOC Code': 'soc2018_from_onet'})
    df = lu.merge_lookup_pandas(df, stem_groups, on="soc2018_from_onet")

    # Merge ONET job zones
    job_zones = job_zones.rename(columns={'Code': 'onet_norm'})
    df = lu.merge_lookup_pandas(df, job_zones, on="onet_norm")

    # Check merged columns
    assert "STEM Group" in df.columns
    assert "Job Zone" in df.columns

    # Calculate posting age days for each row
    df["posting_age_days"] = df.apply(
        lambda row: tf.calculate_posting_age_days(row["date_acquired"], row["expired_date"]), axis=1
    )

    # Check posting age days column
    assert df["posting_age_days"].notnull().all()

    # Check ghost job flag
    df["is_ghost"] = df["ghostjob"].apply(tf.is_ghost_job)
    assert df["is_ghost"].dtype == bool

    # Check remote job flag (simulate remote flag column)
    df["remote_flag"] = ["yes", "no", "yes"]
    df["is_remote"] = df["remote_flag"].apply(tf.is_remote_job)
    assert df["is_remote"].dtype == bool

    # Clean education field
    df["cleaned_education"] = df["requirements_min_education"].apply(tf.clean_education_field)
    assert df["cleaned_education"].notnull().all()