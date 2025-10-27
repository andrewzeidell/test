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
    onet_lookup = lu.load_lookup_csv("onet_lookup.csv", base_dir=str(mock_lookup_csv))
    stem_lookup = lu.load_lookup_csv("stem_lookup.csv", base_dir=str(mock_lookup_csv))

    # Normalize O*NET codes in main df
    df["normalized_onet"] = df["classifications_onet_code"].apply(tf.normalize_onet_code)

    # Merge O*NET lookup using pandas merge
    df = lu.merge_lookup_pandas(df, onet_lookup, on="onet_code")

    # Check merged columns
    assert "occupation" in df.columns

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