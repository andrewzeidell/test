import pytest
import pandas as pd
from src.reader.transforms import lookup_utils as lu

def test_load_lookup_csv_success(tmp_path, mock_lookup_csv):
    # Use the mock lookup CSV directory fixture
    stem_groups_file = mock_lookup_csv / "STEM Groups in the BLS Projections.csv"
    df = lu.load_lookup_csv("STEM Groups in the BLS Projections.csv", base_dir=str(mock_lookup_csv))
    assert not df.empty
    assert "SOC Code" in df.columns

def test_load_lookup_csv_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        lu.load_lookup_csv("nonexistent.csv", base_dir=str(tmp_path))

def test_merge_lookup_pandas():
    main_df = pd.DataFrame({
        "onet_code": ["15-1121", "15-1132"],
        "value": [1, 2]
    })
    lookup_df = pd.DataFrame({
        "onet_code": ["15-1121", "15-1132"],
        "occupation": ["Software Developer", "Analyst"]
    })
    merged = lu.merge_lookup_pandas(main_df, lookup_df, on="onet_code")
    assert "occupation" in merged.columns
    assert merged.shape[0] == main_df.shape[0]

def test_merge_lookup_duckdb():
    main_df = pd.DataFrame({
        "onet_code": ["15-1121", "15-1132"],
        "value": [1, 2]
    })
    lookup_df = pd.DataFrame({
        "onet_code": ["15-1121", "15-1132"],
        "occupation": ["Software Developer", "Analyst"]
    })
    merged = lu.merge_lookup_duckdb(main_df, lookup_df, on="onet_code")
    assert "occupation" in merged.columns
    assert merged.shape[0] == main_df.shape[0]