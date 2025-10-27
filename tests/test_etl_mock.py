import pytest
import pandas as pd
from src.reader import etl

def test_list_parquet_files(mock_parquet_dir):
    files = etl.list_parquet_files(str(mock_parquet_dir))
    assert len(files) == 2
    assert all(f.endswith(".parquet") for f in files)

def test_list_parquet_files_with_date_pattern(mock_parquet_dir):
    # Only files with '2022-01' in name
    files = etl.list_parquet_files(str(mock_parquet_dir), date_pattern=r"2022-01")
    assert len(files) == 1
    assert "2022-01" in files[0]

def test_read_parquet_files_select_columns(mock_parquet_dir):
    files = etl.list_parquet_files(str(mock_parquet_dir))
    columns = ["job_id", "state", "title"]
    df = etl.read_parquet_files(files, columns=columns)
    assert not df.empty
    assert set(df.columns) == set(columns)
    assert df.shape[0] == 3  # 2 rows in first file + 1 in second

def test_load_parquet_data_default_columns(mock_parquet_dir):
    df = etl.load_parquet_data(str(mock_parquet_dir))
    # Should contain minimal columns defined in etl.MINIMAL_COLUMNS
    for col in etl.MINIMAL_COLUMNS:
        assert col in df.columns
    assert df.shape[0] == 3

def test_load_parquet_data_no_files(tmp_path):
    # Empty directory should return empty DataFrame
    df = etl.load_parquet_data(str(tmp_path))
    assert df.empty