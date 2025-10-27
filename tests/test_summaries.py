import pandas as pd
import pytest
from src.reader import summaries
import os
import tempfile

def test_aggregate_posting_age_trends():
    data = {
        'date_acquired': ['2023-01-15', '2023-01-20', '2023-02-10', '2023-02-15'],
        'Post_Age': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    result = summaries.aggregate_posting_age_trends(df)
    assert not result.empty
    assert 'avg_posting_age' in result.columns
    assert 'median_posting_age' in result.columns
    assert 'postings_count' in result.columns
    jan_data = result[result['month'] == pd.Timestamp('2023-01-01')]
    feb_data = result[result['month'] == pd.Timestamp('2023-02-01')]
    assert jan_data['avg_posting_age'].values[0] == 15
    assert feb_data['median_posting_age'].values[0] == 35

def test_detect_hard_to_fill():
    data = {
        'Post_Age': [10, 35, 40, 5, 50],
        'expired': [1, 1, 1, 0, 1],
        'onet_norm': ['111', '111', '222', '222', '111']
    }
    df = pd.DataFrame(data)
    result = summaries.detect_hard_to_fill(df, age_threshold=30, min_postings=2)
    assert not result.empty
    assert 'hard_to_fill_count' in result.columns
    assert 'avg_posting_age' in result.columns
    assert 'median_posting_age' in result.columns
    # Only '111' should appear because it has 3 expired postings with age >= 30
    assert '111' in result['onet_norm'].values
    assert '222' not in result['onet_norm'].values

def test_save_aggregates_with_posting_and_htf(tmp_path):
    # Prepare dummy aggregates
    aggregates = {
        'geography': {
            'by_state': {
                'postings_seen': pd.DataFrame({'state': ['CA'], 'postings_seen': [100]}),
            }
        },
        'occupation': {},
        'credentials': {},
        'top_n': {}
    }
    posting_age_agg = pd.DataFrame({
        'month': [pd.Timestamp('2023-01-01')],
        'avg_posting_age': [15],
        'median_posting_age': [10],
        'postings_count': [5]
    })
    hard_to_fill_agg = pd.DataFrame({
        'onet_norm': ['111'],
        'hard_to_fill_count': [3],
        'avg_posting_age': [35],
        'median_posting_age': [30]
    })

    output_dir = tmp_path

    # Call the save function
    summaries.save_aggregates_with_posting_and_htf(
        aggregates,
        posting_age_agg,
        hard_to_fill_agg,
        str(output_dir)
    )

    # Check files exist
    posting_age_file = output_dir / "posting_age_trends.csv"
    htf_file = output_dir / "hard_to_fill_signals.csv"
    geo_file = output_dir / "by_state_postings_seen.csv"

    assert posting_age_file.exists()
    assert htf_file.exists()
    assert geo_file.exists()

    # Check content of posting age file
    df_posting_age = pd.read_csv(posting_age_file)
    assert 'avg_posting_age' in df_posting_age.columns

    # Check content of hard-to-fill file
    df_htf = pd.read_csv(htf_file)
    assert 'hard_to_fill_count' in df_htf.columns

if __name__ == "__main__":
    pytest.main([__file__])