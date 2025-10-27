import pandas as pd
import pytest
from src.reader import summaries

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

if __name__ == "__main__":
    pytest.main([__file__])