import pytest
import pandas as pd
import os
import tempfile

@pytest.fixture
def mock_parquet_dir(tmp_path):
    """
    Create a temporary directory with mock parquet files for testing.
    Returns the directory path.
    """
    # Create sample dataframes
    df1 = pd.DataFrame({
        "job_id": [1, 2],
        "date_compiled": ["2022-01-01", "2022-01-02"],
        "date_acquired": ["2022-01-01", "2022-01-02"],
        "expired": [False, True],
        "expired_date": ["2022-01-10", "2022-01-12"],
        "state": ["CA", "NY"],
        "city": ["Los Angeles", "New York"],
        "zipcode": ["90001", "10001"],
        "classifications_onet_code": ["15-1121", "15-1132"],
        "classifications_onet_code_25": ["15-1121.00", "15-1132.00"],
        "title": ["Software Engineer", "Data Scientist"],
        "description": ["Develop software", "Analyze data"],
        "ghostjob": [0, 1],
        "jobclass": ["A", "B"],
        "parameters_salary_min": [70000, 80000],
        "parameters_salary_max": [90000, 100000],
        "parameters_salary_unit": ["year", "year"],
        "requirements_min_education": ["Bachelor's", "Master's"],
        "requirements_experience": ["3 years", "5 years"],
        "requirements_license": ["None", "None"],
        "application_company": ["Company A", "Company B"],
    })

    df2 = pd.DataFrame({
        "job_id": [3],
        "date_compiled": ["2022-02-01"],
        "date_acquired": ["2022-02-01"],
        "expired": [False],
        "expired_date": ["2022-02-10"],
        "state": ["TX"],
        "city": ["Houston"],
        "zipcode": ["77001"],
        "classifications_onet_code": ["15-1141"],
        "classifications_onet_code_25": ["15-1141.00"],
        "title": ["Network Engineer"],
        "description": ["Manage networks"],
        "ghostjob": [0],
        "jobclass": ["C"],
        "parameters_salary_min": [60000],
        "parameters_salary_max": [75000],
        "parameters_salary_unit": ["year"],
        "requirements_min_education": ["Associate's"],
        "requirements_experience": ["2 years"],
        "requirements_license": ["None"],
        "application_company": ["Company C"],
    })

    # Write to parquet files
    file1 = tmp_path / "2022-01.parquet"
    file2 = tmp_path / "2022-02.parquet"
    df1.to_parquet(file1)
    df2.to_parquet(file2)

    return tmp_path

@pytest.fixture
def mock_lookup_csv(tmp_path):
    """
    Create a temporary directory with mock lookup CSV files.
    Returns the directory path.
    """
    lookup_dir = tmp_path / "lookups"
    lookup_dir.mkdir()

    stem_groups_df = pd.DataFrame({
        "SOC Code": ["15-1121", "15-1132", "15-1141"],
        "STEM Group": ["Science", "Technology", "Engineering"],
    })
    stem_groups_file = lookup_dir / "STEM Groups in the BLS Projections.csv"
    stem_groups_df.to_csv(stem_groups_file, index=False)

    job_zones_df = pd.DataFrame({
        "Code": ["15-1121", "15-1132", "15-1141"],
        "Job Zone": ["Zone 1", "Zone 2", "Zone 3"],
    })
    job_zones_file = lookup_dir / "ONET_Job_Zones.csv"
    job_zones_df.to_csv(job_zones_file, index=False)

    # Optionally create SOC_Codes.csv if needed
    soc_codes_df = pd.DataFrame({
        "SOC Code": ["15-1121", "15-1132", "15-1141"],
        "Description": ["Desc1", "Desc2", "Desc3"],
    })
    soc_codes_file = lookup_dir / "SOC_Codes.csv"
    soc_codes_df.to_csv(soc_codes_file, index=False)

    return lookup_dir