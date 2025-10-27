import os
import pandas as pd
from typing import Dict, Tuple

def load_lookups(lookups_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load and preprocess the three lookup CSV files used for enrichment and analysis.

    Args:
        lookups_dir (str): Path to the directory containing lookup CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys:
            - 'stem_groups': STEM groups DataFrame
            - 'job_zones': ONET job zones DataFrame
            - 'soc_codes': SOC codes DataFrame (if applicable)
    """
    stem_groups_path = os.path.join(lookups_dir, "STEM Groups in the BLS Projections.csv")
    job_zones_path = os.path.join(lookups_dir, "ONET_Job_Zones.csv")
    # Assuming SOC codes are embedded in STEM groups or separate CSV if needed
    # soc_codes_path = os.path.join(lookups_dir, "SOC_Codes.csv")  # Optional

    lookups = {}

    # Load STEM groups
    if os.path.exists(stem_groups_path):
        stem_groups = pd.read_csv(stem_groups_path, dtype=str)
        stem_groups.columns = stem_groups.columns.str.strip()
        # Normalize SOC Code column
        if 'SOC Code' in stem_groups.columns:
            stem_groups['SOC Code'] = stem_groups['SOC Code'].str.replace(r'[^0-9\-]', '', regex=True)
        lookups['stem_groups'] = stem_groups
    else:
        raise FileNotFoundError(f"STEM groups CSV not found at {stem_groups_path}")

    # Load ONET job zones
    if os.path.exists(job_zones_path):
        job_zones = pd.read_csv(job_zones_path, dtype=str)
        job_zones.columns = job_zones.columns.str.strip()
        lookups['job_zones'] = job_zones
    else:
        raise FileNotFoundError(f"ONET job zones CSV not found at {job_zones_path}")

    # Load SOC codes if applicable
    # if os.path.exists(soc_codes_path):
    #     soc_codes = pd.read_csv(soc_codes_path, dtype=str)
    #     soc_codes.columns = soc_codes.columns.str.strip()
    #     lookups['soc_codes'] = soc_codes

    return lookups


def enrich_data(df: pd.DataFrame, lookups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize codes, merge lookup data, and add derived columns for analysis.

    Args:
        df (pd.DataFrame): Raw job postings DataFrame.
        lookups (Dict[str, pd.DataFrame]): Dictionary of lookup DataFrames.

    Returns:
        pd.DataFrame: Enriched DataFrame ready for analysis.
    """
    # Normalize O*NET codes
    df['onet_raw'] = df['classifications_onet_code_25'].fillna(df['classifications_onet_code'])
    df['onet_norm'] = df['onet_raw'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()

    # Extract SOC code from O*NET
    df['soc2018_from_onet'] = df['onet_norm'].astype(str).str.split('.').str[0]

    # Merge STEM groups
    stem_groups = lookups.get('stem_groups')
    if stem_groups is not None:
        stem_groups_renamed = stem_groups.rename(columns={'SOC Code': 'soc2018_from_onet'})
        df = df.merge(stem_groups_renamed, on='soc2018_from_onet', how='left')

    # Merge ONET job zones
    job_zones = lookups.get('job_zones')
    if job_zones is not None:
        job_zones_renamed = job_zones.rename(columns={'Code': 'onet_norm'})
        df = df.merge(job_zones_renamed, on='onet_norm', how='left')

    # Fill missing text fields
    df['Title'] = df['title'].fillna("")
    df['Description'] = df['description'].fillna("")
    df['City'] = df['city'].replace("", pd.NA)
    df['Zip'] = df['zipcode'].replace("", pd.NA)

    return df

def filter_stem_jobs(df: pd.DataFrame, stem_groups: pd.DataFrame | None) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only rows with SOC codes in the allowed STEM groups.

    Args:
        df (pd.DataFrame): DataFrame with job postings.
        stem_groups (pd.DataFrame | None): STEM groups DataFrame or None if not available.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only STEM job postings.
    """
    if stem_groups is None:
        print("Warning: STEM groups DataFrame is None. Skipping STEM filtering.")
        return df

    # Clean column names and normalize SOC Code column
    stem_groups = stem_groups.copy()
    stem_groups.columns = stem_groups.columns.str.strip()
    if 'SOC Code' in stem_groups.columns:
        stem_groups['SOC Code'] = stem_groups['SOC Code'].astype(str).str.replace(r'[^0-9-]', '', regex=True)

    # Extract allowed SOC codes based on STEM groups excluding non-stem
    keep_groups = [g for g in stem_groups['STEM Group'].unique() if g.lower() not in ["non-stem", "non stem"]]
    allow_soc = stem_groups[stem_groups['STEM Group'].isin(keep_groups)]['SOC Code'].unique()

    before_n = len(df)
    filtered_df = df[df['soc2018_from_onet'].isin(allow_soc)]
    after_n = len(filtered_df)
    print(f"\\nFiltered by SOC allowlist for STEM. Rows kept: {after_n} of {before_n}")

    return filtered_df

def calculate_posting_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate posting age and lag metrics for job postings.

    Args:
        df (pd.DataFrame): Enriched job postings DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added 'Post_Age' column representing
                      days between acquisition and expiration.
    """
    # Ensure date columns are datetime
    df['date_acquired'] = pd.to_datetime(df['date_acquired'], utc=True, errors='coerce')
    df['expired_date'] = pd.to_datetime(df['expired_date'], utc=True, errors='coerce')

    # Calculate posting age only for expired postings
    df['Post_Age'] = pd.NA
    expired_mask = df['expired'] == 1
    df.loc[expired_mask, 'Post_Age'] = (df.loc[expired_mask, 'expired_date'] - df.loc[expired_mask, 'date_acquired']).dt.days

    return df


def normalize_pay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pay fields to hourly rates where feasible.

    Args:
        df (pd.DataFrame): DataFrame with salary columns:
            - parameters_salary_min
            - parameters_salary_max
            - parameters_salary_unit

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - pay_min_hr
            - pay_max_hr
    """

def aggregate_posting_age_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate posting age trends over time by month.

    Args:
        df (pd.DataFrame): DataFrame with 'date_acquired' and 'Post_Age' columns.

    Returns:
        pd.DataFrame: DataFrame indexed by month with average and median posting age.
    """
    df = df.copy()
    df['date_acquired'] = pd.to_datetime(df['date_acquired'], utc=True, errors='coerce')
    df = df.dropna(subset=['date_acquired', 'Post_Age'])
    df['month'] = df['date_acquired'].dt.to_period('M')

    agg = df.groupby('month')['Post_Age'].agg(['mean', 'median', 'count']).reset_index()
    agg['month'] = agg['month'].dt.to_timestamp()
    agg = agg.rename(columns={'mean': 'avg_posting_age', 'median': 'median_posting_age', 'count': 'postings_count'})
    return agg

def detect_hard_to_fill(df: pd.DataFrame, age_threshold: int = 30, min_postings: int = 10) -> pd.DataFrame:
    """
    Detect hard-to-fill job postings based on posting age and expiration.

    Args:
        df (pd.DataFrame): DataFrame with 'Post_Age' and 'expired' columns.
        age_threshold (int): Minimum posting age in days to consider hard-to-fill.
        min_postings (int): Minimum number of postings to consider for aggregation.

    Returns:
        pd.DataFrame: Aggregated DataFrame by 'onet_norm' with hard-to-fill signals.
    """
    df = df.copy()
    df = df.dropna(subset=['Post_Age', 'onet_norm'])

    # Filter postings that are expired and have posting age above threshold
    hard_to_fill_mask = (df['expired'] == 1) & (df['Post_Age'] >= age_threshold)
    hard_to_fill_df = df[hard_to_fill_mask]

    # Aggregate by occupation code
    agg = hard_to_fill_df.groupby('onet_norm').agg(
        hard_to_fill_count=pd.NamedAgg(column='Post_Age', aggfunc='count'),
        avg_posting_age=pd.NamedAgg(column='Post_Age', aggfunc='mean'),
        median_posting_age=pd.NamedAgg(column='Post_Age', aggfunc='median')
    ).reset_index()

    # Filter occupations with at least min_postings hard-to-fill postings
    agg = agg[agg['hard_to_fill_count'] >= min_postings]

    # Sort descending by hard_to_fill_count
    agg = agg.sort_values(by='hard_to_fill_count', ascending=False)

    return agg
    def normalize_to_hourly(row):
        unit = str(row['parameters_salary_unit']).upper()
        min_v, max_v = row['parameters_salary_min'], row['parameters_salary_max']

        conversions = {
            'HOURLY': 1, 'HOUR': 1, 'DAILY': 1 / 8, 'DAY': 1 / 8,
            'WEEKLY': 1 / 40, 'WEEK': 1 / 40, 'BIWEEKLY': 1 / 80,
            'MONTHLY': 1 / 173.3333, 'MONTH': 1 / 173.3333,
            'YEARLY': 1 / 2080, 'YEAR': 1 / 2080, 'ANNUAL': 1 / 2080
        }

        conv = conversions.get(unit, None)

        if conv:
            return pd.Series([min_v * conv if pd.notna(min_v) else pd.NA,
                              max_v * conv if pd.notna(max_v) else pd.NA])
        else:
            return pd.Series([pd.NA, pd.NA])

    if all(col in df.columns for col in ["parameters_salary_min", "parameters_salary_max", "parameters_salary_unit"]):
        pay_cols = df.apply(normalize_to_hourly, axis=1)
        pay_cols.columns = ['pay_min_hr', 'pay_max_hr']
        df = pd.concat([df, pay_cols], axis=1)
    else:
        df['pay_min_hr'] = pd.NA
        df['pay_max_hr'] = pd.NA

    return df


def filter_ghost_jobs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate national and regional ghost jobs.

    Args:
        df (pd.DataFrame): DataFrame with 'ghostjob' boolean column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (national_jobs_df, regional_jobs_df)
    """
    national_jobs_df = df[df['ghostjob'] != True].copy()
    regional_jobs_df = df.copy()
    return national_jobs_df, regional_jobs_df


def aggregate_geography(df: pd.DataFrame) -> dict:
    """
    Aggregate job postings by state, city, and zip with STEM group and job zone breakdowns.

    Args:
        df (pd.DataFrame): DataFrame with job postings.

    Returns:
        dict: Dictionary with keys 'by_state', 'by_city', 'by_zip' containing aggregated DataFrames.
    """
    aggregates = {}

    # State level aggregation
    by_state_seen = df.groupby('state').size().reset_index(name='postings_seen').sort_values('postings_seen', ascending=False)

    by_state_STEMgroups = df.pivot_table(
        index='state',
        columns='STEM Group',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    by_state_jobzones = df.pivot_table(
        index='state',
        columns='Job Zone',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    aggregates['by_state'] = {
        'postings_seen': by_state_seen,
        'STEM_groups': by_state_STEMgroups,
        'job_zones': by_state_jobzones
    }

    # City level aggregation
    by_city_seen = df.dropna(subset=['City']).groupby(['state', 'City']).size().reset_index(name='postings_seen').sort_values('postings_seen', ascending=False)

    by_city_STEMgroups = df.pivot_table(
        index=['state', 'City'],
        columns='STEM Group',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    by_city_jobzones = df.pivot_table(
        index=['state', 'City'],
        columns='Job Zone',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    aggregates['by_city'] = {
        'postings_seen': by_city_seen,
        'STEM_groups': by_city_STEMgroups,
        'job_zones': by_city_jobzones
    }

    # Zip level aggregation
    by_zip_seen = df.dropna(subset=['Zip']).groupby(['state', 'Zip']).size().reset_index(name='postings_seen').sort_values('postings_seen', ascending=False)

    by_zip_STEMgroups = df.pivot_table(
        index=['state', 'Zip'],
        columns='STEM Group',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    by_zip_jobzones = df.pivot_table(
        index=['state', 'Zip'],
        columns='Job Zone',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    aggregates['by_zip'] = {
        'postings_seen': by_zip_seen,
        'STEM_groups': by_zip_STEMgroups,
        'job_zones': by_zip_jobzones
    }

    return aggregates


def aggregate_occupation(df: pd.DataFrame) -> dict:
    """
    Aggregate job postings by O*NET codes and job zones.

    Args:
        df (pd.DataFrame): DataFrame with job postings.

    Returns:
        dict: Dictionary with keys 'by_onet', 'job_zone_stats' containing aggregated DataFrames.
    """
    aggregates = {}

    by_onet = df.dropna(subset=['onet_norm']).groupby('onet_norm').size().reset_index(name='postings')

    by_job_zone = df.pivot_table(
        index='onet_norm',
        columns='Job Zone',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    aggregates['by_onet'] = by_onet
    aggregates['job_zone_stats'] = by_job_zone

    return aggregates


def analyze_credentials(df: pd.DataFrame) -> dict:
    """
    Count education, experience, and license requirements.

    Args:
        df (pd.DataFrame): DataFrame with job postings.

    Returns:
        dict: Dictionary with keys 'education_counts', 'experience_counts', 'license_counts' containing count DataFrames.
    """
    aggregates = {}

    if "requirements_min_education" in df.columns:
        df['req_min_edu'] = df['requirements_min_education'].replace("", "Unspecified").fillna("Unspecified")
        education_counts = df['req_min_edu'].value_counts().reset_index(name='postings').sort_values('postings', ascending=False)
        education_counts.columns = ['req_min_edu', 'postings']
        aggregates['education_counts'] = education_counts
    else:
        aggregates['education_counts'] = pd.DataFrame()

    if "requirements_experience" in df.columns:
        df['req_exp'] = df['requirements_experience'].replace("", "Unspecified").fillna("Unspecified")
        experience_counts = df['req_exp'].value_counts().reset_index(name='postings').sort_values('postings', ascending=False)
        experience_counts.columns = ['req_exp', 'postings']
        aggregates['experience_counts'] = experience_counts
    else:
        aggregates['experience_counts'] = pd.DataFrame()

    if "requirements_license" in df.columns:
        df['req_lic'] = df['requirements_license'].replace("", "Unspecified").fillna("Unspecified")
        license_counts = df['req_lic'].value_counts().reset_index(name='postings').sort_values('postings', ascending=False)
        license_counts.columns = ['req_lic', 'postings']
        aggregates['license_counts'] = license_counts
    else:
        aggregates['license_counts'] = pd.DataFrame()

    return aggregates


def extract_remote_onsite_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract remote vs onsite flags from job postings.

    Args:
        df (pd.DataFrame): DataFrame with job postings.

    Returns:
        pd.DataFrame: DataFrame with added boolean columns 'is_remote' and 'is_onsite'.
    """
    # Define keywords for remote and onsite detection
    remote_keywords = ['remote', 'work from home', 'telecommute', 'telework', 'virtual']
    onsite_keywords = ['onsite', 'on-site', 'in person', 'in-office', 'office']

    def detect_flag(text: str, keywords: list) -> bool:
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    df['is_remote'] = df['jobclass'].apply(lambda x: detect_flag(x, remote_keywords)) | \
                      df['Description'].apply(lambda x: detect_flag(x, remote_keywords))

    df['is_onsite'] = df['jobclass'].apply(lambda x: detect_flag(x, onsite_keywords)) | \
                      df['Description'].apply(lambda x: detect_flag(x, onsite_keywords))

    return df


def compute_top_n(df: pd.DataFrame, n: int) -> dict:
    """
    Compute top-N aggregations by key attributes.

    Args:
        df (pd.DataFrame): DataFrame with job postings.
        n (int): Number of top entries to return.

    Returns:
        dict: Dictionary with keys for top states, occupations, companies.
    """
    top_states = df['state'].value_counts().head(n).reset_index()
    top_states.columns = ['state', 'postings']

    top_onet = df['onet_norm'].value_counts().head(n).reset_index()
    top_onet.columns = ['onet_norm', 'postings']

    top_companies = df['application_company'].value_counts().head(n).reset_index()
    top_companies.columns = ['application_company', 'postings']

    return {
        'top_states': top_states,
        'top_onet': top_onet,
        'top_companies': top_companies
    }


def save_outputs(aggregates: dict, output_dir: str) -> None:
    """
    Save aggregated data to CSV and Excel files.

    Args:
        aggregates (dict): Dictionary of aggregated DataFrames.
        output_dir (str): Directory path to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save geography aggregates
    geo = aggregates.get('geography', {})
    for level, dfs in geo.items():
        for name, df in dfs.items():
            csv_path = os.path.join(output_dir, f"{level}_{name}.csv")
            df.to_csv(csv_path, index=False)

    # Save occupation aggregates
    occ = aggregates.get('occupation', {})
    for name, df in occ.items():
        csv_path = os.path.join(output_dir, f"occupation_{name}.csv")
        df.to_csv(csv_path, index=False)

    # Save credentials aggregates
    creds = aggregates.get('credentials', {})
    for name, df in creds.items():
        csv_path = os.path.join(output_dir, f"credentials_{name}.csv")
        df.to_csv(csv_path, index=False)

    # Save top-N aggregates
    topn = aggregates.get('top_n', {})
    for name, df in topn.items():
        csv_path = os.path.join(output_dir, f"topn_{name}.csv")
        df.to_csv(csv_path, index=False)