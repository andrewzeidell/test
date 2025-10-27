import os

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime, date
import re
import pyarrow
import pyarrow.dataset as ds
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# Suppress all warnings to clean up the output
warnings.filterwarnings("ignore")

# --- CONFIG + PATHS ---
# --- CONFIG + PATHS ---
wd = os.getcwd()
raw_data_folder = os.path.join(wd, "data", "job_postings")
# raw_data_folder = os.path.join(wd, "data")
clean_data_folder = os.path.join(wd, "clean_output")
os.makedirs(raw_data_folder, exist_ok=True)
os.makedirs(clean_data_folder, exist_ok=True)

dataset_base_path = None

# This is testing if it works for CSV outputs like the API gives
# csvtoread = os.path.join(raw_data_folder,'TEST.csv')
# df = pd.read_csv(csvtoread)

pathlists = [] # some code here to get all the paths
# DATE_RE = re.compile(r"_(?P<year>\d{4})_(?P<start>\d{2})(?P<end>\d{2})$")
DATE_RE = re.compile(r'(?<!\d)\d{4}-(0[1-9]|1[0-2])(?!\d)')

def collect_frames() -> pd.DataFrame:
    rows = []
    for entry in os.listdir(raw_data_folder):
    # for entry in os.listdir(raw_data_folder):
        # m = DATE_RE.match(entry)
        match = DATE_RE.search(entry)

        if match:
            rows.append(entry)
            # year = match.group("year")
            # start_mm = match.group("start")
            # end_month = match.group("end")

            # print(f"Year: {year}")
            # print(f"Start Month: {start_month}")
            # print(f"End Month: {end_month}")
        else:
            print("No match found.")

    #     if not match:
    #         continue
    #     # year, start_mm, end_mm = m.groups()
    #     # year = int(year)
    #     # start_mm = int(start_mm)
    #
    #     # csv_path = Path(base) / entry / csv_name
    #     # if not csv_path.exists():
    #     #     continue
    #
    #     # df = pd.read_csv(csv_path)
    #     # df = normalize_columns(df)
    #
    #     # Add time metadata
    #     # df["year"] = year
    #     # df["month"] = start_mm
    #     # # monthly timestamp using first of month (good index key)
    #     # df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
    #     #
    #     # rows.append(df)
    #
    # if not rows:
    #     raise FileNotFoundError("No matching dated folders/CSVs found. Check BASE_PATH/CSV_NAME and folder format YYYY_MMmm.")
    # combined = pd.concat(rows, ignore_index=True)
    # combined = combined.sort_values(["date"]).reset_index(drop=True)
    return rows

# pathlists = collect_frames()
pathlists = ['unredacted__job__2018-04.parquet']
# pathlists = ['job_data_parquet_2023_0405','job_data_parquet_2023_0506','job_data_parquet_2023_0607','job_data_parquet_2023_0708','job_data_parquet_2023_0809']
for paths in pathlists:

    # Use this for actual analysis
    dataset_base_path = paths
    output_dir = "analysis_outputs"
    # if paths == 'job_data_parquet_2023_0910': break
    if dataset_base_path: output_dir = "analysis_outputs/" + output_dir + "_" + dataset_base_path
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading parquet file {dataset_base_path}...")
    # df = pd.read_parquet(dataset_base_path, use_nullable_dtypes=pd.NA)

    # csvtoread = os.path.join(raw_data_folder, 'TEST.csv')
    # df = pd.read_csv(csvtoread)

    # The commented code here is for if we have parquet files
    #
    # Example monthly file (adjust as needed)
    parquet_file = os.path.join(raw_data_folder, dataset_base_path)
    parquet_file = os.path.normpath(parquet_file)
    #
    # -----------------------------
    # EXPANDED SELECT (Jobs table parquet)
    # -----------------------------
    try:
        con = duckdb.connect()
        query = f"""
        SELECT
          job_id,
          date_compiled,
          date_acquired,
          expired,
          expired_date,
          state,
          city,
          zipcode,
          classifications_onet_code,
          classifications_onet_code_25,
          title,
          description,
          ghostjob,
          jobclass,
          parameters_salary_min,
          parameters_salary_max,
          parameters_salary_unit,
          requirements_min_education,
          requirements_experience,
          requirements_license,
          application_company
        FROM read_parquet('{parquet_file}')
        """
        df = con.execute(query).fetchdf()
        con.close()
    except duckdb.OperationalError as e:
        print(f"Error executing DuckDB query: {e}")
        print("Creating an empty DataFrame for demonstration.")
        df = pd.DataFrame(columns=[
            "job_id", "date_compiled", "date_acquired", "expired", "expired_date",
            "state", "city", "zipcode", "classifications_onet_code",
            "classifications_onet_code_25", "title", "description",
            "ghostjob", "jobclass", "parameters_salary_min", "parameters_salary_max",
            "parameters_salary_unit", "requirements_min_education",
            "requirements_experience", "requirements_license", "application_company"
        ])
    print(f"Completed reading parquet file {dataset_base_path}.")
    print("DataFrame info after initial pull:")
    print(df.info())
    print("\nDataFrame head:")
    print(df.head())

    # Sanity
    required_cols = ["job_id", "state", "date_compiled", "title", "ghostjob", "jobclass"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame is missing required columns.")

    # Prefer O*NET v25, normalize code string
    df['onet_raw'] = df['classifications_onet_code_25'].fillna(df['classifications_onet_code'])
    df['onet_norm'] = df['onet_raw'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.strip()

    # ---------------------------------
    # SOC/O*NET allowlist (from CSVs)
    # ---------------------------------
    df['soc2018_from_onet'] = df['onet_norm'].astype(str).str.split('.').str[0]

    # Read STEM groups file
    stem_path = os.path.join(wd, "lookups", "STEM Groups in the BLS Projections.csv")

    # Read Education related SOCs from STEM groups file, before processing df for STEM only
    stem_path = os.path.join(wd, "lookups", "STEM Groups in the BLS Projections.csv")
    if os.path.exists(stem_path):
        stem_bls = pd.read_csv(stem_path, dtype='str')
        stem_bls.columns = stem_bls.columns.str.strip()  # Clean column names
        stem_bls['SOC Code'] = stem_bls['SOC Code'].astype(str).str.replace(r'[^0-9-]', '', regex=True)

        KEEP_GROUPS = [g for g in stem_bls['Education'].unique() if pd.notna(g)]
        allow_soc = stem_bls[stem_bls['Education'].isin(KEEP_GROUPS)]['SOC Code'].unique()

        before_n = len(df)
        df_educ = df[df['soc2018_from_onet'].isin(allow_soc)]
        after_n = len(df_educ)
        print(f"\nFiltered by SOC allowlist for educators. Rows kept: {after_n} of {before_n}")
    else:
        print(f"\nWarning: STEM Groups file not found at {stem_path}. Skipping SOC filter.")

    # Trim df down to STEM only
    if os.path.exists(stem_path):
        stem_bls = pd.read_csv(stem_path, dtype='str')
        stem_bls.columns = stem_bls.columns.str.strip()  # Clean column names
        stem_bls['SOC Code'] = stem_bls['SOC Code'].astype(str).str.replace(r'[^0-9-]', '', regex=True)

        KEEP_GROUPS = [g for g in stem_bls['STEM Group'].unique() if g.lower() not in ["non-stem", "non stem"]]
        allow_soc = stem_bls[stem_bls['STEM Group'].isin(KEEP_GROUPS)]['SOC Code'].unique()

        before_n = len(df)
        df = df[df['soc2018_from_onet'].isin(allow_soc)]
        after_n = len(df)
        print(f"\nFiltered by SOC allowlist for STEM. Rows kept: {after_n} of {before_n}")
    else:
        print(f"\nWarning: STEM Groups file not found at {stem_path}. Skipping SOC filter.")

    # Dates: ymd_open from date_acquired when present; fallback to date_compiled
    df['date_compiled'] = pd.to_datetime(df['date_compiled'], utc=True)
    df['date_acquired'] = pd.to_datetime(df['date_acquired'], utc=True)
    df['expired_date'] = pd.to_datetime(df['expired_date'], utc=True)
    df['date_open_approx'] = df['date_acquired'].fillna(df['date_compiled'])
    df['ymd_open'] = df['date_open_approx'].dt.date
    df['ymd_seen'] = df['date_compiled'].dt.date

    # Text hygiene + geo for future analysis
    df['Title'] = df['title'].fillna("")
    df['Description'] = df['description'].fillna("")
    df['City'] = df['city'].replace("", pd.NA)
    df['Zip'] = df['zipcode'].replace("", pd.NA)

    # ---------------------------------
    # Calculate the time from acquisition to expiration
    # ---------------------------------
    df_expired = df[df['expired'] == 1.0]
    df['Post_Age'] = (df_expired['expired_date']-df_expired['date_acquired']).dt.days

    # Dates: ymd_open from date_acquired when present; fallback to date_compiled
    df_educ['date_compiled'] = pd.to_datetime(df_educ['date_compiled'], utc=True)
    df_educ['date_acquired'] = pd.to_datetime(df_educ['date_acquired'], utc=True)
    df_educ['expired_date'] = pd.to_datetime(df_educ['expired_date'], utc=True)
    df_educ['date_open_approx'] = df_educ['date_acquired'].fillna(df_educ['date_compiled'])
    df_educ['ymd_open'] = df_educ['date_open_approx'].dt.date
    df_educ['ymd_seen'] = df_educ['date_compiled'].dt.date

    # Text hygiene + geo for future analysis
    df_educ['Title'] = df_educ['title'].fillna("")
    df_educ['Description'] = df_educ['description'].fillna("")
    df_educ['City'] = df_educ['city'].replace("", pd.NA)
    df_educ['Zip'] = df_educ['zipcode'].replace("", pd.NA)

    # ---------------------------------
    # Calculate the time from acquisition to expiration
    # ---------------------------------
    df_educ_expired = df_educ[df_educ['expired'] == 1.0]
    df_educ['Post_Age'] = (df_educ_expired['expired_date']-df_educ_expired['date_acquired']).dt.days



    # ---------------------------------
    # Additional analysis, splitting by different sub cateogires to get counts, and organize by zip, state, and also summarize data about education and etc.
    # ---------------------------------

    # Begin by merging the STEM, STEM related, etc on
    # Rename the STEM BLS column to match for merge
    stem_bls = stem_bls.rename(columns={'SOC Code':'soc2018_from_onet'})
    df = pd.merge(df,stem_bls,on='soc2018_from_onet',how='left')
    df_educ = pd.merge(df_educ,stem_bls,on='soc2018_from_onet',how='left')

    # Merge on the Job Zones
    job_zone_path = os.path.join(wd, "lookups", "ONET_Job_Zones.csv")
    job_zones = pd.read_csv(job_zone_path)
    # rename column for merge
    job_zones = job_zones.rename(columns={'Code':'onet_norm'})
    # job_zones['Code'] = job_zones['Code'].astype(str).str.split('.').str[0]
    df = pd.merge(df,job_zones,on='onet_norm',how='left')

    # ---------------------------------
    # Ghost-job policy: national vs regional
    # ---------------------------------
    df_national = df[df['ghostjob'] != True]
    df_regional = df.copy()

    # separate process for education, slimmer dataset
    df_educ_national = df_educ[df_educ['ghostjob'] != True]
    df_educ_regional = df_educ.copy()

    # -----------------------------
    # Time series by O*NET (open date default)
    # -----------------------------
    ts_onet_national = df_national.dropna(subset=['onet_norm']).groupby(['ymd_open', 'onet_norm']).size().reset_index(
        name='postings')
    ts_total_national = df_national.groupby('ymd_open').size().reset_index(name='postings')

    # -----------------------------
    # Geography (state / city / zip)
    # This is where we could add in other data?
    # Other data adding in: counts of job zones, and STEM Group types
    # -----------------------------
    print("Analyzing state postings...")
    by_state_seen = df_regional.groupby('state').size().reset_index(name='postings_seen').sort_values('postings_seen',ascending=False)

    print("Analyzing state postings by STEM group...")
    # Pivot tables for data
    by_state_STEMgroups = df_regional.pivot_table(
        index='state',
        columns='STEM Group',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing state posting age of STEM groups...")
    by_state_STEMgroups_post_age = df_regional.pivot_table(
        index='state',
        columns='STEM Group',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_state_STEMgroups_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_state_STEMgroups_post_age.columns = new_columns

    print("Analyzing state postings job zones...")
    by_state_jobzones = df_regional.pivot_table(
        index='state',
        columns='Job Zone',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    by_state_jobzones = by_state_jobzones.rename(columns=lambda col: f'Job Zone {col}' if isinstance(col, (int, float)) else col)

    print("Analyzing state postings job zones post age...")
    by_state_jobzones_post_age = df_regional.pivot_table(
        index='state',
        columns='Job Zone',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_state_jobzones_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_state_jobzones_post_age.columns = new_columns

    print("Analyzing state postings for educators...")
    # For educators
    by_state_teachers = df_educ_regional.pivot_table(
        index='state',
        columns='Education',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing state postings age for educators...")
    by_state_teachers_post_age = df_educ_regional.pivot_table(
        index='state',
        columns='Education',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_state_teachers_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_state_teachers_post_age.columns = new_columns

    # Merge operations - commenting out and putting into an excel sheet instead
    # by_state_seen = pd.merge(by_state_seen,by_state_STEMgroups,on='state',how='left')
    # by_state_seen = pd.merge(by_state_seen,by_state_STEMgroups_post_age,on='state',how='left')
    # by_state_seen = pd.merge(by_state_seen,by_state_jobzones,on='state',how='left')
    # by_state_seen = pd.merge(by_state_seen,by_state_jobzones_post_age,on='state',how='left')
    # by_state_seen = pd.merge(by_state_seen,by_state_teachers,on='state',how='left')
    # by_state_seen = pd.merge(by_state_seen,by_state_teachers_post_age,on='state',how='left')

    state_file = os.path.join(output_dir, "by_state_seen.xlsx")

    print(f"State analysis complete, writing results to {state_file}")
    with pd.ExcelWriter(state_file, engine='openpyxl') as writer:
        by_state_seen.to_excel(writer, sheet_name='StatePostings', index=False)
    with pd.ExcelWriter(state_file, engine='openpyxl', mode='a') as writer:
        by_state_STEMgroups.to_excel(writer, sheet_name='STEMGroups', index=False)
        by_state_jobzones.to_excel(writer, sheet_name='JobZones', index=False)
        by_state_teachers.to_excel(writer, sheet_name='Teachers', index=False)
        by_state_STEMgroups_post_age.to_excel(writer, sheet_name='STEMPAge', index=False)
        by_state_jobzones_post_age.to_excel(writer, sheet_name='JobZPAge', index=False)
        by_state_teachers_post_age.to_excel(writer, sheet_name='TeacherPAge', index=False)

    print("Analyzing jobs postings by city...")
    by_city_seen = df_regional.dropna(subset=['City']).groupby(['state', 'City']).size().reset_index(
        name='postings_seen').sort_values('postings_seen', ascending=False)

    print("Analyzing STEM group postings by city...")
    # Pivot tables for data
    by_city_STEMgroups = df_regional.pivot_table(
        index=['state','City'],
        columns='STEM Group',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing job zones postings by city...")
    by_city_jobzones = df_regional.pivot_table(
        index=['state','City'],
        columns='Job Zone',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing educator postings by city...")
    # For educators
    by_city_teachers = df_educ_regional.pivot_table(
        index=['state','City'],
        columns='Education',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing STEM group posting age by city...")
    by_city_STEMgroups_post_age = df_regional.pivot_table(
        index=['state','City'],
        columns='STEM Group',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_city_STEMgroups_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'City':
            new_columns.append('City')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_city_STEMgroups_post_age.columns = new_columns

    print("Analyzing job zone posting age by city...")
    by_city_jobzones_post_age = df_regional.pivot_table(
        index=['state','City'],
        columns='Job Zone',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_city_jobzones_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'City':
            new_columns.append('City')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_city_jobzones_post_age.columns = new_columns

    print("Analyzing educator posting age by city...")
    by_city_teachers_post_age = df_regional.pivot_table(
        index=['state','City'],
        columns='Education',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_city_teachers_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'City':
            new_columns.append('City')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_city_teachers_post_age.columns = new_columns

    by_city_jobzones = by_city_jobzones.rename(columns=lambda col: f'Job Zone {col}' if isinstance(col, (int, float)) else col)

    # Merge operations
    # by_city_seen = pd.merge(by_city_seen,by_city_STEMgroups,on='City',how='left')
    # by_city_seen = pd.merge(by_city_seen,by_city_STEMgroups_post_age,on='City',how='left')
    # by_city_seen = pd.merge(by_city_seen,by_city_jobzones,on='City',how='left')
    # by_city_seen = pd.merge(by_city_seen,by_city_jobzones_post_age,on='City',how='left')
    # by_city_seen = pd.merge(by_city_seen,by_city_teachers,on='City',how='left')
    # by_city_seen = pd.merge(by_city_seen,by_city_teachers_post_age,on='City',how='left')
    #
    city_file = os.path.join(output_dir, "by_city_seen.xlsx")

    print(f"City analysis complete, writing results to {city_file}")
    with pd.ExcelWriter(city_file, engine='openpyxl') as writer:
        by_city_seen.to_excel(writer, sheet_name='cityPostings', index=False)
    with pd.ExcelWriter(city_file, engine='openpyxl', mode='a') as writer:
        by_city_STEMgroups.to_excel(writer, sheet_name='STEMGroups', index=False)
        by_city_jobzones.to_excel(writer, sheet_name='JobZones', index=False)
        by_city_teachers.to_excel(writer, sheet_name='Teachers', index=False)
        by_city_STEMgroups_post_age.to_excel(writer, sheet_name='STEMPAge', index=False)
        by_city_jobzones_post_age.to_excel(writer, sheet_name='JobZPAge', index=False)
        by_city_teachers_post_age.to_excel(writer, sheet_name='TeacherPAge', index=False)

    print("Analyzing postings by zip...")
    by_zip_seen = df_regional.dropna(subset=['Zip']).groupby(['state', 'Zip']).size().reset_index(
        name='postings_seen').sort_values('postings_seen', ascending=False)

    print("Analyzing STEM group postings by zip...")
    # Pivot tables for data
    by_zip_STEMgroups = df_regional.pivot_table(
        index=['state','Zip'],
        columns='STEM Group',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing job zones postings by zip...")
    by_zip_jobzones = df_regional.pivot_table(
        index=['state','Zip'],
        columns='Job Zone',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing educator postings by zip...")
    # For educators
    by_zip_teachers = df_educ_regional.pivot_table(
        index=['state','Zip'],
        columns='Education',
        aggfunc='size',
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    print("Analyzing STEM group posting age by zip...")
    by_zip_STEMgroups_post_age = df_regional.pivot_table(
        index=['state','Zip'],
        columns='STEM Group',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_zip_STEMgroups_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'Zip':
            new_columns.append('Zip')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_zip_STEMgroups_post_age.columns = new_columns

    print("Analyzing job zone posting age by zip...")
    by_zip_jobzones_post_age = df_regional.pivot_table(
        index=['state','Zip'],
        columns='Job Zone',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_zip_jobzones_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'Zip':
            new_columns.append('Zip')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_zip_jobzones_post_age.columns = new_columns

    print("Analyzing educator posting age by zip...")
    by_zip_teachers_post_age = df_regional.pivot_table(
        index=['state','Zip'],
        columns='Education',
        values='Post_Age',
        aggfunc=['median','mean'],
        fill_value=0  # Fill any states with no postings in a category with 0
    ).reset_index()

    # Flatten pivot table
    new_columns = []
    for col in by_zip_teachers_post_age.columns.values:
        # Check if the column name is the 'state' index column
        if col[0] == 'Zip':
            new_columns.append('Zip')
        elif col[0] == 'state':
            new_columns.append('state')
        # For all other columns, join the MultiIndex levels with an underscore
        else:
            # Example: ('median', '0-30') becomes 'median_0-30'
            new_columns.append(f'{col[0]}_{col[1]}_Post Age')
    # Rename columns
    by_zip_teachers_post_age.columns = new_columns

    by_zip_jobzones = by_zip_jobzones.rename(columns=lambda col: f'Job Zone {col}' if isinstance(col, (int, float)) else col)

    # Merge operations
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_STEMgroups,on='Zip',how='left')
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_STEMgroups_post_age,on='Zip',how='left')
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_jobzones,on='Zip',how='left')
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_jobzones_post_age,on='Zip',how='left')
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_teachers,on='Zip',how='left')
    # by_zip_seen = pd.merge(by_zip_seen,by_zip_teachers_post_age,on='Zip',how='left')

    zip_file = os.path.join(output_dir, "by_zip_seen.xlsx")

    print(f"Zip analysis complete, writing results to {zip_file}")
    with pd.ExcelWriter(zip_file, engine='openpyxl') as writer:
        by_zip_seen.to_excel(writer, sheet_name='zipPostings', index=False)
    with pd.ExcelWriter(zip_file, engine='openpyxl', mode='a') as writer:
        by_zip_STEMgroups.to_excel(writer, sheet_name='STEMGroups', index=False)
        by_zip_jobzones.to_excel(writer, sheet_name='JobZones', index=False)
        by_zip_teachers.to_excel(writer, sheet_name='Teachers', index=False)
        by_zip_STEMgroups_post_age.to_excel(writer, sheet_name='STEMPAge', index=False)
        by_zip_jobzones_post_age.to_excel(writer, sheet_name='JobZPAge', index=False)
        by_zip_teachers_post_age.to_excel(writer, sheet_name='TeacherPAge', index=False)

    # -----------------------------
    # Pay normalization (to hourly where feasible)
    # -----------------------------
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
            return pd.Series([min_v * conv, max_v * conv])
        else:
            return pd.Series([pd.NA, pd.NA])


    has_pay = all(
        c in df_regional.columns for c in ["parameters_salary_min", "parameters_salary_max", "parameters_salary_unit"])
    if has_pay:
        pay_slice = df_regional.copy()
        pay_slice[['pay_min_hr', 'pay_max_hr']] = pay_slice.apply(normalize_to_hourly, axis=1)
        pay_slice = pay_slice.dropna(subset=['pay_min_hr', 'pay_max_hr'], how='all')

        pay_by_onet = pay_slice.groupby('onet_norm').agg(
            n_with_pay=('job_id', 'count'),
            median_min_hr=('pay_min_hr', 'median'),
            median_max_hr=('pay_max_hr', 'median')
        ).reset_index().sort_values('n_with_pay', ascending=False)
    else:
        pay_slice = pd.DataFrame()
        pay_by_onet = pd.DataFrame()

    # -----------------------------
    # Education / Experience / License (if available)
    # -----------------------------
    print("Analyzing educational requirements...")
    edu_counts = pd.DataFrame()
    if "requirements_min_education" in df_regional.columns:
        df_regional['req_min_edu'] = df_regional['requirements_min_education'].replace("", "Unspecified").fillna(
            "Unspecified")
        edu_counts = df_regional['req_min_edu'].value_counts().reset_index(name='postings').sort_values('postings',
                                                                                                        ascending=False)
        edu_counts.columns = ['req_min_edu', 'postings']  # Ensure consistent column names

    exp_counts = pd.DataFrame()
    if "requirements_experience" in df_regional.columns:
        df_regional['req_exp'] = df_regional['requirements_experience'].replace("", "Unspecified").fillna("Unspecified")
        exp_counts = df_regional['req_exp'].value_counts().reset_index(name='postings').sort_values('postings',
                                                                                                    ascending=False)
        exp_counts.columns = ['req_exp', 'postings']

    lic_counts = pd.DataFrame()
    if "requirements_license" in df_regional.columns:
        df_regional['req_lic'] = df_regional['requirements_license'].replace("", "Unspecified").fillna("Unspecified")
        lic_counts = df_regional['req_lic'].value_counts().reset_index(name='postings').sort_values('postings',
                                                                                                    ascending=False)
        lic_counts.columns = ['req_lic', 'postings']

    # -----------------------------
    # QA samples + outputs
    # -----------------------------
    qa_samples = df_regional[[
        'job_id', 'ymd_open', 'state', 'City', 'Zip', 'onet_norm', 'Title'
    ]].head(25)

    output_dir = "analysis_outputs"
    if dataset_base_path: output_dir = "analysis_outputs/" + output_dir + "_" + dataset_base_path
    os.makedirs(output_dir, exist_ok=True)
    print("Writing miscellaneous files...")
    ts_onet_national.to_csv(os.path.join(output_dir, "ts_onet_national.csv"), index=False)
    ts_total_national.to_csv(os.path.join(output_dir, "ts_total_national.csv"), index=False)
    by_state_seen.to_csv(os.path.join(output_dir, "by_state_seen.csv"), index=False)
    # by_city_seen.to_csv(os.path.join(output_dir, "by_city_seen.csv"), index=False)
    # by_zip_seen.to_csv(os.path.join(output_dir, "by_zip_seen.csv"), index=False)
    if not pay_slice.empty:
        pay_slice.to_csv(os.path.join(output_dir, "pay_slice_hourly.csv"), index=False)
    if not pay_by_onet.empty:
        pay_by_onet.to_csv(os.path.join(output_dir, "pay_by_onet.csv"), index=False)
    if not edu_counts.empty:
        edu_counts.to_csv(os.path.join(output_dir, "education_counts.csv"), index=False)
    if not exp_counts.empty:
        exp_counts.to_csv(os.path.join(output_dir, "experience_counts.csv"), index=False)
    if not lic_counts.empty:
        lic_counts.to_csv(os.path.join(output_dir, "license_counts.csv"), index=False)
    qa_samples.to_csv(os.path.join(output_dir, "qa_samples_head25.csv"), index=False)

    print("\nAnalysis outputs have been saved to the 'analysis_outputs' directory.")