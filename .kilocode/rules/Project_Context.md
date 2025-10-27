# Project Context
We are working with a historical dataset of U.S. job postings (2015–present), stored as monthly parquet files. The data covers S&E, S&E-related, and STEM middle-skill occupations.
Current ingestion is slow and monolithic: it reads everything, then filters, cleans, and pivots later.
We want to re-architect the parquet reader so that it only extracts and precomputes the fields necessary for 10 specific analytical goals.

# Analytical Goals for Reader Design
1. Posting age trend analysis
2. Hard-to-fill signal detection
3. Geographic intensity (state/city/zip)
4. Occupation & ONET zone analysis
5. Credential shifts (education, experience, license)
6. Ghost job filtering
7. Pay normalization to hourly
8. Remote vs onsite flag extraction
9. Pipeline friction indicators (lag metrics)
10. Top-N aggregation by key attributes

# Input Data
- Parquet files with columns including (at least):
  job_id, date_compiled, date_acquired, expired, expired_date,
  state, city, zipcode, classifications_onet_code, classifications_onet_code_25,
  title, description, ghostjob, jobclass,
  parameters_salary_min, parameters_salary_max, parameters_salary_unit,
  requirements_min_education, requirements_experience, requirements_license, application_company
- O*NET & STEM lookup tables are available (CSV)
- Files are stored monthly with YYYY-MM in filename

# Kilocode Tasks
1. **Design a new parquet reader module** (`src/reader/etl.py`) that:
   - Reads parquet with DuckDB or pyarrow efficiently
   - Pre-filters columns relevant to the 10 analytical goals
   - Normalizes ONET codes, pay fields, and credential text
   - Calculates posting age and pipeline lag fields once at ingest
   - Tags records with derived flags (ghost vs real, remote vs onsite if available)

2. **Add early aggregation**:
   - Optionally precompute state/city/zip × ONET × month aggregates
   - Store intermediate feather/parquet tables optimized for analysis (wide or long)

3. **Provide a CLI entrypoint** (`read_and_extract.py`) that:
   - Accepts args like `--year 2022 --month 06` or `--range 2018-01 2019-12`
   - Outputs cleaned parquet or feather files in a `clean/` folder
   - Logs row counts, column shapes, and derived metric summaries

4. **Structure the project**:
src/
reader/
init.py
etl.py # main ingest and precompute logic
transforms/
summaries.py # downstream aggregation (state, ONET, etc.)
tests/
test_etl_mock.py # mock tests using fabricated data
scripts/
read_and_extract.py

pgsql
Copy code

5. **Generate tests with mock data** so we can run and validate logic without waiting for a full 45-minute ingest.

6. Use **Pandas + DuckDB + PyArrow**, minimal dependencies, and readable code with comments.

7. Document each transformation step with short docstrings (esp. derived fields: posting_age, lag_days, hourly pay).

# Output Expectations
- Efficient parquet reader that pulls only relevant columns
- Derived columns for posting age, lag, normalized pay
- Structured intermediate outputs ready for time series and geo analysis
- CLI tool to batch-process historical files
- Mock data test coverage