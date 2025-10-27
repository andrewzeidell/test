# Parquet Reader Module and CLI Script Documentation

## Overview

The new parquet reader module in `src/reader/etl.py` is designed for efficient ingestion and preprocessing of large job posting datasets stored as parquet files. It leverages DuckDB for fast, columnar reads and supports selective column loading to optimize performance. The module reads parquet files from a specified directory, optionally filtered by a date pattern, and returns a Pandas DataFrame with the selected columns.

The CLI script `scripts/read_and_extract.py` provides a command-line interface to:
- Read parquet files using the reader module
- Apply a series of transformations and derived field computations (e.g., posting age, pay normalization, O*NET code normalization)
- Merge lookup tables for enrichment (e.g., O*NET lookups)
- Save the cleaned and transformed data in parquet or feather format

This architecture supports modular, extensible data ingestion tailored to the analytical goals of the ML_Project.

## Setting Up and Running the CLI Script in PyCharm

1. **Open the ML_Project in PyCharm.**

2. **Configure a Python interpreter** with the required dependencies installed (see Dependencies section).

3. **Create a Run Configuration:**
   - Go to `Run > Edit Configurations...`
   - Click `+` and select `Python`
   - Name the configuration (e.g., `Read and Extract`)
   - Set the script path to `<project_root>/scripts/read_and_extract.py`
   - Set the parameters, for example:
     ```
     --input_dir data/raw --date_pattern 2022-06 --output_dir data/clean --format parquet
     ```
   - Set the working directory to the project root directory.

4. **Run the configuration** to execute the script with the specified arguments.

## Dependencies and Installation

The project requires the following Python packages:

- `duckdb`
- `pandas`
- `pyarrow` (for feather file support)
- `argparse` (standard library)
- Other dependencies may be required by transformation and lookup modules.

### Portable Installation

Use a virtual environment to isolate dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:

```
duckdb
pandas
pyarrow
```

This setup ensures consistent environments across machines and IDEs.

## Command-Line Usage Examples

Basic usage reading June 2022 data and outputting parquet:

```bash
python scripts/read_and_extract.py --input_dir data/raw --date_pattern 2022-06 --output_dir data/clean --format parquet
```

Output feather format:

```bash
python scripts/read_and_extract.py --input_dir data/raw --date_pattern 2023-01 --output_dir data/clean --format feather
```

Process all parquet files in input directory (no date filter):

```bash
python scripts/read_and_extract.py --input_dir data/raw --output_dir data/clean
```

## Extending or Modifying the Reader

- **Adding new analytical goals:**
  - Update the `columns` list in `scripts/read_and_extract.py` to include any new required columns.
  - Implement new transformation functions in `src/reader/transforms/transformations.py`.
  - Add calls to these new transformations in the `apply_transformations` function.
  - Update lookup merges as needed in the CLI script.

- **Derived fields:**
  - The `src/reader/etl.py` module has placeholders for adding derived field computations during reading.
  - For complex transformations, prefer applying them after reading in the CLI script.

- **Lookup tables:**
  - Place new lookup CSV files in the appropriate lookup directory.
  - Use or extend `lookup_utils` to load and merge these tables.

## Directory Structure Notes

- **Raw data:** Place raw parquet files in `data/raw/`. Files should be named with date patterns (e.g., `2022-06.parquet`) for filtering.

- **Lookups:** Lookup CSV files (e.g., `STEM Groups in the BLS Projections.csv`, `ONET_Job_Zones.csv`, `SOC_Codes.csv`) should be stored in a designated lookup directory (e.g., `data/lookups/` or as configured in `lookup_utils`).

- **Output:** Cleaned and transformed data files are saved to the specified output directory (e.g., `data/clean/`), in parquet or feather format.

## Troubleshooting Tips

- **Input directory not found:** Ensure the `--input_dir` path exists and contains parquet files.

- **Missing columns:** If transformations fail, verify that all required columns are included in the `columns` list in the CLI script.

- **Dependency errors:** Confirm all required packages are installed in the active Python environment.

- **File write errors:** Check permissions and existence of the output directory.

- **Large memory usage:** Consider filtering columns and date ranges to reduce data size.

- **Logging:** The CLI script outputs informative logs to the console; review these for error details.

---

This documentation should be included in the project docs or README to assist users and developers in effectively using and extending the parquet reader and CLI tools.