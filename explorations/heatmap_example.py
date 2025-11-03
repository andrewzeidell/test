# import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from src.reader.transforms.lookup_utils import add_fips_code_from_csv
#
# # URLs for higher-resolution Census shapefiles
# STATES_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
# COUNTIES_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
# ZCTAS_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
#
# def load_shapefile(geography_level: str) -> gpd.GeoDataFrame:
#     """
#     Load the shapefile GeoDataFrame for the specified geography level.
#
#     Args:
#         geography_level (str): One of 'state', 'county', 'zip', or 'city'.
#
#     Returns:
#         gpd.GeoDataFrame: The loaded GeoDataFrame.
#     """
#     if geography_level == 'state':
#         return gpd.read_file(STATES_SHP_URL)
#     elif geography_level == 'county':
#         return gpd.read_file(COUNTIES_SHP_URL)
#     elif geography_level == 'zip':
#         return gpd.read_file(ZCTAS_SHP_URL)
#     elif geography_level == 'city':
#         # Placeholder for city-level shapefile loading
#         raise NotImplementedError("City-level heatmap is not implemented yet.")
#     else:
#         raise ValueError(f"Unsupported geography level: {geography_level}")
#
# def plot_hard_to_fill_heatmap(
#     df: pd.DataFrame,
#     geography_level: str = 'state',
#     category_col: str = 'onet_norm',
#     state_fips_csv: str = None
# ):
#     """
#     Plot a geographic heatmap of hard-to-fill jobs by geography and job category.
#
#     Args:
#         df (pd.DataFrame): DataFrame with columns for geography, job category, and hard-to-fill count.
#         geography_level (str): Geography level to plot ('state', 'zip', or 'city').
#         category_col (str): Column name for job category (e.g., 'onet_norm').
#         state_fips_csv (str): Optional path to CSV mapping state abbreviations to FIPS codes (required for 'state' level if df uses abbreviations).
#
#     Returns:
#         None: Displays a matplotlib plot.
#     """
#     # Load shapefile for the geography level
#     geo_df = load_shapefile(geography_level)
#
#     # Prepare join keys and merge input data with shapefile
#     if geography_level == 'state':
#         # If df has state abbreviations, convert to FIPS using lookup_utils
#         if state_fips_csv is not None:
#             df = add_fips_code_from_csv(df, state_fips_csv, state_abbr_col='state', fips_code_col='fips_code')
#             geography_col = 'fips_code'
#         else:
#             geography_col = 'state'  # assume df already has FIPS codes as strings
#         join_left = geography_col
#         join_right = 'STATEFP'
#     elif geography_level == 'zip':
#         join_left = 'zip'
#         join_right = 'ZCTA5CE20'
#     elif geography_level == 'city':
#         # Placeholder for city-level join keys
#         raise NotImplementedError("City-level heatmap is not implemented yet.")
#     else:
#         raise ValueError(f"Unsupported geography level: {geography_level}")
#
#     # Aggregate data by geography and category
#     agg = df.groupby([join_left, category_col])['hard_to_fill_count'].sum().reset_index()
#
#     # Merge aggregated data with shapefile
#     merged = geo_df.merge(agg, left_on=join_right, right_on=join_left, how='left')
#     merged['hard_to_fill_count'] = merged['hard_to_fill_count'].fillna(0)
#
#     # Create a color map for categories
#     categories = agg[category_col].unique()
#     colors = plt.cm.get_cmap('tab20', len(categories))
#     color_map = {cat: colors(i) for i, cat in enumerate(categories)}
#
#     # Prepare figure and axis
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#
#     # Plot each category separately with its color
#     max_count = agg['hard_to_fill_count'].max()
#     for cat in categories:
#         cat_data = merged[merged[category_col] == cat]
#         cat_data.plot(
#             column='hard_to_fill_count',
#             ax=ax,
#             color=color_map[cat],
#             alpha=cat_data['hard_to_fill_count'] / max_count,
#             edgecolor='black',
#             linewidth=0.5,
#             label=cat
#         )
#
#     ax.set_title(f'Hard-to-Fill Jobs Heatmap by {geography_level.capitalize()} and Job Category')
#     ax.axis('off')
#     ax.legend(title='Job Category', loc='lower left')
#
#     plt.show()
#
# def plot_from_csv(csv_path: str, geography_level: str = 'state', category_col: str = 'onet_norm', state_fips_csv: str = None):
#     """
#     Load hard-to-fill aggregate data from CSV and plot heatmap.
#
#     Args:
#         csv_path (str): Path to the CSV file with hard-to-fill aggregates.
#         geography_level (str): Geography level to plot ('state', 'zip', or 'city').
#         category_col (str): Column name for job category.
#         state_fips_csv (str): Optional path to CSV mapping state abbreviations to FIPS codes.
#
#     Returns:
#         None
#     """
#     df = pd.read_csv(csv_path)
#     plot_hard_to_fill_heatmap(df, geography_level, category_col, state_fips_csv)
#
# if __name__ == "__main__":
#     # Example usage with dummy data for state level
#     # data_state = {
#     #     'state': ['CA', 'CA', 'NY', 'NY', 'TX', 'TX'],
#     #     'onet_norm': ['Nurse', 'Engineer', 'Nurse', 'Engineer', 'Nurse', 'Engineer'],
#     #     'hard_to_fill_count': [50, 30, 20, 10, 40, 25]
#     # }
#     # df_state = pd.DataFrame(data_state)
#
#     # Path to CSV mapping state abbreviations to FIPS codes (example path)
#     state_fips_csv_path = r"C:\Users\azeidell\OneDrive - National Science Foundation\Documents\Andrew\2023-24\1. Projects\NSDS\NLx\data\lookups\state_fips.csv"
#
#     # # Plot state-level heatmap (requires state_fips_csv_path)
#     # plot_hard_to_fill_heatmap(df_state, geography_level='state', category_col='onet_norm', state_fips_csv=state_fips_csv_path)
#
#     data_path = r"C:\Users\azeidell\OneDrive - National Science Foundation\Documents\Andrew\2023-24\1. Projects\NSDS\NLx\data\clean\aggregates\unredacted__job__2024-06\hard_to_fill_signals_state.csv"
#
#     plot_from_csv(csv_path=data_path,state_fips_csv=state_fips_csv_path)
#
#     # # Example usage with dummy data for zip level
#     # data_zip = {
#     #     'zip': ['90001', '90002', '10001', '10002', '73301', '73344'],
#     #     'onet_norm': ['Nurse', 'Engineer', 'Nurse', 'Engineer', 'Nurse', 'Engineer'],
#     #     'hard_to_fill_count': [15, 10, 25, 5, 20, 8]
#     # }
#     # df_zip = pd.DataFrame(data_zip)
#     #
#     # # Plot zip-level heatmap
#     # plot_hard_to_fill_heatmap(df_zip, geography_level='zip', category_col='onet_norm')

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.reader.transforms.lookup_utils import add_fips_code_from_csv

# URLs for higher-resolution Census shapefiles
STATES_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip"
COUNTIES_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
ZCTAS_SHP_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"

def load_shapefile(geography_level: str) -> gpd.GeoDataFrame:
    """
    Load the shapefile GeoDataFrame for the specified geography level.

    Args:
        geography_level (str): One of 'state', 'county', 'zip', or 'city'.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    if geography_level == 'state':
        return gpd.read_file(STATES_SHP_URL)
    elif geography_level == 'county':
        return gpd.read_file(COUNTIES_SHP_URL)
    elif geography_level == 'zip':
        return gpd.read_file(ZCTAS_SHP_URL)
    elif geography_level == 'city':
        # Placeholder for city-level shapefile loading
        raise NotImplementedError("City-level heatmap is not implemented yet.")
    else:
        raise ValueError(f"Unsupported geography level: {geography_level}")

def filter_top_onet_per_geography(
    df: pd.DataFrame,
    geography_col: str,
    category_col: str = 'onet_norm',
    count_col: str = 'hard_to_fill_count'
) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only the top ONET code by hard-to-fill count per geography unit.

    Args:
        df (pd.DataFrame): Input DataFrame with geography, category, and count columns.
        geography_col (str): Column name for geography unit (e.g., 'state', 'zip', or 'fips_code').
        category_col (str): Column name for job category.
        count_col (str): Column name for hard-to-fill count.

    Returns:
        pd.DataFrame: Filtered DataFrame with only the top ONET code per geography.
    """
    # Sort by geography, count descending
    df_sorted = df.sort_values([geography_col, count_col], ascending=[True, False])
    # Keep top row per geography
    top_df = df_sorted.groupby(geography_col).head(1).reset_index(drop=True)
    return top_df

def plot_hard_to_fill_heatmap(
    df: pd.DataFrame,
    geography_level: str = 'state',
    category_col: str = 'onet_norm',
    state_fips_csv: str = None
):
    """
    Plot a geographic heatmap of hard-to-fill jobs by geography and job category,
    showing only the top ONET code by hard-to-fill count per geography unit.

    Args:
        df (pd.DataFrame): DataFrame with columns for geography, job category, and hard-to-fill count.
        geography_level (str): Geography level to plot ('state', 'zip', or 'city').
        category_col (str): Column name for job category (e.g., 'onet_norm').
        state_fips_csv (str): Optional path to CSV mapping state abbreviations to FIPS codes (required for 'state' level if df uses abbreviations).

    Returns:
        None: Displays a matplotlib plot.
    """
    # Load shapefile for the geography level
    geo_df = load_shapefile(geography_level)
    geo_df =geo_df[~geo_df['STUSPS'].isin(['GU','AS','VI','MP'])]

    # Prepare join keys and merge input data with shapefile
    if geography_level == 'state':
        # If df has state abbreviations, convert to FIPS using lookup_utils
        if state_fips_csv is not None:
            df = add_fips_code_from_csv(df, state_fips_csv, state_abbr_col='state', fips_code_col='fips_code')
            geography_col = 'fips_code'
        else:
            geography_col = 'state'  # assume df already has FIPS codes as strings
        join_left = geography_col
        join_right = 'STATEFP'
    elif geography_level == 'zip':
        geography_col = 'zip'
        join_left = geography_col
        join_right = 'ZCTA5CE20'
    elif geography_level == 'city':
        # Placeholder for city-level join keys
        raise NotImplementedError("City-level heatmap is not implemented yet.")
    else:
        raise ValueError(f"Unsupported geography level: {geography_level}")

    # Filter to top ONET code per geography unit
    df_top = filter_top_onet_per_geography(df, geography_col, category_col, 'hard_to_fill_count')

    # Aggregate data by geography and category (should be single row per geography after filtering)
    agg = df_top.groupby([join_left, category_col])['hard_to_fill_count'].sum().reset_index()

    # Merge aggregated data with shapefile
    merged = geo_df.merge(agg, left_on=join_right, right_on=join_left, how='left')
    merged = merged.dropna()
    merged['hard_to_fill_count'] = merged['hard_to_fill_count'].fillna(0)


    # Create a color map for categories (only top categories)
    categories = agg[category_col].unique()
    colors = plt.cm.get_cmap('tab20', len(categories))
    color_map = {cat: colors(i) for i, cat in enumerate(categories)}

    # Prepare figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot each category separately with its color
    max_count = agg['hard_to_fill_count'].max()
    for cat in categories:
        cat_data = merged[merged[category_col] == cat]
        cat_data.plot(
            column='hard_to_fill_count',
            ax=ax,
            color=color_map[cat],
            alpha=cat_data['hard_to_fill_count'] / cat_data['hard_to_fill_count'].max(), # This normalizes to the cat
            edgecolor='black',
            linewidth=0.5,
            label=cat
        )

    ax.set_title(f'Hard-to-Fill Jobs Heatmap by {geography_level.capitalize()} and Top Job Category')
    ax.axis('off')
    ax.legend(title='Job Category', loc='lower left')
    ax.set_xlim(-180,-60)
    ax.set_ylim(10,75)

    plt.show()

def plot_from_csv(csv_path: str, geography_level: str = 'state', category_col: str = 'onet_norm', state_fips_csv: str = None):
    """
    Load hard-to-fill aggregate data from CSV and plot heatmap.

    Args:
        csv_path (str): Path to the CSV file with hard-to-fill aggregates.
        geography_level (str): Geography level to plot ('state', 'zip', or 'city').
        category_col (str): Column name for job category.
        state_fips_csv (str): Optional path to CSV mapping state abbreviations to FIPS codes.

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    plot_hard_to_fill_heatmap(df, geography_level, category_col, state_fips_csv)

if __name__ == "__main__":
    # Example usage with dummy data for state level
    data_state = {
        'state': ['CA', 'CA', 'NY', 'NY', 'TX', 'TX'],
        'onet_norm': ['Nurse', 'Engineer', 'Nurse', 'Engineer', 'Nurse', 'Engineer'],
        'hard_to_fill_count': [50, 30, 20, 10, 40, 25]
    }
    df_state = pd.DataFrame(data_state)

    # Path to CSV mapping state abbreviations to FIPS codes (example path)
    state_fips_csv_path = r"C:\Users\azeidell\OneDrive - National Science Foundation\Documents\Andrew\2023-24\1. Projects\NSDS\NLx\data\lookups\state_fips.csv"

    data_path = r"C:\Users\azeidell\OneDrive - National Science Foundation\Documents\Andrew\2023-24\1. Projects\NSDS\NLx\data\clean\aggregates\unredacted__job__2024-06\hard_to_fill_signals_state.csv"

    plot_from_csv(csv_path=data_path,state_fips_csv=state_fips_csv_path)

    # # Plot state-level heatmap (requires state_fips_csv_path)
    # plot_hard_to_fill_heatmap(df_state, geography_level='state', category_col='onet_norm', state_fips_csv=state_fips_csv_path)
    #
    # # Example usage with dummy data for zip level
    # data_zip = {
    #     'zip': ['90001', '90002', '10001', '10002', '73301', '73344'],
    #     'onet_norm': ['Nurse', 'Engineer', 'Nurse', 'Engineer', 'Nurse', 'Engineer'],
    #     'hard_to_fill_count': [15, 10, 25, 5, 20, 8]
    # }
    # df_zip = pd.DataFrame(data_zip)
    #
    # # Plot zip-level heatmap
    # plot_hard_to_fill_heatmap(df_zip, geography_level='zip', category_col='onet_norm')
    #
    # # Example usage demonstrating filtering to top ONET code per geography unit
    # print("Example: Plotting only top ONET code per state by hard-to-fill count")
    # plot_hard_to_fill_heatmap(df_state, geography_level='state', category_col='onet_norm', state_fips_csv=state_fips_csv_path)
