import pandas as pd
import geopandas as gpd
import geodatasets as gds
from geodatasets import get_path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_hard_to_fill_heatmap(df: pd.DataFrame, geography_col: str = 'state', category_col: str = 'onet_norm'):
    """
    Plot a geographic heatmap of hard-to-fill jobs by geography and job category.

    Args:
        df (pd.DataFrame): DataFrame with columns for geography, job category, and hard-to-fill count.
        geography_col (str): Column name for geography (e.g., 'state').
        category_col (str): Column name for job category (e.g., 'onet_norm').

    Returns:
        None: Displays a matplotlib plot.
    """
    # Load US states shapefile from geopandas datasets
    # print(sorted(gds.data.flatten().keys()))
    # exit()
    geo_path = get_path("naturalearth.land")
    usa = gpd.read_file(geo_path)
    # usa = usa[usa['name'] == 'United States of America']

    # For state-level, use built-in states shapefile from geopandas or external source
    # Here we use a common US states shapefile from geopandas datasets (requires geopandas 0.10+)
    states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")

    # Aggregate data by geography and category
    agg = df.groupby([geography_col, category_col])['hard_to_fill_count'].sum().reset_index()

    # Create a color map for categories
    categories = agg[category_col].unique()
    colors = plt.cm.get_cmap('tab20', len(categories))
    color_map = {cat: colors(i) for i, cat in enumerate(categories)}

    # Prepare figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot each category separately with its color
    for cat in categories:
        cat_data = agg[agg[category_col] == cat]
        merged = states.merge(cat_data, left_on='id', right_on=geography_col, how='left')
        merged['hard_to_fill_count'] = merged['hard_to_fill_count'].fillna(0)

        # Plot with alpha proportional to count (normalized)
        max_count = agg['hard_to_fill_count'].max()
        merged.plot(column='hard_to_fill_count',
                    ax=ax,
                    color=color_map[cat],
                    alpha=merged['hard_to_fill_count'] / max_count,
                    edgecolor='black',
                    linewidth=0.5,
                    label=cat)

    ax.set_title('Hard-to-Fill Jobs Heatmap by Geography and Job Category')
    ax.axis('off')
    ax.legend(title='Job Category', loc='lower left')

    plt.show()

def plot_from_csv(csv_path: str, geography_col: str = 'state', category_col: str = 'onet_norm'):
    """
    Load hard-to-fill aggregate data from CSV and plot heatmap.

    Args:
        csv_path (str): Path to the CSV file with hard-to-fill aggregates.
        geography_col (str): Column name for geography.
        category_col (str): Column name for job category.

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    plot_hard_to_fill_heatmap(df, geography_col, category_col)

if __name__ == "__main__":
    # Example usage with dummy data
    data = {
        'state': ['01', '01', '02', '02', '04', '04'],
        'onet_norm': ['Nurse', 'Engineer', 'Nurse', 'Engineer', 'Nurse', 'Engineer'],
        'hard_to_fill_count': [50, 30, 20, 10, 40, 25]
    }
    df = pd.DataFrame(data)

    # The 'state' codes here are FIPS codes matching the shapefile 'id' field
    plot_hard_to_fill_heatmap(df)
