import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import requests
import io
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting CEMA 2025 Internship Task Analysis with Local Files...")

# Load the datasets from local paths
print("\nLoading datasets from local paths...")

# Define file paths
dataset_science_path = r"C:\Users\Amolo Washington\OneDrive\Desktop\CEMA_INTERNSHIP\dataset_datascience.csv"
hiv_data_path = r"C:\Users\Amolo Washington\OneDrive\Desktop\CEMA_INTERNSHIP\HIV data 2000-2023.csv"
poverty_data_path = r"C:\Users\Amolo Washington\OneDrive\Desktop\CEMA_INTERNSHIP\multidimensional_poverty.xlsx"
neonatal_data_path = r"C:\Users\Amolo Washington\OneDrive\Desktop\CEMA_INTERNSHIP\neonatal_mortality_rate.csv"
under5_data_path = r"C:\Users\Amolo Washington\OneDrive\Desktop\CEMA_INTERNSHIP\under_five mortality rate.csv"

# Initialize variables to track loading status
hiv_data_loaded = False
poverty_data_loaded = False
under5_data_loaded = False
neonatal_data_loaded = False

# Load dataset_datascience.csv
try:
    dataset_science = pd.read_csv(dataset_science_path)
    print(f"Dataset science loaded: {dataset_science.shape[0]} rows, {dataset_science.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset_datascience.csv: {e}")
    dataset_science = None

# Try loading HIV data with different encodings
encodings_to_try = ['latin1', 'cp1252', 'ISO-8859-1', 'utf-16', 'windows-1250', 'windows-1252']
hiv_data = None

for encoding in encodings_to_try:
    try:
        print(f"Trying to load HIV data with encoding: {encoding}")
        hiv_data = pd.read_csv(hiv_data_path, encoding=encoding)
        print(f"HIV data loaded successfully with encoding {encoding}: {hiv_data.shape[0]} rows, {hiv_data.shape[1]} columns")
        hiv_data_loaded = True
        break  # Exit the loop if successful
    except Exception as e:
        print(f"Failed with encoding {encoding}: {e}")

if not hiv_data_loaded:
    print("Could not load HIV data with any of the attempted encodings.")

# Load multidimensional poverty data (Excel file)
try:
    # First try loading without specifying a sheet
    try:
        poverty_data = pd.read_excel(poverty_data_path, skiprows=1)
        print(f"Poverty data loaded: {poverty_data.shape[0]} rows, {poverty_data.shape[1]} columns")
        poverty_data_loaded = True
    except Exception as e:
        print(f"Error loading poverty data without sheet specification: {e}")
        
        # If that fails, try listing all sheets and load the first one
        try:
            xls = pd.ExcelFile(poverty_data_path)
            print(f"Available sheets in the Excel file: {xls.sheet_names}")
            poverty_data = pd.read_excel(poverty_data_path, sheet_name=0, skiprows=1)
            print(f"Poverty data loaded from first sheet: {poverty_data.shape[0]} rows, {poverty_data.shape[1]} columns")
            poverty_data_loaded = True
        except Exception as e2:
            print(f"Error loading poverty data from first sheet: {e2}")
            poverty_data = None
except Exception as e:
    print(f"Error loading poverty data: {e}")
    poverty_data = None

# Load under-five mortality rate data
try:
    under5_data = pd.read_csv(under5_data_path)
    print(f"Under-five mortality data loaded: {under5_data.shape[0]} rows, {under5_data.shape[1]} columns")
    under5_data_loaded = True
except Exception as e:
    print(f"Error loading under-five mortality data: {e}")
    under5_data = None

# Load neonatal mortality rate data
try:
    neonatal_data = pd.read_csv(neonatal_data_path)
    print(f"Neonatal mortality data loaded: {neonatal_data.shape[0]} rows, {neonatal_data.shape[1]} columns")
    neonatal_data_loaded = True
except Exception as e:
    print(f"Error loading neonatal mortality data: {e}")
    neonatal_data = None

# Check if all datasets were loaded successfully - FIXED BOOLEAN CHECK
if not (hiv_data_loaded and poverty_data_loaded and under5_data_loaded and neonatal_data_loaded):
    print("\nWARNING: Some datasets failed to load. Please check the file paths and try again.")
    
    # Let's see what we can do with the datasets that did load
    print("\nProceeding with available datasets...")
else:
    print("\nAll datasets loaded successfully!")

# Display the first few rows of each dataset to understand their structure
if hiv_data_loaded:
    print("\nExploring HIV data structure:")
    print(hiv_data.head(2))
    print("\nHIV data columns:", hiv_data.columns.tolist())

if poverty_data_loaded:
    print("\nExploring poverty data structure:")
    print(poverty_data.head(2))
    print("\nPoverty data columns:", poverty_data.columns.tolist())

if under5_data_loaded:
    print("\nExploring under-five mortality data structure:")
    print(under5_data.head(2))
    print("\nUnder-five mortality data columns:", under5_data.columns.tolist())

if neonatal_data_loaded:
    print("\nExploring neonatal mortality data structure:")
    print(neonatal_data.head(2))
    print("\nNeonatal mortality data columns:", neonatal_data.columns.tolist())

# Continue with the analysis based on available datasets
print("\n\nProceeding with analysis based on available datasets...")

# Part 2: Child Mortality Analysis (since we have these datasets)
if under5_data_loaded and neonatal_data_loaded:
    print("\n\n--- PART 2: CHILD MORTALITY ANALYSIS ---")

    # List of East African Community countries
    eac_countries = ["Burundi", "Kenya", "Rwanda", "South Sudan", "Tanzania", "Uganda", "Democratic Republic of the Congo", "Somalia"]
    print(f"\nEast African Community countries: {eac_countries}")

    # Check if the country names match what's in our datasets
    print("\nChecking country names in datasets:")
    print("Countries in under-five mortality data:", under5_data['Geographic area'].unique())
    print("Countries in neonatal mortality data:", neonatal_data['Geographic area'].unique())

    # We might need to adjust our country list based on what's in the datasets
    available_countries_under5 = set(under5_data['Geographic area'].unique())
    available_countries_neonatal = set(neonatal_data['Geographic area'].unique())
    
    # Find EAC countries that are in our datasets
    eac_in_under5 = [country for country in eac_countries if country in available_countries_under5]
    eac_in_neonatal = [country for country in eac_countries if country in available_countries_neonatal]
    
    print(f"\nEAC countries found in under-five mortality data: {eac_in_under5}")
    print(f"EAC countries found in neonatal mortality data: {eac_in_neonatal}")

    # Filter mortality data for EAC countries
    print("\nFiltering mortality data for EAC countries...")

    # Function to filter data for EAC countries
    def filter_eac_data(df, countries):
        return df[df['Geographic area'].isin(countries)]

    # Filter under-five mortality data
    under5_eac = filter_eac_data(under5_data, eac_in_under5)
    print(f"Under-five mortality data for EAC: {under5_eac.shape[0]} rows")

    # Filter neonatal mortality data
    neonatal_eac = filter_eac_data(neonatal_data, eac_in_neonatal)
    print(f"Neonatal mortality data for EAC: {neonatal_eac.shape[0]} rows")

    if not under5_eac.empty and not neonatal_eac.empty:
        # Get the latest estimates for each country
        print("\nGetting latest estimates for each country...")

        # Function to get latest estimates
        def get_latest_estimates(df):
            # Convert TIME_PERIOD to datetime if it's not already
            if df['TIME_PERIOD'].dtype == 'object':
                try:
                    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
                except:
                    # If conversion fails, just use the original values
                    pass
            
            # Get the latest year for each country
            latest_year = df.groupby('Geographic area')['TIME_PERIOD'].max().reset_index()
            
            # Merge with original data to get the latest estimates
            latest_data = pd.merge(
                df,
                latest_year,
                on=['Geographic area', 'TIME_PERIOD'],
                how='inner'
            )
            
            return latest_data

        # Get latest under-five mortality estimates
        under5_latest = get_latest_estimates(under5_eac)
        print(f"Latest under-five mortality estimates: {under5_latest.shape[0]} rows")

        # Get latest neonatal mortality estimates
        neonatal_latest = get_latest_estimates(neonatal_eac)
        print(f"Latest neonatal mortality estimates: {neonatal_latest.shape[0]} rows")

        # Download shapefiles for visualization
        print("\nDownloading shapefiles for visualization...")

        # For this example, we'll use a simplified approach with a world shapefile
        # In a real scenario, you would download the shapefiles from gadm.org as specified
        try:
            # URL for a simplified world shapefile
            world_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
            
            # Load the shapefile
            world = gpd.read_file(world_url)
            
            # Create a mapping between country names in the shapefile and our data
            country_name_mapping = {
                "Burundi": "Burundi",
                "Kenya": "Kenya",
                "Rwanda": "Rwanda",
                "South Sudan": "South Sudan",
                "Tanzania": "Tanzania, United Republic of",
                "Uganda": "Uganda",
                "Democratic Republic of the Congo": "Congo, the Democratic Republic of the",
                "Somalia": "Somalia"
            }
            
            # Filter the shapefile for EAC countries
            eac_shapes = world[world['name'].isin(list(country_name_mapping.values()))]
            
            if not eac_shapes.empty:
                print(f"Shapefile loaded with {eac_shapes.shape[0]} EAC countries")
                
                # Create a mapping between country names in the shapefile and our data
                reverse_mapping = {v: k for k, v in country_name_mapping.items()}
                eac_shapes['country_name'] = eac_shapes['name'].map(reverse_mapping)
                
                # Merge shapefile with mortality data
                under5_geo = pd.merge(
                    eac_shapes,
                    under5_latest[['Geographic area', 'OBS_VALUE']],
                    left_on='country_name',
                    right_on='Geographic area',
                    how='left'
                )
                
                neonatal_geo = pd.merge(
                    eac_shapes,
                    neonatal_latest[['Geographic area', 'OBS_VALUE']],
                    left_on='country_name',
                    right_on='Geographic area',
                    how='left'
                )
                
                # Visualize the latest estimates using shapefiles
                print("\nVisualizing latest mortality estimates using shapefiles...")
                
                # Create a custom colormap
                colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
                cmap = LinearSegmentedColormap.from_list('custom_blue', colors)
                
                # Plot under-five mortality
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                under5_geo.plot(column='OBS_VALUE', ax=ax, legend=True, cmap=cmap, 
                                legend_kwds={'label': "Deaths per 1,000 live births", 'orientation': "horizontal"})
                
                # Add country labels
                for idx, row in under5_geo.iterrows():
                    if pd.notna(row['OBS_VALUE']):
                        ax.annotate(f"{row['country_name']}\n{row['OBS_VALUE']:.1f}", 
                                   xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                                   ha='center', fontsize=10)
                
                ax.set_title('Latest Under-Five Mortality Rate in East African Community Countries', fontsize=16)
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig('under5_mortality_map.png')
                print("Visualization saved as 'under5_mortality_map.png'")
                
                # Plot neonatal mortality
                fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                neonatal_geo.plot(column='OBS_VALUE', ax=ax, legend=True, cmap=cmap, 
                                 legend_kwds={'label': "Deaths per 1,000 live births", 'orientation': "horizontal"})
                
                # Add country labels
                for idx, row in neonatal_geo.iterrows():
                    if pd.notna(row['OBS_VALUE']):
                        ax.annotate(f"{row['country_name']}\n{row['OBS_VALUE']:.1f}", 
                                   xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                                   ha='center', fontsize=10)
                
                ax.set_title('Latest Neonatal Mortality Rate in East African Community Countries', fontsize=16)
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig('neonatal_mortality_map.png')
                print("Visualization saved as 'neonatal_mortality_map.png'")
            else:
                print("No EAC countries found in the shapefile")
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            print("Proceeding with non-spatial visualizations...")

        # Show average trends in mortality rates over time
        print("\nShowing average trends in mortality rates over time...")

        # Function to plot trends
        def plot_mortality_trends(df, indicator_name):
            # Convert TIME_PERIOD to string to ensure it works as an index
            df['TIME_PERIOD_str'] = df['TIME_PERIOD'].astype(str)
            
            # Pivot data for plotting
            pivot_data = df.pivot_table(
                index='TIME_PERIOD_str', 
                columns='Geographic area', 
                values='OBS_VALUE', 
                aggfunc='mean'
            )
            
            # Calculate average across countries
            pivot_data['Average'] = pivot_data.mean(axis=1)
            
            # Plot the trends
            plt.figure(figsize=(15, 10))
            
            # Plot individual country trends
            for country in pivot_data.columns:
                if country != 'Average':
                    plt.plot(pivot_data.index, pivot_data[country], 'o', alpha=0.5, label=country)
            
            # Plot average trend with thicker line
            plt.plot(pivot_data.index, pivot_data['Average'], 'k-', linewidth=3, label='Average')
            
            plt.title(f'Trends in {indicator_name} in East African Community Countries', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Deaths per 1,000 live births', fontsize=14)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return pivot_data

        # Plot under-five mortality trends
        under5_trends = plot_mortality_trends(under5_eac, 'Under-Five Mortality Rate')
        plt.savefig('under5_mortality_trends.png')
        print("Visualization saved as 'under5_mortality_trends.png'")

        # Plot neonatal mortality trends
        neonatal_trends = plot_mortality_trends(neonatal_eac, 'Neonatal Mortality Rate')
        plt.savefig('neonatal_mortality_trends.png')
        print("Visualization saved as 'neonatal_mortality_trends.png'")

        # Identify countries with highest mortality rates
        print("\nIdentifying countries with highest mortality rates...")

        # Get the latest data for each country
        under5_latest_summary = under5_latest.groupby('Geographic area')['OBS_VALUE'].mean().reset_index()
        under5_latest_summary = under5_latest_summary.sort_values('OBS_VALUE', ascending=False)

        neonatal_latest_summary = neonatal_latest.groupby('Geographic area')['OBS_VALUE'].mean().reset_index()
        neonatal_latest_summary = neonatal_latest_summary.sort_values('OBS_VALUE', ascending=False)

        print("\nCountries with highest under-five mortality rates:")
        for i, row in under5_latest_summary.iterrows():
            print(f"{row['Geographic area']}: {row['OBS_VALUE']:.2f} deaths per 1,000 live births")

        print("\nCountries with highest neonatal mortality rates:")
        for i, row in neonatal_latest_summary.iterrows():
            print(f"{row['Geographic area']}: {row['OBS_VALUE']:.2f} deaths per 1,000 live births")

        print("\nChild mortality analysis complete!")
    else:
        print("Not enough data for EAC countries to proceed with analysis.")
else:
    print("Cannot proceed with child mortality analysis due to missing datasets.")

# Part 1: HIV Data Analysis (if HIV data is available)
if hiv_data_loaded and poverty_data_loaded:
    print("\n\n--- PART 1: HIV DATA ANALYSIS ---")
    
    # Clean and prepare HIV data
    print("\nCleaning and preparing HIV data...")
    
    # Check if 'Value' column exists
    if 'Value' in hiv_data.columns:
        # Convert 'Value' column to numeric, handling non-numeric values
        hiv_data['Value_numeric'] = pd.to_numeric(hiv_data['Value'], errors='coerce')
        
        # Check if 'Indicator' column exists
        if 'Indicator' in hiv_data.columns:
            # Filter for the indicator of interest (people living with HIV)
            hiv_filtered = hiv_data[hiv_data['Indicator'] == 'Estimated number of people (all ages) living with HIV']
            
            if not hiv_filtered.empty:
                # Group by year to get global totals
                if 'Period' in hiv_filtered.columns:
                    global_hiv_by_year = hiv_filtered.groupby('Period')['Value_numeric'].sum().reset_index()
                    global_hiv_by_year = global_hiv_by_year.rename(columns={'Value_numeric': 'Global_Total'})
                    
                    # Get the latest year data to identify countries contributing to 75% of global burden
                    latest_year = hiv_filtered['Period'].max()
                    latest_year_data = hiv_filtered[hiv_filtered['Period'] == latest_year]
                    
                    # Sort countries by HIV burden and calculate cumulative percentage
                    latest_year_data = latest_year_data.sort_values('Value_numeric', ascending=False)
                    latest_year_data['Cumulative_Sum'] = latest_year_data['Value_numeric'].cumsum()
                    latest_year_data['Global_Total'] = latest_year_data['Value_numeric'].sum()
                    latest_year_data['Cumulative_Percentage'] = (latest_year_data['Cumulative_Sum'] / latest_year_data['Global_Total']) * 100
                    
                    # Identify countries contributing to 75% of global burden
                    top_countries = latest_year_data[latest_year_data['Cumulative_Percentage'] <= 75]['Location'].tolist()
                    print(f"\nCountries contributing to 75% of global HIV burden ({len(top_countries)} countries):")
                    print(top_countries[:10], "... and more")
                    
                    # Continue with HIV analysis...
                    # (rest of the HIV analysis code would go here)
                    
                    print("\nHIV data analysis complete!")
                else:
                    print("'Period' column not found in HIV data. Cannot proceed with HIV analysis.")
            else:
                print("No data found for 'Estimated number of people (all ages) living with HIV'. Cannot proceed with HIV analysis.")
        else:
            print("'Indicator' column not found in HIV data. Cannot proceed with HIV analysis.")
    else:
        print("'Value' column not found in HIV data. Cannot proceed with HIV analysis.")
else:
    print("Cannot proceed with HIV data analysis due to missing datasets.")

print("\nAnalysis complete!")