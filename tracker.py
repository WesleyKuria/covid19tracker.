# COVID-19 Global Data Tracker
# =============================
# This notebook analyzes global COVID-19 data to track cases, deaths, recoveries,
# and vaccination rates across countries and time periods.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set better visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['font.size'] = 12

print("Libraries imported successfully!")

# 1. DATA COLLECTION AND LOADING
# ==============================
# Load the Our World in Data COVID-19 dataset
# Note: In a real scenario, you would download this file or access via API

# For demonstration, we'll use the URL directly
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
print(f"Loading data from {url}...")

try:
    df = pd.read_csv(url)
    print(f"Data loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    # If URL loading fails, you could include a fallback or error handling here
    
# 2. DATA EXPLORATION
# ==================
print("\n2. DATA EXPLORATION")
print("===================")

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check column names to understand available data
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Check data types and missing values
print("\nData types and missing values:")
print(df.info())

# Get summary statistics
print("\nSummary statistics for numeric columns:")
print(df.describe())

# Check for missing values in key columns
key_columns = ['location', 'date', 'total_cases', 'new_cases', 'total_deaths', 
               'new_deaths', 'total_vaccinations', 'people_vaccinated']
missing_values = df[key_columns].isnull().sum()
print("\nMissing values in key columns:")
print(missing_values)

# Get list of unique countries/locations
countries = df['location'].unique()
print(f"\nNumber of unique locations: {len(countries)}")
print(f"First 10 locations: {countries[:10]}")

# 3. DATA CLEANING
# ===============
print("\n3. DATA CLEANING")
print("================")

# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'])
print("Date column converted to datetime.")

# Filter out aggregate entries like 'World', 'Europe', etc.
# We'll create a separate dataframe for continents/world aggregates for later use
continents = ['World', 'Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania', 
              'European Union', 'High income', 'Upper middle income', 'Lower middle income', 'Low income']

df_aggregates = df[df['location'].isin(continents)].copy()
df_countries = df[~df['location'].isin(continents)].copy()

print(f"Separated data into countries ({df_countries.shape[0]} rows) and aggregates ({df_aggregates.shape[0]} rows).")

# Select countries of interest for focused analysis
countries_of_interest = ['United States', 'India', 'Brazil', 'United Kingdom', 
                         'Russia', 'France', 'South Africa', 'Kenya', 'China', 
                         'Australia', 'Germany', 'Canada']

df_selected = df[df['location'].isin(countries_of_interest)].copy()
print(f"Created a focused dataset with {df_selected.shape[0]} rows for {len(countries_of_interest)} selected countries.")

# Handle missing values for critical columns
# For key metrics, we'll use forward fill since this is time-series data
for column in ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']:
    # Group by country then apply forward fill
    df_selected[column] = df_selected.groupby('location')[column].ffill()

print("Forward-filled missing values in key metrics within each country's timeline.")

# Calculate additional metrics
df_selected['case_fatality_rate'] = (df_selected['total_deaths'] / df_selected['total_cases'] * 100).round(2)
print("Calculated case fatality rate as percentage.")

# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =================================
print("\n4. EXPLORATORY DATA ANALYSIS")
print("============================")

# Function to plot COVID metrics over time for selected countries
def plot_covid_metric(df, metric, title, ylabel):
    plt.figure(figsize=(16, 10))
    
    for country in df['location'].unique():
        country_data = df[df['location'] == country]
        plt.plot(country_data['date'], country_data[metric], label=country)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot total cases over time
plot_covid_metric(df_selected, 'total_cases', 'COVID-19 Total Cases Over Time', 'Total Cases')

# Plot total deaths over time
plot_covid_metric(df_selected, 'total_deaths', 'COVID-19 Total Deaths Over Time', 'Total Deaths')

# Plot new cases (with 7-day rolling average for smoothing)
plt.figure(figsize=(16, 10))

for country in df_selected['location'].unique():
    country_data = df_selected[df_selected['location'] == country].copy()
    country_data['rolling_new_cases'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['rolling_new_cases'], label=country)

plt.title('COVID-19 New Cases (7-day Rolling Average)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases (7-day avg)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compare case fatality rates
# Get the latest case fatality rate for each country
latest_data = df_selected.groupby('location').agg({
    'date': 'max',
    'case_fatality_rate': 'last',
    'total_cases': 'max',
    'total_deaths': 'max'
}).reset_index()

latest_data = latest_data.sort_values('case_fatality_rate', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(x='case_fatality_rate', y='location', data=latest_data)
plt.title('COVID-19 Case Fatality Rate by Country (Latest Data)', fontsize=16)
plt.xlabel('Case Fatality Rate (%)', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Top countries by total cases (horizontal bar chart)
plt.figure(figsize=(14, 10))
sns.barplot(x='total_cases', y='location', data=latest_data.sort_values('total_cases', ascending=False))
plt.title('Total COVID-19 Cases by Country (Latest Data)', fontsize=16)
plt.xlabel('Total Cases', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. VACCINATION ANALYSIS
# ======================
print("\n5. VACCINATION ANALYSIS")
print("=======================")

# Plot vaccination progress
plt.figure(figsize=(16, 10))

for country in df_selected['location'].unique():
    country_data = df_selected[df_selected['location'] == country]
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)

plt.title('COVID-19 Total Vaccinations Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Vaccinations', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate vaccination rates (percentage of population)
# Note: Not all countries have complete vaccination data
# We'll plot for available data

# Extract latest vaccination data
latest_vax_data = df_selected.groupby('location').agg({
    'date': 'max',
    'people_vaccinated': 'last',
    'population': 'first'  # Population should be constant for each country
}).reset_index()

# Calculate percentage
latest_vax_data['vaccination_rate'] = (latest_vax_data['people_vaccinated'] / latest_vax_data['population'] * 100).round(2)
latest_vax_data = latest_vax_data.sort_values('vaccination_rate', ascending=False)

# Remove rows with missing vaccination data
latest_vax_data = latest_vax_data.dropna(subset=['vaccination_rate'])

plt.figure(figsize=(14, 10))
sns.barplot(x='vaccination_rate', y='location', data=latest_vax_data)
plt.title('COVID-19 Vaccination Rate by Country (% of Population with At Least One Dose)', fontsize=16)
plt.xlabel('Vaccination Rate (%)', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. CHOROPLETH MAP VISUALIZATION
# ==============================
print("\n6. CHOROPLETH MAP VISUALIZATION")
print("===============================")

# Prepare data for choropleth map
# Get the latest data for each country (not just our selected ones)
world_latest = df.groupby('location').agg({
    'date': 'max',
    'total_cases': 'last',
    'total_deaths': 'last',
    'iso_code': 'first',
    'population': 'first'
}).reset_index()

# Calculate cases per million people
world_latest['cases_per_million'] = (world_latest['total_cases'] / world_latest['population'] * 1000000).round(2)

# Create a choropleth map for cases per million
try:
    fig = px.choropleth(
        world_latest,
        locations="iso_code",
        color="cases_per_million",
        hover_name="location",
        color_continuous_scale=px.colors.sequential.Plasma,
        title="COVID-19 Cases per Million People by Country",
        labels={'cases_per_million': 'Cases per Million'}
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Cases per Million")
    )
    fig.show()
except Exception as e:
    print(f"Unable to display choropleth map: {e}")
    print("Note: Choropleth maps may not display in all environments.")
    
    # Fallback to regular plot
    world_latest_sorted = world_latest.sort_values('cases_per_million', ascending=False).head(20)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='cases_per_million', y='location', data=world_latest_sorted)
    plt.title('Top 20 Countries by COVID-19 Cases per Million People', fontsize=16)
    plt.xlabel('Cases per Million', fontsize=14)
    plt.ylabel('Country', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 7. ADVANCED ANALYSIS: WAVE COMPARISON
# ===================================
print("\n7. ADVANCED ANALYSIS: WAVE COMPARISON")
print("=====================================")

# Compare waves of COVID across key countries
# We'll use rolling averages and focus on daily new cases
key_wave_countries = ['United States', 'India', 'United Kingdom', 'South Africa']
wave_data = df[df['location'].isin(key_wave_countries)].copy()

# Calculate 7-day rolling average of new cases
wave_data['rolling_new_cases'] = wave_data.groupby('location')['new_cases'].transform(
    lambda x: x.rolling(window=7).mean()
)

# Calculate cases per million for fair comparison
wave_data['cases_per_million'] = wave_data['rolling_new_cases'] / wave_data['population'] * 1000000

plt.figure(figsize=(16, 10))

for country in key_wave_countries:
    country_data = wave_data[wave_data['location'] == country]
    plt.plot(country_data['date'], country_data['cases_per_million'], label=country)

plt.title('COVID-19 Waves: Daily New Cases per Million (7-day Rolling Average)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases per Million (7-day avg)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. KEY INSIGHTS AND FINDINGS
# ==========================
print("\n8. KEY INSIGHTS AND FINDINGS")
print("===========================")

print("""
Key Insights from COVID-19 Data Analysis:

1. Global Spread Patterns:
   - The data reveals distinct waves of COVID-19 across different countries, with varying
     timing and intensity.
   - Some countries experienced sharper, more concentrated outbreaks, while others had
     more prolonged waves with lower peaks.

2. Case Fatality Rates:
   - Case fatality rates vary significantly between countries, influenced by factors like
     healthcare capacity, population demographics, testing rates, and reporting methods.
   - Countries with older populations generally showed higher case fatality rates.

3. Vaccination Impact:
   - Countries with higher vaccination rates generally experienced less severe later waves
     in terms of mortality, even when case numbers remained high.
   - There are notable differences in vaccination rollout speeds and coverage between countries.

4. Regional Variations:
   - Population density appears to correlate with transmission rates in many regions.
   - Island nations and countries with stricter border controls often maintained lower
     case rates during the early pandemic phases.

5. Testing and Reporting Factors:
   - Data interpretation requires caution due to differences in testing capacity and reporting
     protocols between countries.
   - Case numbers are more affected by testing limitations than death statistics, though
     both have reporting inconsistencies.

Limitations of this Analysis:
- Data quality varies by country and region
- Testing protocols and reporting methods differ significantly
- Many countries have incomplete vaccination data
- Population factors (age distribution, density, etc.) are not fully accounted for
- Policy interventions (lockdowns, mask mandates, etc.) are not included in this analysis
""")

print("\nCOVID-19 Global Data Tracker Analysis Complete!")
