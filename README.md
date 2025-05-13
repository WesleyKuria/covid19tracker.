# covid19tracker
Key Components:

Data Collection & Loading

Loads COVID data directly from Our World in Data's CSV URL
Shows how to handle the dataset programmatically


Data Exploration

Examines dataset structure, columns, and key metrics
Identifies missing values and data quality issues


Data Cleaning

Converts date fields to proper datetime format
Separates country-level data from aggregate data (continents, world)
Handles missing values using forward fill (appropriate for time series)
Calculates derived metrics like case fatality rate


Exploratory Data Analysis

Creates time series visualizations for total cases and deaths
Uses 7-day rolling averages to smooth daily fluctuations
Creates bar charts to compare countries on key metrics


Vaccination Analysis

Tracks vaccination progress across countries
Calculates and visualizes vaccination rates as percentage of population


Choropleth Map Visualization

Creates a world map showing cases per million people
Includes a fallback visualization if the environment doesn't support interactive maps


Advanced Analysis: Wave Comparison

Normalizes data (cases per million) for fair country comparison
Visualizes different COVID waves across key countries


Key Insights

Provides meaningful narrative insights based on the analysis
Highlights limitations and considerations when interpreting the data



How to Use This Notebook:

Setup: The notebook is ready to run in any Jupyter environment with standard data science libraries (pandas, matplotlib, seaborn, plotly)
Data Source: It connects directly to Our World in Data's COVID dataset, so no manual download is needed
Execution: Run all cells in sequence to perform the complete analysis
Results: The notebook produces publication-quality visualizations and includes narrative insights
