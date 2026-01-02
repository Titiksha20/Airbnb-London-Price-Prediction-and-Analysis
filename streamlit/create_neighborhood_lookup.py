"""
Create Neighborhood Lookup File
Extracts neighborhood statistics for Streamlit app
"""

import pandas as pd
import numpy as np

print('Creating Neighborhood Lookup File')

# Load original cleaned data (has neighborhood names)
try:
    df = pd.read_csv('final_data/listings_clean_FINAL.csv')
    print(f'Loaded: {len(df):,} rows')
except:
    print('ERROR: listings_clean_FINAL.csv not found')
    print('This file should contain the cleaned data with neighborhood names')
    exit(1)

# Check for neighborhood column
if 'neighbourhood_cleansed' in df.columns:
    nbhd_col = 'neighbourhood_cleansed'
elif 'neighborhood' in df.columns:
    nbhd_col = 'neighborhood'
else:
    print('ERROR: No neighborhood column found')
    print('Available columns:', df.columns.tolist()[:20])
    exit(1)

print(f'Using column: {nbhd_col}')
print(f'Unique neighborhoods: {df[nbhd_col].nunique()}')
print('')

# Calculate neighborhood statistics
neighborhood_stats = df.groupby(nbhd_col).agg({
    'price_clean': ['mean', 'median', 'std', 'count']
}).reset_index()

neighborhood_stats.columns = ['neighborhood', 'price_mean', 'price_median', 'price_std', 'listing_count']

# Sort by price median
neighborhood_stats = neighborhood_stats.sort_values('price_median', ascending=False)

# Display top and bottom neighborhoods
print('Top 10 Most Expensive Neighborhoods:')
print(neighborhood_stats.head(10)[['neighborhood', 'price_median', 'listing_count']].to_string(index=False))
print('')

print('Top 10 Cheapest Neighborhoods:')
print(neighborhood_stats.tail(10)[['neighborhood', 'price_median', 'listing_count']].to_string(index=False))
print('')

# Save
neighborhood_stats.to_csv('final_data/neighborhood_lookup.csv', index=False)
print('Saved: neighborhood_lookup.csv')
print('')

# Create summary
print('SUMMARY:')
print(f'Total neighborhoods: {len(neighborhood_stats)}')
print(f'Price range: £{neighborhood_stats["price_median"].min():.0f} - £{neighborhood_stats["price_median"].max():.0f}')
print(f'Average median price: £{neighborhood_stats["price_median"].mean():.0f}')
print('')
print('Use this file in Streamlit app for neighborhood dropdown!')
