"""
Exploratory Data Analysis - Inside Airbnb London
Final Cleaned Dataset Analysis

Purpose: Understand patterns, relationships, and insights before feature engineering
Dataset: listings_clean_FINAL.csv (60,796 listings)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Plotting configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print('EDA - Inside Airbnb London (Final Cleaned Data)')
print('Dataset: listings_clean_FINAL.csv')
print('')

# Load data
print('Loading data...')
df = pd.read_csv('final_data/listings_clean_FINAL.csv')

print(f'Dataset loaded: {len(df):,} rows x {len(df.columns)} columns')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
print('')

# Basic info
print('DATASET OVERVIEW')
print('')
print(f'Shape: {df.shape}')
print(f'Date range: {df.columns[0]} to {df.columns[-1]}')
print('')

# Data completeness
missing_total = df.isnull().sum().sum()
completeness = (1 - missing_total / df.size) * 100
print(f'Data completeness: {completeness:.2f}%')
print(f'Total missing values: {missing_total:,}')
print('')

# Column types
print('Column data types:')
print(df.dtypes.value_counts())
print('')

# First few rows
print('Sample data (first 5 rows):')
print(df.head())
print('')

# SECTION 1: TARGET VARIABLE ANALYSIS
print('')
print('SECTION 1: TARGET VARIABLE - PRICE')
print('')

print('Price statistics:')
print(df['price_clean'].describe())
print('')

# Price distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original distribution
axes[0].hist(df['price_clean'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df['price_clean'].median(), color='red', linestyle='--', linewidth=2, 
                label=f"Median: £{df['price_clean'].median():.0f}")
axes[0].axvline(df['price_clean'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f"Mean: £{df['price_clean'].mean():.0f}")
axes[0].set_xlabel('Price (GBP)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Price Distribution', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Log-transformed
axes[1].hist(np.log1p(df['price_clean']), bins=100, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Log(Price + 1)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Price Distribution (Log Scale)', fontweight='bold')
axes[1].grid(alpha=0.3)

# Box plot
bp = axes[2].boxplot(df['price_clean'], vert=True, patch_artist=True,
                      boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[2].set_ylabel('Price (GBP)', fontweight='bold')
axes[2].set_title('Price Box Plot', fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Skewness
skewness_original = stats.skew(df['price_clean'])
skewness_log = stats.skew(np.log1p(df['price_clean']))
print(f'Skewness (original): {skewness_original:.3f}')
print(f'Skewness (log-transformed): {skewness_log:.3f}')
print('')

# SECTION 2: PROPERTY CHARACTERISTICS
print('')
print('SECTION 2: PROPERTY CHARACTERISTICS')
print('')

# Room type analysis
print('Price by Room Type:')
room_type_stats = df.groupby('room_type')['price_clean'].agg(['count', 'mean', 'median', 'std']).round(2)
room_type_stats['percentage'] = (room_type_stats['count'] / len(df) * 100).round(1)
print(room_type_stats.sort_values('mean', ascending=False))
print('')

# Visualize room type
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Count
room_counts = df['room_type'].value_counts()
axes[0].bar(room_counts.index, room_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Room Type', fontweight='bold')
axes[0].set_ylabel('Number of Listings', fontweight='bold')
axes[0].set_title('Listings Count by Room Type', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

for i, (idx, val) in enumerate(room_counts.items()):
    pct = val / len(df) * 100
    axes[0].text(i, val, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# Price distribution by room type
room_types_list = df['room_type'].unique()
price_data = [df[df['room_type'] == rt]['price_clean'].values for rt in room_types_list]

vp = axes[1].violinplot(price_data, positions=range(len(room_types_list)), showmeans=True, showmedians=True)
axes[1].set_xticks(range(len(room_types_list)))
axes[1].set_xticklabels(room_types_list, rotation=45, ha='right')
axes[1].set_ylabel('Price (GBP)', fontweight='bold')
axes[1].set_title('Price Distribution by Room Type', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_room_type_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Accommodates vs Price
print('Price by Guest Capacity (Accommodates):')
accommodates_stats = df.groupby('accommodates')['price_clean'].agg(['count', 'mean', 'median']).round(2)
accommodates_stats['percentage'] = (accommodates_stats['count'] / len(df) * 100).round(1)
print(accommodates_stats.head(10))
print('')

# Visualize accommodates
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribution
accom_counts = df['accommodates'].value_counts().sort_index().head(12)
axes[0].bar(accom_counts.index, accom_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Number of Guests', fontweight='bold')
axes[0].set_ylabel('Number of Listings', fontweight='bold')
axes[0].set_title('Distribution by Guest Capacity', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Price trend
accom_price = df.groupby('accommodates')['price_clean'].mean().head(12)
axes[1].scatter(accom_price.index, accom_price.values, s=150, color='coral', 
                edgecolor='black', alpha=0.7, zorder=3)
axes[1].plot(accom_price.index, accom_price.values, color='red', linestyle='--', 
             linewidth=2, alpha=0.5, zorder=2)

# Trend line
z = np.polyfit(accom_price.index, accom_price.values, 1)
p = np.poly1d(z)
axes[1].plot(accom_price.index, p(accom_price.index), "g-", linewidth=2, 
             label=f'Trend: £{z[0]:.1f} per guest', zorder=1)

axes[1].set_xlabel('Number of Guests', fontweight='bold')
axes[1].set_ylabel('Average Price (GBP)', fontweight='bold')
axes[1].set_title('Price vs Guest Capacity', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_accommodates_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Price per additional guest: £{z[0]:.2f}')
print('')

# Bedrooms analysis
print('Price by Number of Bedrooms:')
bedroom_stats = df.groupby('bedrooms_filled')['price_clean'].agg(['count', 'mean', 'median']).round(2)
print(bedroom_stats.head(8))
print('')

# Correlation analysis
print('Correlation with Price (Top Property Features):')
property_features = ['accommodates', 'bedrooms_filled', 'beds', 'bathrooms']
property_features_existing = [f for f in property_features if f in df.columns]

for feature in property_features_existing:
    corr = df[[feature, 'price_clean']].corr().iloc[0, 1]
    print(f'  {feature:<20s}: {corr:.3f}')
print('')

# SECTION 3: HOST FEATURES
print('')
print('SECTION 3: HOST FEATURES')
print('')

# Superhost analysis
print('Superhost Analysis:')
superhost_stats = df.groupby('host_is_superhost')['price_clean'].agg(['count', 'mean', 'median']).round(2)
superhost_stats['percentage'] = (superhost_stats['count'] / len(df) * 100).round(1)
print(superhost_stats)

if 't' in superhost_stats.index and 'f' in superhost_stats.index:
    premium = ((superhost_stats.loc['t', 'mean'] / superhost_stats.loc['f', 'mean']) - 1) * 100
    absolute_diff = superhost_stats.loc['t', 'mean'] - superhost_stats.loc['f', 'mean']
    print(f'Superhost premium: {premium:.1f}%')
    print(f'Absolute difference: £{absolute_diff:.2f} per night')
print('')

# Visualize Superhost impact
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribution comparison
df[df['host_is_superhost'] == 'f']['price_clean'].hist(
    bins=50, alpha=0.6, label='Regular Host', color='coral', edgecolor='black', ax=axes[0]
)
df[df['host_is_superhost'] == 't']['price_clean'].hist(
    bins=50, alpha=0.6, label='Superhost', color='steelblue', edgecolor='black', ax=axes[0]
)
axes[0].set_xlabel('Price (GBP)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Price Distribution: Superhost vs Regular', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot comparison
superhost_data = [
    df[df['host_is_superhost'] == 'f']['price_clean'].values,
    df[df['host_is_superhost'] == 't']['price_clean'].values
]
bp = axes[1].boxplot(superhost_data, labels=['Regular Host', 'Superhost'], patch_artist=True)
bp['boxes'][0].set_facecolor('coral')
bp['boxes'][1].set_facecolor('steelblue')
for patch in bp['boxes']:
    patch.set_alpha(0.7)

axes[1].set_ylabel('Price (GBP)', fontweight='bold')
axes[1].set_title(f'Price Comparison (Premium: {premium:.1f}%)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_superhost_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# SECTION 4: REVIEW FEATURES
print('')
print('SECTION 4: REVIEW FEATURES')
print('')

# Reviews analysis
print('Review Statistics:')
print(df[['number_of_reviews', 'reviews_per_month', 'review_scores_rating_filled']].describe())
print('')

# Has reviews breakdown
print('Listings by Review Status:')
print(df['has_reviews'].value_counts())
print('')

review_status = df.groupby('has_reviews')['price_clean'].agg(['count', 'mean', 'median']).round(2)
print(review_status)
print('')

# Reviews vs price visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Number of reviews distribution
axes[0, 0].hist(df['number_of_reviews'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Reviews', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Distribution of Review Count', fontweight='bold')
axes[0, 0].set_xlim(0, df['number_of_reviews'].quantile(0.95))
axes[0, 0].grid(alpha=0.3)

# Rating distribution
axes[0, 1].hist(df['review_scores_rating_filled'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Rating Score', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Distribution of Review Ratings', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Reviews vs price
axes[1, 0].scatter(df['number_of_reviews'], df['price_clean'], alpha=0.1, s=10, color='steelblue')
axes[1, 0].set_xlabel('Number of Reviews', fontweight='bold')
axes[1, 0].set_ylabel('Price (GBP)', fontweight='bold')
axes[1, 0].set_title('Price vs Number of Reviews', fontweight='bold')
axes[1, 0].set_xlim(0, df['number_of_reviews'].quantile(0.95))
axes[1, 0].set_ylim(0, df['price_clean'].quantile(0.95))
axes[1, 0].grid(alpha=0.3)

# Rating vs price
axes[1, 1].scatter(df['review_scores_rating_filled'], df['price_clean'], alpha=0.1, s=10, color='coral')
axes[1, 1].set_xlabel('Review Rating', fontweight='bold')
axes[1, 1].set_ylabel('Price (GBP)', fontweight='bold')
axes[1, 1].set_title('Price vs Review Rating', fontweight='bold')
axes[1, 1].set_ylim(0, df['price_clean'].quantile(0.95))
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_review_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlations
review_corr = df[['number_of_reviews', 'price_clean']].corr().iloc[0, 1]
rating_corr = df[['review_scores_rating_filled', 'price_clean']].corr().iloc[0, 1]

print(f'Correlation - Reviews vs Price: {review_corr:.3f}')
print(f'Correlation - Rating vs Price: {rating_corr:.3f}')
print('')

# SECTION 5: GEOGRAPHIC ANALYSIS
print('')
print('SECTION 5: GEOGRAPHIC ANALYSIS')
print('')

# Neighborhood analysis
print('Top 15 Most Expensive Neighborhoods (by mean price):')
neighborhood_stats = df.groupby('neighbourhood_cleansed')['price_clean'].agg(['count', 'mean', 'median']).round(2)
neighborhood_stats = neighborhood_stats[neighborhood_stats['count'] >= 100]  # Filter for reliability
neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=False)
print(neighborhood_stats.head(15))
print('')

# Visualize top neighborhoods
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Top 15 expensive
top_15_expensive = neighborhood_stats.head(15).sort_values('mean', ascending=True)
axes[0].barh(top_15_expensive.index, top_15_expensive['mean'], 
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Average Price (GBP)', fontweight='bold')
axes[0].set_ylabel('Neighborhood', fontweight='bold')
axes[0].set_title('Top 15 Most Expensive Neighborhoods', fontweight='bold', pad=20)
axes[0].grid(axis='x', alpha=0.3)

# Top 15 cheapest
top_15_cheap = neighborhood_stats.tail(15).sort_values('mean', ascending=True)
axes[1].barh(top_15_cheap.index, top_15_cheap['mean'],
             color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Average Price (GBP)', fontweight='bold')
axes[1].set_ylabel('Neighborhood', fontweight='bold')
axes[1].set_title('Top 15 Cheapest Neighborhoods', fontweight='bold', pad=20)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/eda_neighborhood_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Geographic scatter
print('Geographic Distribution:')
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df['longitude'], df['latitude'], 
                     c=df['price_clean'], cmap='RdYlGn_r', 
                     alpha=0.3, s=5, vmin=0, vmax=df['price_clean'].quantile(0.95))
plt.colorbar(scatter, label='Price (GBP)')
plt.xlabel('Longitude', fontweight='bold')
plt.ylabel('Latitude', fontweight='bold')
plt.title('Geographic Distribution of Listings (Color = Price)', fontweight='bold', pad=20)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/eda_geographic_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# SECTION 6: CORRELATION MATRIX
print('')
print('SECTION 6: CORRELATION ANALYSIS')
print('')

# Select key numeric features
numeric_features = [
    'price_clean', 'accommodates', 'bedrooms_filled', 'beds', 'bathrooms',
    'minimum_nights', 'maximum_nights', 'number_of_reviews', 
    'reviews_per_month', 'review_scores_rating_filled', 'availability_365',
    'days_since_last_review', 'has_reviews'
]

numeric_features_existing = [f for f in numeric_features if f in df.columns]

# Correlation matrix
corr_matrix = df[numeric_features_existing].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Key Features', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_plots/eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print('Top correlations with Price:')
price_corr = corr_matrix['price_clean'].sort_values(ascending=False)
for feature, corr in price_corr.items():
    if feature != 'price_clean':
        print(f'  {feature:<30s}: {corr:7.3f}')
print('')

# SECTION 7: KEY INSIGHTS SUMMARY
print('')
print('SECTION 7: KEY INSIGHTS SUMMARY')
print('')

insights = []

# Price insights
median_price = df['price_clean'].median()
mean_price = df['price_clean'].mean()
insights.append(f'Median price: £{median_price:.2f}, Mean price: £{mean_price:.2f}')

# Room type insights
most_common_room = df['room_type'].value_counts().index[0]
most_common_pct = df['room_type'].value_counts().values[0] / len(df) * 100
insights.append(f'Most common room type: {most_common_room} ({most_common_pct:.1f}%)')

# Accommodates insights
most_common_capacity = df['accommodates'].mode()[0]
price_per_guest = z[0]
insights.append(f'Most listings accommodate {most_common_capacity} guests')
insights.append(f'Each additional guest adds ~£{price_per_guest:.2f} to price')

# Superhost insights
if 't' in superhost_stats.index and 'f' in superhost_stats.index:
    insights.append(f'Superhost premium: {premium:.1f}% (£{absolute_diff:.2f} per night)')

# Review insights
pct_reviewed = (df['has_reviews'].sum() / len(df)) * 100
insights.append(f'Listings with reviews: {pct_reviewed:.1f}%')

# Neighborhood insights
most_expensive_neighborhood = neighborhood_stats.head(1).index[0]
most_expensive_price = neighborhood_stats.head(1)['mean'].values[0]
cheapest_neighborhood = neighborhood_stats.tail(1).index[0]
cheapest_price = neighborhood_stats.tail(1)['mean'].values[0]
insights.append(f'Most expensive neighborhood: {most_expensive_neighborhood} (£{most_expensive_price:.2f})')
insights.append(f'Cheapest neighborhood: {cheapest_neighborhood} (£{cheapest_price:.2f})')

# Correlation insights
strongest_corr_feature = price_corr[price_corr.index != 'price_clean'].index[0]
strongest_corr_value = price_corr[price_corr.index != 'price_clean'].values[0]
insights.append(f'Strongest price predictor: {strongest_corr_feature} (r={strongest_corr_value:.3f})')

print('Key Insights:')
for i, insight in enumerate(insights, 1):
    print(f'{i}. {insight}')
print('')

# Save insights to file
with open('eda_plots/eda_insights_summary.txt', 'w') as f:
    f.write('EDA Insights Summary - Inside Airbnb London\n')
    f.write('\n')
    for insight in insights:
        f.write(f'- {insight}\n')

print('Analysis complete. Visualizations saved as PNG files.')
print('Insights summary saved to: eda_insights_summary.txt')
print('')
print('Ready for feature engineering.')
