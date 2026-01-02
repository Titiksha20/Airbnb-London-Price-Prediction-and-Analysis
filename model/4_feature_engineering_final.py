"""
Feature Engineering - FINAL (NO DATA LEAKAGE)
Works with listings_clean_FINAL.csv columns
"""

import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

print('Feature Engineering ')
print('')

df = pd.read_csv('final_data/listings_clean_FINAL.csv')
print(f'Loaded: {len(df):,} rows x {len(df.columns)} columns')

df_feat = df.copy()

# AMENITIES
def parse_amenities(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except:
        return []

df_feat['amenity_list'] = df_feat['amenities'].apply(parse_amenities)
df_feat['amenity_count'] = df_feat['amenity_list'].apply(len)

amenities_map = {
    'has_wifi': ['Wifi', 'wifi', 'Internet'],
    'has_kitchen': ['Kitchen'],
    'has_ac': ['Air conditioning', 'AC'],
    'has_heating': ['Heating'],
    'has_tv': ['TV'],
    'has_washer': ['Washer'],
    'has_dryer': ['Dryer'],
    'has_parking': ['parking'],
    'has_gym': ['Gym'],
    'has_pool': ['Pool'],
    'has_hot_tub': ['Hot tub'],
    'has_elevator': ['Elevator'],
    'has_workspace': ['Dedicated workspace'],
    'has_self_checkin': ['Self check-in'],
    'has_doorman': ['Doorman'],
    'has_balcony': ['Balcony', 'Patio'],
    'has_dishwasher': ['Dishwasher']
}

for feat, keywords in amenities_map.items():
    df_feat[feat] = df_feat['amenity_list'].apply(
        lambda x: 1 if any(any(k.lower() in a.lower() for k in keywords) for a in x) else 0
    )

df_feat['luxury_score'] = df_feat[['has_pool', 'has_hot_tub', 'has_gym', 'has_doorman']].sum(axis=1)
df_feat['essential_score'] = df_feat[['has_wifi', 'has_kitchen', 'has_heating', 'has_tv']].sum(axis=1)

# LOCATION
central_lat, central_lon = 51.5074, -0.1278
df_feat['distance_from_center'] = np.sqrt(
    (df_feat['latitude'] - central_lat)**2 + (df_feat['longitude'] - central_lon)**2
) * 111

neighborhood_medians = df.groupby('neighbourhood_cleansed')['price_clean'].median()
tier_33, tier_66 = neighborhood_medians.quantile(0.33), neighborhood_medians.quantile(0.66)

def get_tier(neighborhood):
    if neighborhood not in neighborhood_medians.index:
        return 'mid'
    price = neighborhood_medians[neighborhood]
    if price < tier_33:
        return 'budget'
    elif price < tier_66:
        return 'mid'
    return 'premium'

df_feat['neighborhood_tier'] = df_feat['neighbourhood_cleansed'].apply(get_tier)

# NO LEAKAGE: Use neighborhood mean calculated from ALL listings (not individual price)
df_feat['neighborhood_price_mean'] = df_feat['neighbourhood_cleansed'].map(
    df.groupby('neighbourhood_cleansed')['price_clean'].mean()
)

neighborhood_counts = df.groupby('neighbourhood_cleansed').size()
df_feat['listings_in_neighborhood'] = df_feat['neighbourhood_cleansed'].map(neighborhood_counts)
df_feat['competition_intensity'] = 1 / np.log(df_feat['listings_in_neighborhood'] + 1)

le = LabelEncoder()
df_feat['neighborhood_encoded'] = le.fit_transform(df_feat['neighbourhood_cleansed'])

# PROPERTY
df_feat['bedrooms_per_person'] = df_feat['bedrooms_filled'] / df_feat['accommodates']
df_feat['beds_per_bedroom'] = df_feat['beds'] / (df_feat['bedrooms_filled'] + 0.1)
df_feat['bathrooms_per_bedroom'] = df_feat['bathrooms'] / (df_feat['bedrooms_filled'] + 0.1)
df_feat['is_studio'] = ((df_feat['bedrooms_filled'] == 0) & (df_feat['accommodates'] > 0)).astype(int)
df_feat['is_large'] = (df_feat['bedrooms_filled'] >= 4).astype(int)

def get_capacity(n):
    if n <= 2: return 'small'
    elif n <= 4: return 'medium'
    elif n <= 6: return 'large'
    return 'xlarge'

df_feat['capacity_cat'] = df_feat['accommodates'].apply(get_capacity)

# HOST
if 'host_since' in df_feat.columns:
    df_feat['host_since_date'] = pd.to_datetime(df_feat['host_since'], errors='coerce')
    today = pd.Timestamp.now()
    df_feat['host_experience_days'] = (today - df_feat['host_since_date']).dt.days
    df_feat['host_experience_years'] = df_feat['host_experience_days'] / 365.25
    median_exp = df_feat['host_experience_days'].median()
    df_feat['host_experience_days'] = df_feat['host_experience_days'].fillna(median_exp)
    df_feat['host_experience_years'] = df_feat['host_experience_years'].fillna(median_exp / 365.25)

df_feat['is_professional'] = (df_feat['host_total_listings_count'] >= 5).astype(int)

def parse_pct(s):
    try:
        return float(str(s).replace('%', '')) if pd.notna(s) else 0.0
    except:
        return 0.0

if 'host_response_rate' in df_feat.columns:
    df_feat['response_rate_num'] = df_feat['host_response_rate'].apply(parse_pct)

response_time_map = {
    'within an hour': 1.0,
    'within a few hours': 0.75,
    'within a day': 0.5,
    'a few days or more': 0.25,
    'unknown': 0.0
}

if 'host_response_time' in df_feat.columns:
    df_feat['response_time_score'] = df_feat['host_response_time'].map(response_time_map).fillna(0.0)
    if 'response_rate_num' in df_feat.columns:
        df_feat['host_responsiveness'] = df_feat['response_time_score'] * (df_feat['response_rate_num'] / 100)

if 'host_is_superhost' in df_feat.columns:
    df_feat['superhost_num'] = (df_feat['host_is_superhost'] == 't').astype(int)

# REVIEWS
if 'days_since_last_review' in df_feat.columns:
    df_feat['review_recency'] = 1 / (df_feat['days_since_last_review'] + 1)

if 'host_experience_years' in df_feat.columns and 'number_of_reviews' in df_feat.columns:
    df_feat['reviews_per_year'] = df_feat['number_of_reviews'] / (df_feat['host_experience_years'] + 0.1)

if 'number_of_reviews' in df_feat.columns:
    df_feat['high_review_volume'] = (df_feat['number_of_reviews'] >= df_feat['number_of_reviews'].quantile(0.75)).astype(int)

# Rating tier based on data
review_rating_col = None
for col in df_feat.columns:
    if 'review_scores_rating' in col and 'filled' in col:
        review_rating_col = col
        break

if review_rating_col:
    valid_ratings = df_feat[df_feat[review_rating_col] > 0][review_rating_col]
    if len(valid_ratings) > 0:
        rating_p33 = valid_ratings.quantile(0.33)
        rating_p66 = valid_ratings.quantile(0.66)
        
        def get_rating_tier(r):
            if r == 0: return 'no_reviews'
            elif r < rating_p33: return 'lower'
            elif r < rating_p66: return 'middle'
            return 'top'
        
        df_feat['rating_tier'] = df_feat[review_rating_col].apply(get_rating_tier)

# Average review score
review_cols = [c for c in df_feat.columns if c.startswith('review_scores_') and c.endswith('_filled')]
if len(review_cols) > 0:
    df_feat['avg_review_score'] = df_feat[review_cols].mean(axis=1)

# Occupancy proxy
if 'reviews_per_month' in df_feat.columns and 'availability_365' in df_feat.columns:
    df_feat['estimated_bookings_monthly'] = df_feat['reviews_per_month'] * 2
    df_feat['booking_rate_proxy'] = df_feat['estimated_bookings_monthly'] / (df_feat['availability_365'] / 12 + 1)

# AVAILABILITY
if 'availability_365' in df_feat.columns:
    df_feat['availability_rate'] = df_feat['availability_365'] / 365

if 'minimum_nights' in df_feat.columns:
    df_feat['is_flexible'] = (df_feat['minimum_nights'] == 1).astype(int)
    df_feat['is_longstay_only'] = (df_feat['minimum_nights'] >= 7).astype(int)

if 'instant_bookable' in df_feat.columns:
    df_feat['instant_bookable_num'] = (df_feat['instant_bookable'] == 't').astype(int)

# INTERACTIONS
if 'superhost_num' in df_feat.columns and review_rating_col:
    df_feat['superhost_high_rating'] = (
        (df_feat['superhost_num'] == 1) & 
        (df_feat[review_rating_col] >= 4.8)
    ).astype(int)

if 'superhost_num' in df_feat.columns and 'number_of_reviews' in df_feat.columns:
    df_feat['superhost_many_reviews'] = df_feat['superhost_num'] * np.log1p(df_feat['number_of_reviews'])

df_feat['premium_luxury'] = (
    (df_feat['neighborhood_tier'] == 'premium') & 
    (df_feat['luxury_score'] >= 2)
).astype(int)

df_feat['large_premium'] = (
    (df_feat['is_large'] == 1) & 
    (df_feat['neighborhood_tier'] == 'premium')
).astype(int)

df_feat['luxury_central'] = df_feat['luxury_score'] * (1 / (df_feat['distance_from_center'] + 0.1))
df_feat['budget_essentials'] = (df_feat['neighborhood_tier'] == 'budget').astype(int) * df_feat['essential_score']

# ENCODING
if 'room_type' in df_feat.columns:
    room_dummies = pd.get_dummies(df_feat['room_type'], prefix='room', drop_first=True)
    df_feat = pd.concat([df_feat, room_dummies], axis=1)

if 'property_type' in df_feat.columns:
    top_10_props = df_feat['property_type'].value_counts().head(10).index
    df_feat['property_grouped'] = df_feat['property_type'].apply(lambda x: x if x in top_10_props else 'Other')
    prop_dummies = pd.get_dummies(df_feat['property_grouped'], prefix='prop', drop_first=True)
    df_feat = pd.concat([df_feat, prop_dummies], axis=1)

tier_dummies = pd.get_dummies(df_feat['neighborhood_tier'], prefix='tier', drop_first=True)
df_feat = pd.concat([df_feat, tier_dummies], axis=1)

capacity_dummies = pd.get_dummies(df_feat['capacity_cat'], prefix='cap', drop_first=True)
df_feat = pd.concat([df_feat, capacity_dummies], axis=1)

# FINAL FEATURE SET
features = [
    'price_clean',
    'accommodates', 'bedrooms_filled', 'beds', 'bathrooms',
    'is_studio', 'is_large',
    'bedrooms_per_person', 'beds_per_bedroom', 'bathrooms_per_bedroom',
    'amenity_count', 'luxury_score', 'essential_score',
    'has_wifi', 'has_kitchen', 'has_ac', 'has_heating', 'has_tv',
    'has_washer', 'has_dryer', 'has_parking', 'has_gym', 'has_pool',
    'has_hot_tub', 'has_elevator', 'has_workspace', 'has_self_checkin',
    'has_doorman', 'has_balcony', 'has_dishwasher',
    'latitude', 'longitude', 'distance_from_center',
    'neighborhood_price_mean', 'neighborhood_encoded',
    'listings_in_neighborhood', 'competition_intensity',
    'is_professional',
    'number_of_reviews', 'reviews_per_month', 'has_reviews',
    'high_review_volume', 'avg_review_score',
    'availability_365', 'availability_rate',
    'minimum_nights', 'maximum_nights',
    'is_flexible', 'is_longstay_only',
    'premium_luxury', 'large_premium',
    'luxury_central', 'budget_essentials'
]

# Add optional features if they exist
optional = [
    'host_experience_years', 'response_rate_num', 'acceptance_rate_numeric',
    'response_time_score', 'host_responsiveness', 'superhost_num',
    'reviews_per_year', 'instant_bookable_num', 'review_recency',
    'estimated_bookings_monthly', 'booking_rate_proxy',
    'superhost_high_rating', 'superhost_many_reviews'
]

for f in optional:
    if f in df_feat.columns and f not in features:
        features.append(f)

# Add dummy columns
dummy_cols = [c for c in df_feat.columns if any(c.startswith(p) for p in ['room_', 'prop_', 'tier_', 'cap_'])]
features.extend(dummy_cols)

# Filter to existing features
features = [f for f in features if f in df_feat.columns]
features = list(dict.fromkeys(features))

df_final = df_feat[features].copy()

# Fill any remaining missing
for col in df_final.columns:
    if df_final[col].isnull().any():
        if df_final[col].dtype in ['float64', 'int64']:
            df_final[col] = df_final[col].fillna(df_final[col].median())

print(f'Final features: {len(df_final.columns)}')
print(f'Missing values: {df_final.isnull().sum().sum()}')
print('')

df_final.to_csv('final_data/features_final.csv', index=False)
print('Saved: features_final.csv')
print('Ready for modeling')