"""
FINAL Data Cleaning Script 


Incorporates lessons learned:
1. Review scores: Use NaN + flags (not 0) to avoid penalizing new listings
2. Superhost: Fill with 'f' (agreed - safe assumption)
3. Last review: Create days_since + flag
4. Acceptance rate: Fill with median (not 0%)
"""

import pandas as pd
import numpy as np
from datetime import datetime

def final_improved_cleaning(filepath='raw_data/listings.csv.gz'):
    """
    Final improved cleaning pipeline
    """
    
    print("FINAL IMPROVED DATA CLEANING PIPELINE")


    # 1. LOAD DATA

    print("\n1. Loading data...")
    df = pd.read_csv(filepath, compression='gzip', low_memory=False)
    print(f"   Loaded: {len(df):,} rows × {len(df.columns)} columns")
    

    # 2. CLEAN PRICE

    print("\n2. Cleaning price column...")
    df['price_clean'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Remove missing prices
    df = df[df['price_clean'].notna()].copy()
    print(f"   Removed missing prices. Rows: {len(df):,}")
    
    # Filter outliers (1st-99th percentile)
    p1 = df['price_clean'].quantile(0.01)
    p99 = df['price_clean'].quantile(0.99)
    df = df[(df['price_clean'] >= p1) & (df['price_clean'] <= p99)].copy()
    print(f"   Filtered outliers (£{p1:.0f} - £{p99:.0f}). Rows: {len(df):,}")

    # 3. PROPERTY FEATURES

    print("\n3. Handling property features...")
    
    # Bedrooms
    mode_bedrooms = df['bedrooms'].mode()[0]
    df['bedrooms_filled'] = df['bedrooms'].fillna(mode_bedrooms)
    print(f"    Filled {df['bedrooms'].isnull().sum()} bedrooms with mode ({mode_bedrooms})")
    
    # Beds
    df['beds'] = df['beds'].fillna(df['bedrooms_filled'])
    print(f"    Filled beds with bedrooms")
    
    # Bathrooms
    median_bath = df['bathrooms'].median()
    df['bathrooms'] = df['bathrooms'].fillna(median_bath)
    print(f"    Filled bathrooms with median ({median_bath})")
    

    # 4. REVIEW FEATURES (IMPROVED STRATEGY!)

    print("\n4. Handling review features (IMPROVED)...")
    
    # Identify listings with no reviews
    no_reviews = df['number_of_reviews'] == 0
    num_no_reviews = no_reviews.sum()
    
    print(f"   Listings with 0 reviews: {num_no_reviews:,} ({num_no_reviews/len(df)*100:.1f}%)")
    
    # Create has_reviews flag FIRST
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    print(f"   Created 'has_reviews' flag (0/1)")
    
    # CRITICAL: Keep review scores as NaN for no-review listings
    # Do NOT fill with 0 (would penalize new listings!)
    # Do NOT fill with median (would be misleading!)
    # Strategy: Create separate "has_score" flags + filled values
    
    review_score_cols = [
        'review_scores_rating',
        'review_scores_accuracy',
        'review_scores_cleanliness',
        'review_scores_checkin',
        'review_scores_communication',
        'review_scores_location',
        'review_scores_value'
    ]
    
    for col in review_score_cols:
        if col in df.columns:
            # Create flag: does this listing have this score?
            flag_col = f'has_{col}'
            df[flag_col] = df[col].notna().astype(int)
            
            # For modeling: fill NaN with median of REVIEWED listings only
            median_score = df[df['has_reviews']==1][col].median()
            filled_col = f'{col}_filled'
            df[filled_col] = df[col].fillna(median_score)
            
            print(f"    Created {flag_col} + {filled_col} (median: {median_score:.2f})")
    
    # Reviews per month
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    print(f"    Filled reviews_per_month with 0")
    
    # Last review - create temporal feature
    if 'last_review' in df.columns:
        # Convert to datetime
        df['last_review_date'] = pd.to_datetime(df['last_review'], errors='coerce')
        
        # Calculate days since last review
        today = pd.Timestamp.now()
        df['days_since_last_review'] = (today - df['last_review_date']).dt.days
        
        # For listings never reviewed, set to high value (e.g., 9999)
        df.loc[df['last_review_date'].isna(), 'days_since_last_review'] = 9999
        
        # Create flag
        df['has_last_review'] = df['last_review_date'].notna().astype(int)
        
        print(f"    Created 'days_since_last_review' (9999 for never reviewed)")
        print(f"    Created 'has_last_review' flag")
    

    # 5. HOST FEATURES

    print("\n5. Handling host features...")
    
    # Superhost - fill with 'f' (agreed strategy)
    if 'host_is_superhost' in df.columns:
        missing = df['host_is_superhost'].isnull().sum()
        df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
        print(f"     Filled {missing:,} host_is_superhost with 'f'")
        print(f"     REASON: New hosts are not Superhosts, safe assumption")
    
    # Host response rate - create flag + fill with 0%
    if 'host_response_rate' in df.columns:
        missing = df['host_response_rate'].isnull().sum()
        df['host_response_missing'] = df['host_response_rate'].isnull().astype(int)
        df['host_response_rate'] = df['host_response_rate'].fillna('0%')
        print(f"    Created host_response_missing flag + filled {missing:,} with '0%'")
    
    # Host response time
    if 'host_response_time' in df.columns:
        missing = df['host_response_time'].isnull().sum()
        df['host_response_time'] = df['host_response_time'].fillna('unknown')
        print(f"    Filled {missing:,} host_response_time with 'unknown'")
    
    # Host acceptance rate - IMPROVED: fill with MEDIAN not 0%!
    if 'host_acceptance_rate' in df.columns:
        missing = df['host_acceptance_rate'].isnull().sum()
        
        # Extract numeric value from percentage string
        df['acceptance_rate_numeric'] = df['host_acceptance_rate'].str.replace('%', '').astype(float, errors='ignore')
        median_acceptance = df['acceptance_rate_numeric'].median()
        
        # Fill missing with median
        df['host_acceptance_missing'] = df['host_acceptance_rate'].isnull().astype(int)
        df.loc[df['host_acceptance_rate'].isna(), 'host_acceptance_rate'] = f'{median_acceptance:.0f}%'
        df.loc[df['acceptance_rate_numeric'].isna(), 'acceptance_rate_numeric'] = median_acceptance
        
        print(f"    Filled {missing:,} host_acceptance_rate with MEDIAN ({median_acceptance:.0f}%)")
        print(f"     REASON: 0% too pessimistic, median more realistic")
    
    # Other host features (tiny missing amount)
    if 'has_availability' in df.columns:
        df['has_availability'] = df['has_availability'].fillna('t')
    if 'bathrooms_text' in df.columns:
        df['bathrooms_text'] = df['bathrooms_text'].fillna('1 bath')
    

    # 6. DROP USELESS COLUMNS

    print("\n6. Dropping useless columns...")
    
    drop_cols = [
        'license', 'calendar_updated', 'neighbourhood_group_cleansed',
        'neighborhood_overview', 'host_about', 'description', 'name',
        'host_location', 'host_neighbourhood', 'neighbourhood',
        'first_review', 'last_scraped', 'calendar_last_scraped',
        'listing_url', 'scrape_id', 'picture_url', 'host_url',
        'host_thumbnail_url', 'host_picture_url', 'bedrooms', 'price',
        'host_verifications', 'source', 'last_review', 'last_review_date'
    ]
    
    drop_cols_existing = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols_existing)
    print(f"    Dropped {len(drop_cols_existing)} columns")
    
    # Also drop original review score columns (keep _filled and has_ versions)
    for col in review_score_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    print(f"    Dropped original review_scores_* (kept _filled + has_ versions)")
    

    # 7. FINAL CLEANUP

    print("\n7. Final cleanup...")
    
    # Drop rows with missing critical features
    critical_features = [
        'price_clean', 'accommodates', 'room_type', 'property_type',
        'neighbourhood_cleansed', 'latitude', 'longitude', 'instant_bookable'
    ]
    critical_existing = [f for f in critical_features if f in df.columns]
    
    rows_before = len(df)
    df = df.dropna(subset=critical_existing)
    rows_after = len(df)
    
    if rows_before - rows_after > 0:
        print(f"    Dropped {rows_before - rows_after:,} rows with missing critical features")
    

    # 8. SUMMARY

    print("CLEANING SUMMARY")
    
    print(f"\nFinal dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    missing_total = df.isnull().sum().sum()
    completeness = (1 - missing_total / df.size) * 100
    print(f"Data completeness: {completeness:.2f}%")
    
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    
    if len(remaining_missing) > 0:
        print(f"\nRemaining missing data ({len(remaining_missing)} columns):")
        for col, count in remaining_missing.sort_values(ascending=False).items():
            pct = count / len(df) * 100
            print(f"   {col:<40s} {count:>6,} ({pct:>5.1f}%)")
    else:
        print("\n No remaining missing data!")
    

    # 9. KEY IMPROVEMENTS SUMMARY

    print("\n")
    print("KEY IMPROVEMENTS IN THIS VERSION")

    
    improvements = [
        " Review scores: Created has_* flags + _filled columns (not 0!)",
        " Days since review: Created with 9999 for never reviewed",
        " Host acceptance rate: Filled with MEDIAN (not 0%)",
        " Superhost: Kept 'f' strategy (validated as correct)",
        " All decisions now have clear justification"
    ]
    
    for imp in improvements:
        print(f"  {imp}")
    
    print("\n" )
    
    return df



# MAIN EXECUTION


if __name__ == "__main__":
    # Run the improved pipeline
    df_clean = final_improved_cleaning('raw_data/listings.csv.gz')
    
    # Save
    output_file = 'final_data/listings_clean_FINAL.csv'
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n SAVED: {output_file}")
    print(f" Ready for feature engineering!")
    
    # Generate cleaning report
    with open('final_data/cleaning_report_FINAL.txt', 'w', encoding='utf-8') as f:
        
        f.write("FINAL DATA CLEANING REPORT\n")
       
        
        f.write("KEY DECISIONS:\n\n")
        
        f.write("1. REVIEW SCORES:\n")
        f.write("   - Created has_review_scores_* flags (0/1)\n")
        f.write("   - Created review_scores_*_filled with median\n")
        f.write("   - REASON: Avoid penalizing new listings with 0 scores\n\n")
        
        f.write("2. DAYS SINCE REVIEW:\n")
        f.write("   - Created days_since_last_review feature\n")
        f.write("   - Set to 9999 for never-reviewed listings\n")
        f.write("   - Created has_last_review flag\n\n")
        
        f.write("3. HOST ACCEPTANCE RATE:\n")
        f.write("   - Filled with MEDIAN (not 0%)\n")
        f.write("   - Created host_acceptance_missing flag\n")
        f.write("   - REASON: 0% too pessimistic for new hosts\n\n")
        
        f.write("4. SUPERHOST:\n")
        f.write("   - Filled missing with 'f' (not superhost)\n")
        f.write("   - REASON: Platform conservative, safe assumption\n\n")
        
        f.write(f"Final dataset: {len(df_clean):,} rows × {len(df_clean.columns)} columns\n")
        f.write(f"Ready for modeling!\n")
    
    print(f" SAVED: final_data/cleaning_report_FINAL.txt")
