"""
Data Loader for Inside Airbnb London Dataset
Loads the three main files: listings, calendar, and reviews
"""

import pandas as pd
import numpy as np

def load_listings():
    """Load the detailed listings data"""
    print("Loading listings.csv.gz...")
    listings = pd.read_csv('raw_data/listings.csv.gz', compression='gzip', low_memory=False)
    print(f"Loaded {len(listings):,} listings with {listings.shape[1]} columns")
    return listings

def load_calendar():
    """Load the calendar data (availability and pricing)"""
    print("Loading calendar.csv.gz...")
    calendar = pd.read_csv('raw_data/calendar.csv.gz', compression='gzip', low_memory=False)
    print(f"Loaded {len(calendar):,} calendar records")
    return calendar

def load_reviews():
    """Load the reviews data"""
    print("Loading reviews.csv.gz...")
    reviews = pd.read_csv('raw_data/reviews.csv.gz', compression='gzip', low_memory=False)
    print(f"Loaded {len(reviews):,} reviews")
    return reviews

def load_all_data():
    """Load all three datasets"""
    print("LOADING INSIDE AIRBNB DATA - LONDON")

    print()
    
    listings = load_listings()
    calendar = load_calendar()
    reviews = load_reviews()
    
    print()

    print("DATA LOADED SUCCESSFULLY")

    
    return {
        'listings': listings,
        'calendar': calendar,
        'reviews': reviews
    }

def quick_summary(data):
    """Display quick summary of the loaded data"""
    
    listings = data['listings']
    calendar = data['calendar']
    reviews = data['reviews']
    
 
    print("DATASET SUMMARY")

    
    print(f"\n LISTINGS:")
    print(f"  Total listings: {len(listings):,}")
    print(f"  Columns: {listings.shape[1]}")
    print(f"  Date range: {listings.columns.tolist()[:5]}...")
    
    print(f"\n CALENDAR:")
    print(f"   Total records: {len(calendar):,}")
    print(f"   Unique listings: {calendar['listing_id'].nunique():,}")
    
    print(f"\n REVIEWS:")
    print(f"   Total reviews: {len(reviews):,}")
    print(f"   Listings with reviews: {reviews['listing_id'].nunique():,}")
    
   


if __name__ == "__main__":
    # Load all data
    data = load_all_data()
    
    # Show summary
    quick_summary(data)
    
    # Show first few rows of listings
    print("\nFirst 5 listings preview:")
    print(data['listings'].head())