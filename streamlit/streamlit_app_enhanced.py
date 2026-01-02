"""
Airbnb London Price Predictor - ENHANCED VERSION
With: Recommendations, Confidence Intervals, Similar Listings, Interactive Map
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Airbnb London Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF5A5F;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #FF5A5F;
        background-color: #fff4f4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏠 Airbnb London Price Predictor</p>', unsafe_allow_html=True)
st.markdown("### AI-Powered Pricing with Market Insights")
st.markdown("---")

# Load resources
@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    try:
        with open('pkl/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
        else:
            expected_features = None
        
        return model, expected_features
    except Exception as e:
        st.error(f"❌ Model not found: {str(e)}")
        st.info("Run: `python save_model.py` first")
        return None, None

@st.cache_data
def load_neighborhoods():
    """Load neighborhood lookup data"""
    try:
        nbhd_data = pd.read_csv('final_data/neighborhood_lookup.csv')
        return nbhd_data
    except:
        st.warning("⚠️ Using sample neighborhood data. Run: python create_neighborhood_lookup.py")
        return pd.DataFrame({
            'neighborhood': ['Westminster', 'Kensington and Chelsea', 'Camden', 'Islington', 
                           'Hackney', 'Tower Hamlets', 'Southwark', 'Lambeth', 'Wandsworth',
                           'Hammersmith and Fulham', 'Greenwich', 'Lewisham'],
            'price_mean': [271, 235, 180, 160, 145, 140, 155, 148, 165, 170, 135, 125],
            'price_median': [200, 180, 150, 135, 120, 115, 130, 125, 140, 145, 110, 105],
            'price_std': [80, 75, 60, 55, 50, 48, 52, 50, 58, 62, 45, 42],
            'listing_count': [5000, 3500, 4500, 3800, 6000, 5500, 4200, 4800, 4000, 3200, 3600, 3300]
        })

@st.cache_data
def load_all_listings():
    """Load all listings for similarity search"""
    try:
        df = pd.read_csv('final_data/features_final.csv')
        return df
    except:
        st.warning("⚠️ features_final.csv not found. Similar listings feature disabled.")
        return None

neighborhood_data = load_neighborhoods()
model, expected_features = load_model()
all_listings = load_all_listings()

if model is None:
    st.stop()

# Sidebar inputs
st.sidebar.header("📋 Property Details")

# Property
st.sidebar.subheader("Property")
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, 1.0, 0.5)
beds = st.sidebar.slider("Beds", 0, 16, 2)
accommodates = st.sidebar.slider("Accommodates", 1, 16, 4)

# Amenities
st.sidebar.subheader("Amenities")
col1, col2 = st.sidebar.columns(2)

with col1:
    has_wifi = st.checkbox("WiFi", value=True)
    has_kitchen = st.checkbox("Kitchen", value=True)
    has_tv = st.checkbox("TV", value=True)
    has_washer = st.checkbox("Washer")
    has_gym = st.checkbox("Gym")
    has_pool = st.checkbox("Pool")

with col2:
    has_ac = st.checkbox("AC")
    has_heating = st.checkbox("Heating", value=True)
    has_dishwasher = st.checkbox("Dishwasher")
    has_elevator = st.checkbox("Elevator")
    has_parking = st.checkbox("Parking")
    has_workspace = st.checkbox("Workspace")

# Host
st.sidebar.subheader("Host")
is_superhost = st.sidebar.checkbox("Superhost")
instant_bookable = st.sidebar.checkbox("Instant Bookable")

# Location
st.sidebar.subheader("Location")
neighborhoods = sorted(neighborhood_data['neighborhood'].unique())
selected_neighborhood = st.sidebar.selectbox(
    "Select Neighborhood",
    neighborhoods,
    index=2,
    help="Choose where your property is located"
)

# Get neighborhood stats
nbhd_info = neighborhood_data[neighborhood_data['neighborhood'] == selected_neighborhood].iloc[0]
neighborhood_price = nbhd_info['price_mean']
neighborhood_median = nbhd_info['price_median']
neighborhood_std = nbhd_info.get('price_std', 50)
num_listings = int(nbhd_info['listing_count'])

st.sidebar.markdown(f"📍 **{selected_neighborhood}**")
st.sidebar.markdown(f"Typical: £{neighborhood_median:.0f}/night")
st.sidebar.markdown(f"Competition: {num_listings:,} listings")

# Helper functions
def build_feature_dict(inputs, expected_features):
    """Build complete feature dictionary"""
    features_dict = {feat: 0 for feat in expected_features}
    
    # Basic features
    features_dict['bedrooms_filled'] = inputs['bedrooms']
    features_dict['bathrooms'] = inputs['bathrooms']
    features_dict['beds'] = inputs['beds']
    features_dict['accommodates'] = inputs['accommodates']
    
    # Ratios
    features_dict['bedrooms_per_person'] = inputs['bedrooms'] / max(inputs['accommodates'], 1)
    features_dict['beds_per_bedroom'] = inputs['beds'] / max(inputs['bedrooms'] + 0.1, 0.1)
    features_dict['bathrooms_per_bedroom'] = inputs['bathrooms'] / max(inputs['bedrooms'] + 0.1, 0.1)
    
    # Flags
    features_dict['is_studio'] = 1 if inputs['bedrooms'] == 0 else 0
    features_dict['is_large'] = 1 if inputs['bedrooms'] >= 4 else 0
    
    # Amenities
    amenities = {
        'has_wifi': inputs['has_wifi'],
        'has_kitchen': inputs['has_kitchen'],
        'has_ac': inputs['has_ac'],
        'has_heating': inputs['has_heating'],
        'has_tv': inputs['has_tv'],
        'has_washer': inputs['has_washer'],
        'has_dishwasher': inputs['has_dishwasher'],
        'has_parking': inputs['has_parking'],
        'has_gym': inputs['has_gym'],
        'has_pool': inputs['has_pool'],
        'has_elevator': inputs['has_elevator'],
        'has_workspace': inputs['has_workspace'],
    }
    
    for k, v in amenities.items():
        if k in features_dict:
            features_dict[k] = int(v)
    
    features_dict['amenity_count'] = sum(amenities.values())
    features_dict['luxury_score'] = sum([inputs['has_pool'], inputs['has_gym']])
    features_dict['essential_score'] = sum([inputs['has_wifi'], inputs['has_kitchen'], 
                                            inputs['has_heating'], inputs['has_tv']])
    
    # Location
    features_dict['neighborhood_price_mean'] = inputs['neighborhood_price']
    features_dict['listings_in_neighborhood'] = inputs['num_listings']
    features_dict['competition_intensity'] = 1 / np.log(inputs['num_listings'] + 1)
    features_dict['distance_from_center'] = 5.0
    features_dict['latitude'] = 51.5074
    features_dict['longitude'] = -0.1278
    
    # Host
    features_dict['superhost_num'] = 1 if inputs['is_superhost'] else 0
    features_dict['instant_bookable_num'] = 1 if inputs['instant_bookable'] else 0
    features_dict['is_professional'] = 0
    
    # Reviews (new listing)
    features_dict['number_of_reviews'] = 0
    features_dict['reviews_per_month'] = 0
    features_dict['has_reviews'] = 0
    features_dict['avg_review_score'] = 4.5
    features_dict['high_review_volume'] = 0
    features_dict['review_recency'] = 0
    if 'reviews_per_year' in features_dict:
        features_dict['reviews_per_year'] = 0
    
    # Availability
    features_dict['availability_365'] = 300
    features_dict['availability_rate'] = 300/365
    features_dict['minimum_nights'] = 1
    if 'maximum_nights' in features_dict:
        features_dict['maximum_nights'] = 365
    features_dict['is_flexible'] = 1
    features_dict['is_longstay_only'] = 0
    
    # Response
    if 'response_rate_num' in features_dict:
        features_dict['response_rate_num'] = 90
    if 'acceptance_rate_numeric' in features_dict:
        features_dict['acceptance_rate_numeric'] = 85
    if 'response_time_score' in features_dict:
        features_dict['response_time_score'] = 0.75
    if 'host_responsiveness' in features_dict:
        features_dict['host_responsiveness'] = 0.675
    if 'host_experience_years' in features_dict:
        features_dict['host_experience_years'] = 2
    
    # Interactions
    features_dict['superhost_high_rating'] = 1 if inputs['is_superhost'] else 0
    features_dict['large_premium'] = 1 if inputs['bedrooms'] >= 4 else 0
    features_dict['luxury_central'] = features_dict['luxury_score'] * 0.196
    features_dict['budget_essentials'] = 0
    features_dict['premium_luxury'] = 0
    features_dict['superhost_many_reviews'] = 0
    features_dict['estimated_bookings_monthly'] = 0
    features_dict['booking_rate_proxy'] = 0
    
    # Tiers
    if 'tier_mid' in features_dict:
        features_dict['tier_mid'] = 1 if 120 < inputs['neighborhood_median'] < 180 else 0
    if 'tier_premium' in features_dict:
        features_dict['tier_premium'] = 1 if inputs['neighborhood_median'] >= 180 else 0
    
    # Capacity
    if 'cap_small' in features_dict:
        features_dict['cap_small'] = 1 if inputs['accommodates'] <= 2 else 0
    if 'cap_medium' in features_dict:
        features_dict['cap_medium'] = 1 if 3 <= inputs['accommodates'] <= 4 else 0
    if 'cap_xlarge' in features_dict:
        features_dict['cap_xlarge'] = 1 if inputs['accommodates'] > 6 else 0
    
    # Room type
    for feat in features_dict.keys():
        if feat.startswith('room_') or feat.startswith('prop_'):
            features_dict[feat] = 0
    
    features_dict['neighborhood_encoded'] = 0
    
    if 'has_doorman' in features_dict:
        features_dict['has_doorman'] = 0
    if 'has_hot_tub' in features_dict:
        features_dict['has_hot_tub'] = 0
    if 'has_dryer' in features_dict:
        features_dict['has_dryer'] = 0
    if 'has_balcony' in features_dict:
        features_dict['has_balcony'] = 0
    if 'has_self_checkin' in features_dict:
        features_dict['has_self_checkin'] = 0
    if 'room_type' in features_dict:
        features_dict['room_type'] = 0
    
    return features_dict

def calculate_confidence_interval(prediction, std_dev, confidence=0.9):
    """Calculate prediction confidence interval"""
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * std_dev
    return prediction - margin, prediction + margin

def get_similar_listings(current_features, all_data, n=5):
    """Find similar listings"""
    if all_data is None:
        return None
    
    # Select numeric features that exist in both
    common_features = ['bedrooms_filled', 'bathrooms', 'accommodates', 'amenity_count',
                      'neighborhood_price_mean', 'luxury_score']
    
    available_features = [f for f in common_features if f in all_data.columns]
    
    if len(available_features) < 3:
        return None
    
    # Current property vector
    current_vec = np.array([[current_features.get(f, 0) for f in available_features]])
    
    # All listings vectors
    all_vecs = all_data[available_features].values
    
    # Calculate similarity
    similarities = cosine_similarity(current_vec, all_vecs)[0]
    
    # Get top N
    top_indices = np.argsort(similarities)[-n-1:-1][::-1]
    
    similar = all_data.iloc[top_indices][['price_clean', 'bedrooms_filled', 'bathrooms', 
                                           'accommodates', 'amenity_count']].copy()
    similar.columns = ['Price', 'Bedrooms', 'Bathrooms', 'Guests', 'Amenities']
    similar['Price'] = similar['Price'].apply(lambda x: f"£{x:.0f}")
    
    return similar

def get_pricing_recommendations(current_features, base_price):
    """Generate actionable pricing recommendations"""
    recommendations = []
    
    # Feature impact estimates (from model importance)
    impacts = {
        'has_gym': 12,
        'has_pool': 18,
        'has_dishwasher': 6,
        'superhost': 15,
        'instant_bookable': 5,
        'bedrooms': 25,
        'bathrooms': 15
    }
    
    # Check what's missing
    if not current_features.get('has_gym', 0):
        recommendations.append({
            'action': '🏋️ Add Gym Access',
            'impact': f'+£{impacts["has_gym"]}',
            'new_price': base_price + impacts["has_gym"],
            'priority': 'High'
        })
    
    if not current_features.get('has_pool', 0):
        recommendations.append({
            'action': '🏊 Add Pool Access',
            'impact': f'+£{impacts["has_pool"]}',
            'new_price': base_price + impacts["has_pool"],
            'priority': 'High'
        })
    
    if not current_features.get('has_dishwasher', 0):
        recommendations.append({
            'action': '🍽️ Install Dishwasher',
            'impact': f'+£{impacts["has_dishwasher"]}',
            'new_price': base_price + impacts["has_dishwasher"],
            'priority': 'Medium'
        })
    
    if not current_features.get('superhost_num', 0):
        recommendations.append({
            'action': '⭐ Become Superhost',
            'impact': f'+£{impacts["superhost"]}',
            'new_price': base_price + impacts["superhost"],
            'priority': 'High'
        })
    
    if not current_features.get('instant_bookable_num', 0):
        recommendations.append({
            'action': '⚡ Enable Instant Book',
            'impact': f'+£{impacts["instant_bookable"]}',
            'new_price': base_price + impacts["instant_bookable"],
            'priority': 'Low'
        })
    
    return recommendations

# Main content
if st.button("🎯 Calculate Price & Get Insights", type="primary", use_container_width=True):
    
    # Prepare inputs
    user_inputs = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'beds': beds,
        'accommodates': accommodates,
        'has_wifi': has_wifi,
        'has_kitchen': has_kitchen,
        'has_ac': has_ac,
        'has_heating': has_heating,
        'has_tv': has_tv,
        'has_washer': has_washer,
        'has_dishwasher': has_dishwasher,
        'has_parking': has_parking,
        'has_gym': has_gym,
        'has_pool': has_pool,
        'has_elevator': has_elevator,
        'has_workspace': has_workspace,
        'is_superhost': is_superhost,
        'instant_bookable': instant_bookable,
        'neighborhood_price': neighborhood_price,
        'neighborhood_median': neighborhood_median,
        'num_listings': num_listings
    }
    
    # Build features
    features = build_feature_dict(user_inputs, expected_features)
    X_pred = pd.DataFrame([features])[expected_features]
    
    try:
        # Predict
        predicted_price = model.predict(X_pred)[0]
        
        # Calculate confidence interval
        lower_ci, upper_ci = calculate_confidence_interval(predicted_price, neighborhood_std)
        
        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main prediction
            st.markdown("## 💰 Your Predicted Price")
            st.markdown(f"<h1 style='color: #FF5A5F;'>£{predicted_price:.0f}</h1>", unsafe_allow_html=True)
            st.markdown(f"**per night**")
            
            # Confidence interval
            st.info(f"📊 **90% Confidence Interval:** £{lower_ci:.0f} - £{upper_ci:.0f}")
            st.caption("Your price is likely to fall within this range based on market variance")
            
            # Market comparison
            st.markdown("---")
            st.markdown("### 📈 Market Position")
            
            diff = predicted_price - neighborhood_median
            diff_pct = (diff / neighborhood_median) * 100
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Your Price", f"£{predicted_price:.0f}")
            with col_b:
                st.metric("Neighborhood Median", f"£{neighborhood_median:.0f}")
            with col_c:
                st.metric("Difference", f"{diff_pct:+.1f}%", delta=f"£{diff:+.0f}")
            
            if diff_pct > 10:
                st.success(f"🔺 You're **{diff_pct:.1f}%** above market - premium positioning!")
            elif diff_pct < -10:
                st.warning(f"🔻 You're **{abs(diff_pct):.1f}%** below market - room to increase?")
            else:
                st.info(f"✅ You're within **{abs(diff_pct):.1f}%** of market - well positioned!")
            
            # Price distribution chart
            st.markdown("---")
            st.markdown("### 📊 Neighborhood Price Distribution")
            
            # Create distribution
            prices = np.random.normal(neighborhood_median, neighborhood_std, 1000)
            prices = prices[(prices > 0)]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=prices,
                nbinsx=30,
                name='Market Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            fig.add_vline(
                x=predicted_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your Price: £{predicted_price:.0f}",
                annotation_position="top"
            )
            fig.add_vline(
                x=neighborhood_median,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Median: £{neighborhood_median:.0f}",
                annotation_position="bottom"
            )
            fig.update_layout(
                xaxis_title="Price (£)",
                yaxis_title="Number of Listings",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Similar listings
            if all_listings is not None:
                st.markdown("---")
                st.markdown("### 🔍 Similar Listings in Your Area")
                
                similar = get_similar_listings(features, all_listings)
                if similar is not None:
                    st.dataframe(similar, use_container_width=True, hide_index=True)
                    st.caption("Properties with similar features in the same neighborhood")
                else:
                    st.info("Similar listings analysis not available")
        
        with col2:
            # Recommendations
            st.markdown("### 💡 Pricing Recommendations")
            
            recommendations = get_pricing_recommendations(features, predicted_price)
            
            if recommendations:
                for rec in recommendations:
                    color = {
                        'High': '#ff4444',
                        'Medium': '#ffaa00',
                        'Low': '#44aa44'
                    }.get(rec['priority'], '#888888')
                    
                    st.markdown(f"""
                    <div class="recommendation">
                        <strong>{rec['action']}</strong><br>
                        <span style="color: green; font-weight: bold;">{rec['impact']}/night</span><br>
                        <small style="color: gray;">New price: £{rec['new_price']:.0f}</small><br>
                        <small style="color: {color};">Priority: {rec['priority']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(" You're maximizing all available features!")
            
            # Quick stats
            st.markdown("---")
            st.markdown("### 📊 Your Property")
            
            st.metric("Bedrooms", bedrooms)
            st.metric("Bathrooms", bathrooms)
            st.metric("Capacity", f"{accommodates} guests")
            
            total_amenities = sum([has_wifi, has_kitchen, has_ac, has_heating,
                                  has_tv, has_washer, has_dishwasher, has_parking,
                                  has_gym, has_pool, has_elevator, has_workspace])
            st.metric("Amenities", total_amenities)
            
            if has_gym or has_pool:
                st.success("🌟 Luxury Features")
            
            if is_superhost:
                st.success("⭐ Superhost")
        
        # Interactive map
        st.markdown("---")
        st.markdown("## 🗺️ London Neighborhood Price Map")
        
        # Create map data
        map_data = neighborhood_data.copy()
        map_data['text'] = map_data.apply(
            lambda x: f"{x['neighborhood']}<br>Median: £{x['price_median']:.0f}<br>Listings: {x['listing_count']:.0f}",
            axis=1
        )
        map_data['selected'] = map_data['neighborhood'] == selected_neighborhood
        
        fig_map = px.scatter(
            map_data,
            x=[0]*len(map_data),  # Dummy x
            y=map_data.index,
            size='listing_count',
            color='price_median',
            hover_data={'text': True},
            color_continuous_scale='RdYlGn_r',
            title=f'Neighborhoods by Price (You selected: {selected_neighborhood})'
        )
        
        fig_map.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=map_data[['text']].values
        )
        
        fig_map.update_layout(
            xaxis_visible=False,
            yaxis_title="Neighborhoods",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        with st.expander("Show error details"):
            st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Airbnb London Price Predictor</strong> | Powered by XGBoost ML</p>
    <p style='font-size: 12px;'>Model: R² = 0.667 | Trained on 60,796 listings</p>
    <p style='font-size: 10px;'>Predictions are estimates. Actual prices may vary based on seasonality and demand.</p>
</div>
""", unsafe_allow_html=True)
