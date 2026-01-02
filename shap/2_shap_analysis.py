"""
SHAP Analysis - Model Interpretability
Understand what drives predictions at global and local levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print('SHAP ANALYSIS - Model Interpretability')
print('')

# Load data
print('Loading data...')
df = pd.read_csv('final_data/features_final.csv')
print(f'Loaded: {len(df):,} rows')

y = df['price_clean']
X = df.drop('price_clean', axis=1)

# Prepare features (same as model training)
X = X.select_dtypes(include=[np.number])
nunique = X.nunique()
X = X.loc[:, nunique > 1]

# Remove very low correlation
corr = pd.Series(dtype=float)
for col in X.columns:
    try:
        c = X[col].corr(y)
        if not np.isnan(c):
            corr[col] = abs(c)
    except:
        pass

threshold = 0.005
low = corr[corr < threshold].index
if len(low) > 0:
    X = X.drop(columns=low)

print(f'Features: {X.shape[1]}')
print('')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
print('Loading model...')
with open('pkl/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model loaded')
print('')

# SHAP ANALYSIS

print('COMPUTING SHAP VALUES')
print('')

# Create SHAP explainer
print('Creating SHAP explainer...')
explainer = shap.TreeExplainer(model)
print('Explainer created')
print('')

# Calculate SHAP values on test set (sample for speed)
sample_size = min(1000, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42)
y_sample = y_test.loc[X_sample.index]

print(f'Computing SHAP values for {sample_size} samples...')
print('(This may take 1-2 minutes)')
shap_values = explainer.shap_values(X_sample)
print('SHAP values computed')
print('')


# GLOBAL FEATURE IMPORTANCE

print('GLOBAL FEATURE IMPORTANCE')
print('')

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

print('Top 20 Features by SHAP Importance:')
print(feature_importance.head(20).to_string(index=False))
print('')

# Save
feature_importance.to_csv('shap/shap_feature_importance.csv', index=False)
print('Saved: shap_feature_importance.csv')
print('')


# VISUALIZATIONS

print('CREATING VISUALIZATIONS')
print('')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Summary Plot (Beeswarm)
print('Creating summary plot...')
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.title('SHAP Summary Plot - Feature Impact on Price', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impact on Price)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap/shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_summary_plot.png')

# Figure 2: Bar Plot (Mean Absolute SHAP)
print('Creating bar plot...')
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = feature_importance.head(20).sort_values('importance')
ax.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'], fontsize=10)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Features - Average Impact on Price', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('shap/shap_bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_bar_plot.png')

# Figure 3: Dependence Plots (Top 4 features)
print('Creating dependence plots...')
top_4_features = feature_importance.head(4)['feature'].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_4_features):
    ax = axes[idx]
    shap.dependence_plot(
        feature, 
        shap_values, 
        X_sample, 
        interaction_index=None,
        ax=ax,
        show=False
    )
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')

plt.suptitle('Feature Dependence Plots - Top 4 Features', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('shap/shap_dependence_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_dependence_plots.png')

# Figure 4: Force Plot for Sample Predictions
print('Creating force plots...')

# High price example
high_price_idx = y_sample.nlargest(1).index[0]
high_price_local_idx = X_sample.index.get_loc(high_price_idx)

shap.initjs()
force_plot_high = shap.force_plot(
    explainer.expected_value,
    shap_values[high_price_local_idx],
    X_sample.iloc[high_price_local_idx],
    matplotlib=True,
    show=False
)
plt.title(f'High Price Example (£{y_sample.loc[high_price_idx]:.0f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap/shap_force_high.png', dpi=150, bbox_inches='tight')
plt.close()

# Low price example
low_price_idx = y_sample.nsmallest(1).index[0]
low_price_local_idx = X_sample.index.get_loc(low_price_idx)

force_plot_low = shap.force_plot(
    explainer.expected_value,
    shap_values[low_price_local_idx],
    X_sample.iloc[low_price_local_idx],
    matplotlib=True,
    show=False
)
plt.title(f'Low Price Example (£{y_sample.loc[low_price_idx]:.0f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap/shap_force_low.png', dpi=150, bbox_inches='tight')
plt.close()

print('Saved: shap_force_high.png')
print('Saved: shap_force_low.png')

print('')

# FEATURE INTERACTIONS

print('FEATURE INTERACTIONS')
print('')

print('Analyzing top feature interactions...')

# Get top 2 features
top_2 = feature_importance.head(2)['feature'].values
feat1, feat2 = top_2[0], top_2[1]

print(f'Top interaction pair: {feat1} × {feat2}')

# Interaction dependence plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot(
    feat1,
    shap_values,
    X_sample,
    interaction_index=feat2,
    ax=ax,
    show=False
)
ax.set_title(f'Interaction: {feat1} × {feat2}', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap/shap_interaction_top.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_interaction_top.png')
print('')

# ============================================================================
# BUSINESS INSIGHTS
# ============================================================================
print('=' * 80)
print('BUSINESS INSIGHTS FROM SHAP')
print('=' * 80)
print('')

# Analyze top features
insights = []

for idx, row in feature_importance.head(10).iterrows():
    feature = row['feature']
    importance = row['importance']
    
    # Get feature values and SHAP values
    feat_idx = list(X_sample.columns).index(feature)
    feat_values = X_sample[feature].values
    feat_shap = shap_values[:, feat_idx]
    
    # Calculate correlation
    if len(feat_values) > 1 and np.std(feat_values) > 0:
        correlation = np.corrcoef(feat_values, feat_shap)[0, 1]
    else:
        correlation = 0
    
    # Interpret
    direction = "positive" if correlation > 0 else "negative"
    
    insights.append({
        'Feature': feature,
        'Avg Impact': f'£{importance:.2f}',
        'Direction': direction,
        'Insight': f'Higher {feature} → {"higher" if correlation > 0 else "lower"} price'
    })

insights_df = pd.DataFrame(insights)
print(insights_df.to_string(index=False))
print('')

# Save insights
insights_df.to_csv('shap/shap_business_insights.csv', index=False)
print('Saved: shap_business_insights.csv')
print('')


# QUANTIFY FEATURE IMPACTS

print('FEATURE IMPACT QUANTIFICATION')

print('')

print('Average £ impact per unit increase:')
print('')

# For top continuous features, calculate £ per unit
impact_quantification = []

for feature in feature_importance.head(10)['feature']:
    feat_idx = list(X_sample.columns).index(feature)
    feat_values = X_sample[feature].values
    feat_shap = shap_values[:, feat_idx]
    
    # Only for features with variance
    if np.std(feat_values) > 0.01:
        # Linear regression: SHAP ~ feature
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(feat_values, feat_shap)
        
        impact_quantification.append({
            'Feature': feature,
            'Impact per Unit': f'£{slope:.2f}',
            'R²': f'{r_value**2:.3f}',
            'P-value': f'{p_value:.4f}'
        })

if impact_quantification:
    impact_df = pd.DataFrame(impact_quantification)
    print(impact_df.to_string(index=False))
    print('')
    impact_df.to_csv('shap/shap_impact_quantification.csv', index=False)
    print('Saved: shap_impact_quantification.csv')
else:
    print('No continuous features with sufficient variance')

print('')

# SHAP WATERFALL FOR SPECIFIC CASES


print('WATERFALL PLOTS - EXPLAINING SPECIFIC PREDICTIONS')
print('')

print('Creating waterfall plots for example predictions...')

# Example 1: Median price property
median_price = y_sample.median()
median_idx = (y_sample - median_price).abs().idxmin()
median_local_idx = X_sample.index.get_loc(median_idx)

fig, ax = plt.subplots(figsize=(10, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[median_local_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[median_local_idx].values,
        feature_names=X_sample.columns.tolist()
    ),
    show=False
)
plt.title(f'Waterfall: Median Price Property (£{y_sample.loc[median_idx]:.0f})', 
          fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap/shap_waterfall_median.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: shap_waterfall_median.png')

print('')


# SUMMARY REPORT


print('SUMMARY REPORT')

print('')

summary = f"""
SHAP ANALYSIS COMPLETE

Model Base Value (Expected): £{explainer.expected_value:.2f}

TOP 5 DRIVERS:
{feature_importance.head(5).to_string(index=False)}

KEY INSIGHTS:
1. {insights_df.iloc[0]['Feature']}: {insights_df.iloc[0]['Insight']}
2. {insights_df.iloc[1]['Feature']}: {insights_df.iloc[1]['Insight']}
3. {insights_df.iloc[2]['Feature']}: {insights_df.iloc[2]['Insight']}

VISUALIZATIONS CREATED:
- shap_summary_plot.png (beeswarm - feature impact distribution)
- shap_bar_plot.png (mean importance)
- shap_dependence_plots.png (top 4 feature relationships)
- shap_force_high.png (high price explanation)
- shap_force_low.png (low price explanation)
- shap_interaction_top.png (top interaction)
- shap_waterfall_median.png (median case breakdown)

DATA FILES SAVED:
- shap_feature_importance.csv
- shap_business_insights.csv
- shap_impact_quantification.csv

NEXT STEPS:
1. Review visualizations
2. Extract business recommendations
3. Update Streamlit app with insights
4. Document findings in README
"""

print(summary)

# Save summary
with open('shap/shap_analysis_summary2.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print('Saved: shap_analysis_summary2.txt')
print('')
print('=' * 80)
print('SHAP ANALYSIS COMPLETE')
print('=' * 80)