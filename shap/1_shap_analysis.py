"""
Advanced SHAP Analysis - Part 2
1. SHAP Interaction Matrix (all pairwise interactions)
2. Cluster Analysis (market segmentation)
All results output as TEXT for easy interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('ADVANCED SHAP ANALYSIS - INTERACTIONS & CLUSTERING')
print('=' * 80)
print('')

# Load data
print('Loading data and model...')
df = pd.read_csv('final_data/features_final.csv')
y = df['price_clean']
X = df.drop('price_clean', axis=1)

# Prepare features (same as before)
X = X.select_dtypes(include=[np.number])
nunique = X.nunique()
X = X.loc[:, nunique > 1]

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
with open('pkl/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model loaded')
print('')

# Sample for speed
sample_size = min(500, len(X_test))  # Smaller for interaction matrix
X_sample = X_test.sample(sample_size, random_state=42)
y_sample = y_test.loc[X_sample.index]

print(f'Using {sample_size} samples for analysis')
print('')


# PART 1: SHAP INTERACTION VALUES


print('PART 1: SHAP INTERACTION MATRIX')
print('')

print('Computing SHAP interaction values...')
print('(This will take 10-15 minutes - computing all pairwise interactions)')
print('')

explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_sample)

print('Interaction values computed!')
print('')

# Analyze interaction strengths
print('Analyzing interaction strengths...')
print('')

# Get mean absolute interaction values
n_features = len(X_sample.columns)
interaction_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        if i != j:
            # Off-diagonal = interaction strength
            interaction_matrix[i, j] = np.abs(shap_interaction_values[:, i, j]).mean()
        else:
            # Diagonal = main effect
            interaction_matrix[i, j] = np.abs(shap_interaction_values[:, i, i]).mean()

# Create DataFrame
interaction_df = pd.DataFrame(
    interaction_matrix,
    index=X_sample.columns,
    columns=X_sample.columns
)

# Find top interactions (off-diagonal only)
interactions_list = []
for i in range(n_features):
    for j in range(i+1, n_features):  # Upper triangle only
        interactions_list.append({
            'Feature_1': X_sample.columns[i],
            'Feature_2': X_sample.columns[j],
            'Interaction_Strength': interaction_matrix[i, j],
            'Feature_1_Main': interaction_matrix[i, i],
            'Feature_2_Main': interaction_matrix[j, j],
            'Relative_Strength': interaction_matrix[i, j] / (interaction_matrix[i, i] + 0.001)
        })

interactions_ranked = pd.DataFrame(interactions_list).sort_values(
    'Interaction_Strength', ascending=False
)

# OUTPUT: TOP INTERACTIONS


output = []

output.append("TOP 20 FEATURE INTERACTIONS")
output.append("")
output.append("Interaction Strength = How much features work together to affect price")
output.append("Higher value = Stronger interaction effect")
output.append("")

for idx, row in interactions_ranked.head(20).iterrows():
    output.append(f"{idx+1}. {row['Feature_1']} × {row['Feature_2']}")
    output.append(f"   Interaction Strength: {row['Interaction_Strength']:.2f}")
    output.append(f"   {row['Feature_1']} main effect: {row['Feature_1_Main']:.2f}")
    output.append(f"   {row['Feature_2']} main effect: {row['Feature_2_Main']:.2f}")
    output.append(f"   Relative to main: {row['Relative_Strength']:.1%}")
    output.append("")

# Detailed interpretation of top 5
output.append("=" * 80)
output.append("DETAILED INTERPRETATION - TOP 5 INTERACTIONS")
output.append("=" * 80)
output.append("")

for idx, row in interactions_ranked.head(5).iterrows():
    feat1, feat2 = row['Feature_1'], row['Feature_2']
    strength = row['Interaction_Strength']
    
    output.append(f"INTERACTION #{idx+1}: {feat1} × {feat2}")
    output.append(f"Strength: {strength:.2f}")
    output.append("")
    
    # Interpret based on feature names
    if 'bedroom' in feat1.lower() and 'bathroom' in feat2.lower():
        output.append("INTERPRETATION:")
        output.append("  The value of bedrooms changes based on bathroom count")
        output.append("  Example: 3BR with 1 bath ≠ 3BR with 2 baths")
        output.append("  Business insight: Bathroom/bedroom ratio matters")
        
    elif 'neighborhood' in feat1.lower() or 'neighborhood' in feat2.lower():
        output.append("INTERPRETATION:")
        output.append("  This feature's impact depends on neighborhood quality")
        output.append("  Example: Gym matters MORE in premium neighborhoods")
        output.append("  Business insight: Context-dependent amenity value")
        
    elif 'distance' in feat1.lower() or 'distance' in feat2.lower():
        output.append("INTERPRETATION:")
        output.append("  Distance effect varies by other location factors")
        output.append("  Example: Being far from center hurts LESS in good neighborhoods")
        output.append("  Business insight: Distance penalty is conditional")
        
    elif 'review' in feat1.lower() or 'review' in feat2.lower():
        output.append("INTERPRETATION:")
        output.append("  Review impact depends on property characteristics")
        output.append("  Example: Good reviews matter MORE for budget properties")
        output.append("  Business insight: Reviews as trust signal for unknowns")
        
    elif 'amenity' in feat1.lower() or 'has_' in feat1.lower():
        output.append("INTERPRETATION:")
        output.append("  Amenity value changes based on property type")
        output.append("  Example: Dishwasher matters MORE in larger properties")
        output.append("  Business insight: Amenity ROI varies by segment")
    else:
        output.append("INTERPRETATION:")
        output.append("  These features influence each other's price impact")
        output.append("  Check dependence plot colored by second feature")
        output.append("  Business insight: Combined effect ≠ sum of parts")
    
    output.append("")
    output.append("-" * 80)
    output.append("")

# Save interactions
interactions_ranked.to_csv('shap/shap_interactions_ranked.csv', index=False)
output.append("Saved: shap_interactions_ranked.csv")
output.append("")


# VISUALIZE INTERACTION MATRIX

print('Creating interaction matrix heatmap...')

# Select top 20 features for visualization
top_features = interactions_ranked['Feature_1'].head(20).unique()[:15]
top_features = list(top_features)

# Add a few more important main effects
main_effects = pd.Series(
    np.diag(interaction_matrix),
    index=X_sample.columns
).sort_values(ascending=False)

for feat in main_effects.head(10).index:
    if feat not in top_features and len(top_features) < 20:
        top_features.append(feat)

interaction_subset = interaction_df.loc[top_features, top_features]

# Plot
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    interaction_subset,
    annot=False,
    cmap='RdYlBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Interaction Strength'},
    ax=ax
)
ax.set_title('SHAP Interaction Matrix - Top Features', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('shap/shap_interaction_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print('Saved: shap_interaction_matrix.png')
print('')


# PART 2: CLUSTER ANALYSIS

print('PART 2: CLUSTER ANALYSIS - MARKET SEGMENTATION')

print('')

print('Computing SHAP values for clustering...')
shap_values = explainer.shap_values(X_sample)
print('Complete')
print('')

# Use SHAP values as features for clustering
shap_features = pd.DataFrame(
    shap_values,
    columns=X_sample.columns,
    index=X_sample.index
)

# Add actual features for interpretation
cluster_data = X_sample.copy()
cluster_data['price'] = y_sample

# Standardize SHAP values
scaler = StandardScaler()
shap_scaled = scaler.fit_transform(shap_features)

# Find optimal number of clusters (elbow method)
print('Finding optimal number of clusters...')
inertias = []
silhouettes = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(shap_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(shap_scaled, kmeans.labels_))

# Plot elbow
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters', fontweight='bold')
axes[0].set_ylabel('Inertia', fontweight='bold')
axes[0].set_title('Elbow Method', fontweight='bold', pad=15)
axes[0].grid(alpha=0.3)

axes[1].plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters', fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontweight='bold', pad=15)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('shap/cluster_optimization.png', dpi=150, bbox_inches='tight')
plt.close()

# Choose optimal k (using silhouette score)
optimal_k = K_range[np.argmax(silhouettes)]
print(f'Optimal number of clusters: {optimal_k} (by silhouette score)')
print('')

# Perform final clustering
print(f'Clustering with k={optimal_k}...')
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(shap_scaled)

cluster_data['cluster'] = clusters

print(f'Clusters created')
print('')

# ANALYZE CLUSTERS


output.append("CLUSTER ANALYSIS - MARKET SEGMENTS")
output.append("")
output.append(f"Optimal clusters: {optimal_k}")
output.append("")

cluster_summary = []

for cluster_id in range(optimal_k):
    cluster_mask = clusters == cluster_id
    cluster_subset = cluster_data[cluster_mask]
    shap_subset = shap_features[cluster_mask]
    
    output.append("=" * 80)
    output.append(f"CLUSTER {cluster_id}: {cluster_mask.sum()} listings ({cluster_mask.sum()/len(clusters)*100:.1f}%)")
    output.append("=" * 80)
    output.append("")
    
    # Basic statistics
    output.append("PRICE STATISTICS:")
    output.append(f"  Mean: £{cluster_subset['price'].mean():.2f}")
    output.append(f"  Median: £{cluster_subset['price'].median():.2f}")
    output.append(f"  Std: £{cluster_subset['price'].std():.2f}")
    output.append(f"  Range: £{cluster_subset['price'].min():.0f} - £{cluster_subset['price'].max():.0f}")
    output.append("")
    
    # Property characteristics
    if 'bedrooms_filled' in cluster_subset.columns:
        output.append("PROPERTY CHARACTERISTICS:")
        output.append(f"  Avg Bedrooms: {cluster_subset['bedrooms_filled'].mean():.1f}")
        output.append(f"  Avg Bathrooms: {cluster_subset['bathrooms'].mean():.1f}")
        output.append(f"  Avg Capacity: {cluster_subset['accommodates'].mean():.1f}")
        output.append("")
    
    # Top SHAP contributors (what makes this cluster different)
    avg_shap = shap_subset.mean()
    top_positive = avg_shap.nlargest(5)
    top_negative = avg_shap.nsmallest(5)
    
    output.append("TOP POSITIVE PRICE DRIVERS:")
    for feat, val in top_positive.items():
        output.append(f"  {feat}: +£{val:.2f}")
    output.append("")
    
    output.append("TOP NEGATIVE PRICE DRIVERS:")
    for feat, val in top_negative.items():
        output.append(f"  {feat}: £{val:.2f}")
    output.append("")
    
    # Characterize cluster
    avg_price = cluster_subset['price'].mean()
    avg_bedrooms = cluster_subset['bedrooms_filled'].mean() if 'bedrooms_filled' in cluster_subset.columns else 0
    
    if avg_price > 250:
        segment = "LUXURY"
    elif avg_price > 150:
        segment = "PREMIUM"
    elif avg_price > 100:
        segment = "MID-MARKET"
    else:
        segment = "BUDGET"
    
    if avg_bedrooms > 3:
        size = "LARGE"
    elif avg_bedrooms > 1.5:
        size = "MEDIUM"
    else:
        size = "SMALL/STUDIO"
    
    output.append(f"SEGMENT LABEL: {segment} {size}")
    output.append("")
    
    cluster_summary.append({
        'Cluster': cluster_id,
        'Count': cluster_mask.sum(),
        'Pct': f"{cluster_mask.sum()/len(clusters)*100:.1f}%",
        'Avg_Price': f"£{cluster_subset['price'].mean():.0f}",
        'Segment': f"{segment} {size}"
    })
    
    output.append("-" * 80)
    output.append("")

# Cluster summary table
output.append("=" * 80)
output.append("CLUSTER SUMMARY TABLE")
output.append("=" * 80)
output.append("")

cluster_summary_df = pd.DataFrame(cluster_summary)
output.append(cluster_summary_df.to_string(index=False))
output.append("")

cluster_summary_df.to_csv('shap/cluster_summary.csv', index=False)
output.append("Saved: cluster_summary.csv")
output.append("")

# CLUSTER VISUALIZATION

print('Creating cluster visualizations...')

# PCA for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
shap_pca = pca.fit_transform(shap_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Clusters in SHAP space
scatter = axes[0].scatter(
    shap_pca[:, 0],
    shap_pca[:, 1],
    c=clusters,
    cmap='Set1',
    s=50,
    alpha=0.6,
    edgecolor='black',
    linewidth=0.5
)
axes[0].set_xlabel('First Principal Component', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Second Principal Component', fontweight='bold', fontsize=11)
axes[0].set_title('Clusters in SHAP Space (PCA)', fontweight='bold', fontsize=14, pad=15)
axes[0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Plot 2: Clusters by price
scatter2 = axes[1].scatter(
    cluster_data['bedrooms_filled'] if 'bedrooms_filled' in cluster_data.columns else range(len(cluster_data)),
    cluster_data['price'],
    c=clusters,
    cmap='Set1',
    s=50,
    alpha=0.6,
    edgecolor='black',
    linewidth=0.5
)
axes[1].set_xlabel('Bedrooms', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Price (£)', fontweight='bold', fontsize=11)
axes[1].set_title('Clusters by Bedrooms vs Price', fontweight='bold', fontsize=14, pad=15)
axes[1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('shap/cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print('Saved: cluster_visualization.png')
print('Saved: cluster_optimization.png')
print('')


# FEATURE IMPORTANCE BY CLUSTER

output.append("=" * 80)
output.append("FEATURE IMPORTANCE BY CLUSTER")
output.append("=" * 80)
output.append("")
output.append("Shows which features matter MOST in each market segment")
output.append("")

fig, axes = plt.subplots(optimal_k, 1, figsize=(12, 4*optimal_k))
if optimal_k == 1:
    axes = [axes]

for cluster_id in range(optimal_k):
    cluster_mask = clusters == cluster_id
    shap_subset = shap_features[cluster_mask]
    
    # Mean absolute SHAP by feature
    importance = shap_subset.abs().mean().sort_values(ascending=False).head(15)
    
    # Plot
    ax = axes[cluster_id]
    importance.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
    ax.set_title(f'Cluster {cluster_id} - Top 15 Features', fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add to text output
    output.append(f"CLUSTER {cluster_id} - TOP 10 DRIVERS:")
    for idx, (feat, val) in enumerate(importance.head(10).items(), 1):
        output.append(f"  {idx}. {feat}: {val:.2f}")
    output.append("")

plt.tight_layout()
plt.savefig('shap/cluster_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print('Saved: shap/cluster_feature_importance.png')
print('')


# SAVE ALL TEXT OUTPUT

output_text = "\n".join(output)

with open('shap/shap_advanced_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(output_text)

print('=' * 80)
print('ADVANCED ANALYSIS COMPLETE')
print('=' * 80)
print('')
print('FILES CREATED:')
print('  1. shap_interactions_ranked.csv - All pairwise interactions')
print('  2. shap_interaction_matrix.png - Heatmap visualization')
print('  3. cluster_summary.csv - Cluster statistics')
print('  4. cluster_optimization.png - Elbow & silhouette plots')
print('  5. cluster_visualization.png - Cluster scatter plots')
print('  6. cluster_feature_importance.png - Drivers by segment')
print('  7. shap_advanced_analysis.txt - Complete text interpretation')
print('')
print('Review shap_advanced_analysis.txt for detailed findings!')
print('')

# Print summary to console
print(output_text)
