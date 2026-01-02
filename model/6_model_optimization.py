"""
Improved Modeling 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print('Improved Modeling Pipeline')

print('')

df = pd.read_csv('final_data/features_final.csv')
print(f'Loaded: {len(df):,} rows x {len(df.columns)} columns')

y = df['price_clean']
X = df.drop('price_clean', axis=1)

# STEP 1: SMART FEATURES
print('\nSTEP 1: Creating Smart Features')


if 'bedrooms_filled' in X.columns:
    X['is_studio_flag'] = (X['bedrooms_filled'] == 0).astype(int)
    X['is_small_1bed'] = (X['bedrooms_filled'] == 1).astype(int)
    X['is_medium_2_3bed'] = (X['bedrooms_filled'].isin([2, 3])).astype(int)
    X['is_large_4plus'] = (X['bedrooms_filled'] >= 4).astype(int)
    print('Created bedroom tier flags')

if 'bathrooms' in X.columns:
    X['is_multi_bathroom'] = (X['bathrooms'] > 1.5).astype(int)
    X['is_luxury_bathrooms'] = (X['bathrooms'] >= 3).astype(int)
    print('Created bathroom tier flags')

if all(col in X.columns for col in ['has_gym', 'has_pool', 'has_doorman', 'bathrooms']):
    X['is_luxury_property'] = (
        ((X['has_gym'] == 1) | (X['has_pool'] == 1) | (X['has_doorman'] == 1)) &
        (X['bathrooms'] >= 2)
    ).astype(int)
    print('Created luxury property flag')

if all(col in X.columns for col in ['bedrooms_filled', 'bathrooms', 'accommodates']):
    X['size_value_score'] = (
        X['bedrooms_filled'] * 0.4 +
        X['bathrooms'] * 0.3 +
        X['accommodates'] * 0.3
    )
    print('Created size value score')

print(f'Total columns: {X.shape[1]}')
print('')

# STEP 2: FEATURE SELECTION
print('STEP 2: Feature Selection')


X = X.select_dtypes(include=[np.number])
print(f'Numeric features: {X.shape[1]}')

nunique = X.nunique()
X = X.loc[:, nunique > 1]
print(f'After removing constants: {X.shape[1]}')

corr = pd.Series(dtype=float)
for col in X.columns:
    try:
        c = X[col].corr(y)
        if not np.isnan(c):
            corr[col] = abs(c)
    except:
        pass

threshold = 0.01
low = corr[corr < threshold].index
if len(low) > 0:
    X = X.drop(columns=low)
    print(f'Removed {len(low)} features with correlation < {threshold}')

print(f'Features: {X.shape[1]}')

top_features = [
    'is_large', 'bedrooms_filled', 'neighborhood_price_mean', 'bathrooms',
    'has_gym', 'listings_in_neighborhood', 'accommodates', 'superhost_high_rating',
    'large_premium', 'has_dishwasher', 'has_tv', 'superhost_num',
    'estimated_bookings_monthly', 'luxury_central', 'booking_rate_proxy',
    'is_flexible', 'distance_from_center', 'instant_bookable_num',
    'is_longstay_only', 'availability_365', 'latitude', 'bedrooms_per_person',
    'review_recency', 'has_doorman', 'tier_mid', 'tier_premium',
    'is_professional', 'is_studio_flag', 'is_small_1bed', 'is_medium_2_3bed',
    'is_large_4plus', 'is_multi_bathroom', 'is_luxury_bathrooms',
    'is_luxury_property', 'size_value_score'
]

selected_features = []
for feat in top_features:
    if feat in X.columns:
        selected_features.append(feat)

if len(selected_features) < 30:
    remaining = corr.drop(selected_features, errors='ignore').sort_values(ascending=False)
    for feat in remaining.head(30 - len(selected_features)).index:
        if feat not in selected_features:
            selected_features.append(feat)

X_selected = X[selected_features]
print(f'Selected {len(selected_features)} features')
print('')

# STEP 3: TRAIN-TEST SPLIT
print('STEP 3: Train-Test Split')


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f'Train: {len(X_train):,} samples')
print(f'Test:  {len(X_test):,} samples')
print(f'Features: {X_train.shape[1]}')
print('')

# STEP 4: CONSERVATIVE XGBOOST
print('STEP 4: Training Conservative XGBoost')


xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

print('Training...')
xgb_model.fit(X_train, y_train)
print('Training complete')
print('')

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
overfit = train_r2 - test_r2

print('RESULTS:')
print(f'  Train R2:   {train_r2:.4f}')
print(f'  Test R2:    {test_r2:.4f}')
print(f'  Train RMSE: £{train_rmse:.2f}')
print(f'  Test RMSE:  £{test_rmse:.2f}')
print(f'  Train MAE:  £{train_mae:.2f}')
print(f'  Test MAE:   £{test_mae:.2f}')
print(f'  Overfit:    {overfit:.4f}')
print('')

# STEP 5: COMPARISON
print('STEP 5: Comparison with Previous Results')

comparison = pd.DataFrame({
    'Model': ['XGB_Best (GridSearch)', 'XGBoost (Initial)', 'XGB_Conservative (New)'],
    'Test_R2': [0.6845, 0.6640, test_r2],
    'Test_RMSE': [84.05, 86.75, test_rmse],
    'Test_MAE': [48.41, 50.32, test_mae],
    'Overfit': [0.2083, 0.0699, overfit],
    'Features': [60, 60, len(selected_features)]
})

print(comparison.to_string(index=False))
print('')

if overfit < 0.10:
    print('OVERFIT STATUS: EXCELLENT')
elif overfit < 0.15:
    print('OVERFIT STATUS: GOOD')
else:
    print('OVERFIT STATUS: NEEDS IMPROVEMENT')

print('')

# STEP 6: FEATURE IMPORTANCE
print('STEP 6: Feature Importance')

feat_imp = pd.DataFrame({
    'feature': selected_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print('Top 20 Features:')
print(feat_imp.head(20).to_string(index=False))
print('')

# SAVE RESULTS
comparison.to_csv('results/model_comparison_opt.csv', index=False)
feat_imp.to_csv('results/feature_importance_opt.csv', index=False)

with open('results/selected_features_opt.txt', 'w') as f:
    f.write(f'Selected Features ({len(selected_features)})\n\n')
    for i, feat in enumerate(selected_features, 1):
        f.write(f'{i}. {feat}\n')

print('Saved:')
print('  - model_comparison_opt.csv')
print('  - feature_importance_opt.csv')
print('  - selected_features_opt.txt')
print('')

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R2 comparison
ax = axes[0, 0]
models = comparison['Model'].tolist()
x_pos = np.arange(len(models))
ax.bar(x_pos, comparison['Test_R2'], color=['coral', 'steelblue', 'green'], alpha=0.7)
ax.set_ylabel('Test R2 Score')
ax.set_title('Model Comparison - Test R2')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Overfitting
ax = axes[0, 1]
colors = ['red' if x > 0.15 else 'orange' if x > 0.10 else 'green' for x in comparison['Overfit']]
ax.bar(x_pos, comparison['Overfit'], color=colors, alpha=0.7)
ax.axhline(y=0.10, color='green', linestyle='--', linewidth=2)
ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=2)
ax.set_ylabel('Overfitting')
ax.set_title('Overfitting Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Feature importance
ax = axes[1, 0]
top_15 = feat_imp.head(15).sort_values('importance')
ax.barh(range(len(top_15)), top_15['importance'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['feature'], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Top 15 Features')
ax.grid(axis='x', alpha=0.3)

# Predictions vs Actual
ax = axes[1, 1]
sample_size = min(1000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test.iloc[idx], y_pred_test[idx], alpha=0.4, s=15, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title(f'Predictions vs Actual (R2={test_r2:.4f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/opt_model_results.png', dpi=150, bbox_inches='tight')
print('Saved: opt_model_results.png')
print('')

print('SUMMARY')

print(f'Test R2:    {test_r2:.4f}')
print(f'Test RMSE:  £{test_rmse:.2f}')
print(f'Test MAE:   £{test_mae:.2f}')
print(f'Overfit:    {overfit:.4f}')
print(f'Features:   {len(selected_features)}')
print('')
print('COMPLETE')
