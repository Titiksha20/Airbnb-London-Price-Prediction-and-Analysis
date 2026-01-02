"""
OPTION C: Best Balance Approach
- Start with proven XGBoost Initial parameters
- Add smart categorical features
- Apply light regularization
- Target: R2 0.67-0.70, Overfit < 0.10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print('OPTION C: Best Balance Model')
print('')

df = pd.read_csv('final_data/features_final.csv')
print(f'Loaded: {len(df):,} rows x {len(df.columns)} columns')

y = df['price_clean']
X = df.drop('price_clean', axis=1)

# STEP 1: ADD SMART CATEGORICAL FEATURES
print('\nSTEP 1: Adding Smart Categorical Features')
print('-' * 80)

if 'bedrooms_filled' in X.columns:
    X['is_studio_flag'] = (X['bedrooms_filled'] == 0).astype(int)
    X['is_small_1bed'] = (X['bedrooms_filled'] == 1).astype(int)
    X['is_medium_2_3bed'] = (X['bedrooms_filled'].isin([2, 3])).astype(int)
    X['is_large_4plus'] = (X['bedrooms_filled'] >= 4).astype(int)
    print('Added bedroom tier flags')

if 'bathrooms' in X.columns:
    X['is_multi_bathroom'] = (X['bathrooms'] > 1.5).astype(int)
    X['is_luxury_bathrooms'] = (X['bathrooms'] >= 3).astype(int)
    print('Added bathroom tier flags')

if all(col in X.columns for col in ['has_gym', 'has_pool', 'has_doorman', 'bathrooms']):
    X['is_luxury_property'] = (
        ((X['has_gym'] == 1) | (X['has_pool'] == 1) | (X['has_doorman'] == 1)) &
        (X['bathrooms'] >= 2)
    ).astype(int)
    print('Added luxury property flag')

if all(col in X.columns for col in ['bedrooms_filled', 'bathrooms', 'accommodates']):
    X['size_value_score'] = (
        X['bedrooms_filled'] * 0.4 +
        X['bathrooms'] * 0.3 +
        X['accommodates'] * 0.3
    )
    print('Added size value score')

print(f'Total columns after smart features: {X.shape[1]}')
print('')

# STEP 2: MINIMAL FEATURE SELECTION (Keep most features)
print('STEP 2: Minimal Feature Selection')
print('-' * 80)

X = X.select_dtypes(include=[np.number])
print(f'Numeric features: {X.shape[1]}')

nunique = X.nunique()
X = X.loc[:, nunique > 1]
print(f'After removing constants: {X.shape[1]}')

# Only remove VERY low correlation (< 0.005)
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
    print(f'Removed {len(low)} features with correlation < {threshold}')

print(f'Final features: {X.shape[1]}')
print('')

# STEP 3: TRAIN-TEST SPLIT
print('STEP 3: Train-Test Split')
print('-' * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Train: {len(X_train):,} samples')
print(f'Test:  {len(X_test):,} samples')
print(f'Features: {X_train.shape[1]}')
print('')

# STEP 4: BEST BALANCE XGBOOST
print('STEP 4: Training Best Balance XGBoost')
print('-' * 80)

xgb_balanced = xgb.XGBRegressor(
    n_estimators=200,        # Proven to work well
    max_depth=6,             # Good balance
    learning_rate=0.05,      # Not too fast, not too slow
    subsample=0.8,           # Standard
    colsample_bytree=0.8,    # Standard
    min_child_weight=3,      # Standard
    reg_lambda=0.5,          # Light L2 regularization (NEW)
    random_state=42,
    n_jobs=-1
)

print('Parameters:')
print('  n_estimators: 200 (proven)')
print('  max_depth: 6 (balanced)')
print('  learning_rate: 0.05 (proven)')
print('  subsample: 0.8')
print('  colsample_bytree: 0.8')
print('  min_child_weight: 3')
print('  reg_lambda: 0.5 (light regularization)')
print('')

print('Training...')
xgb_balanced.fit(X_train, y_train)
print('Complete')
print('')

# Cross-validation
print('Cross-validating (5-fold)...')
cv_scores = cross_val_score(xgb_balanced, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f'CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
print('')

# Predictions
y_pred_train = xgb_balanced.predict(X_train)
y_pred_test = xgb_balanced.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
overfit = train_r2 - test_r2

print('BEST BALANCE RESULTS:')
print(f'  Train R2:   {train_r2:.4f}')
print(f'  Test R2:    {test_r2:.4f}')
print(f'  Train RMSE: £{train_rmse:.2f}')
print(f'  Test RMSE:  £{test_rmse:.2f}')
print(f'  Train MAE:  £{train_mae:.2f}')
print(f'  Test MAE:   £{test_mae:.2f}')
print(f'  Overfit:    {overfit:.4f}')
print('')

# STEP 5: COMPREHENSIVE COMPARISON
print('STEP 5: Complete Model Comparison')
print('-' * 80)

comparison = pd.DataFrame({
    'Model': [
        'XGB_Best (GridSearch)',
        'XGBoost (Initial)',
        'XGB_Conservative',
        'XGB_Balanced (Option C)'
    ],
    'Test_R2': [0.6845, 0.6640, 0.6089, test_r2],
    'Test_RMSE': [84.05, 86.75, 93.60, test_rmse],
    'Test_MAE': [48.41, 50.32, 55.01, test_mae],
    'Overfit': [0.2083, 0.0699, 0.0155, overfit],
    'Features': [60, 60, 31, X_train.shape[1]],
    'CV_R2': [0.6682, 0.6516, np.nan, cv_scores.mean()]
})

print(comparison.to_string(index=False))
print('')

# Status
if test_r2 >= 0.67 and overfit <= 0.10:
    print('STATUS: TARGET ACHIEVED!')
    print('  Test R2 >= 0.67 AND Overfit <= 0.10')
elif test_r2 >= 0.65 and overfit <= 0.10:
    print('STATUS: VERY GOOD')
    print('  Excellent generalization, good performance')
elif test_r2 >= 0.60:
    print('STATUS: ACCEPTABLE')
    print('  Room for improvement')
else:
    print('STATUS: NEEDS MORE WORK')

print('')

# Best model selection
best_idx = comparison['Test_R2'].idxmax()
best_model = comparison.loc[best_idx, 'Model']
best_r2 = comparison.loc[best_idx, 'Test_R2']
best_overfit = comparison.loc[best_idx, 'Overfit']

print(f'BEST MODEL BY R2: {best_model}')
print(f'  R2: {best_r2:.4f}')
print(f'  Overfit: {best_overfit:.4f}')
print('')

# Best balanced model (R2 + low overfit)
comparison['balance_score'] = comparison['Test_R2'] - (comparison['Overfit'] * 0.5)
best_balanced_idx = comparison['balance_score'].idxmax()
best_balanced = comparison.loc[best_balanced_idx, 'Model']

print(f'BEST BALANCED MODEL: {best_balanced}')
print(f'  (Maximizes R2 while minimizing overfit)')
print('')

# STEP 6: FEATURE IMPORTANCE
print('STEP 6: Feature Importance Analysis')


feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_balanced.feature_importances_
}).sort_values('importance', ascending=False)

print('Top 25 Features:')
print(feat_imp.head(25).to_string(index=False))
print('')

# Check smart features performance
smart_features = ['is_studio_flag', 'is_small_1bed', 'is_medium_2_3bed', 'is_large_4plus',
                  'is_multi_bathroom', 'is_luxury_bathrooms', 'is_luxury_property', 'size_value_score']

smart_in_top = feat_imp.head(25)['feature'].isin(smart_features).sum()
print(f'Smart features in top 25: {smart_in_top}/{len(smart_features)}')

if smart_in_top > 0:
    print('\nSmart features performance:')
    smart_perf = feat_imp[feat_imp['feature'].isin(smart_features)]
    print(smart_perf.to_string(index=False))

print('')

# STEP 7: SAVE RESULTS
print('STEP 7: Saving Results')


comparison.to_csv('results/model_comparison_final.csv', index=False)
feat_imp.to_csv('results/feature_importance_final.csv', index=False)

with open('results/model_summary_final.txt', 'w') as f:
    f.write('OPTION C: Best Balance Model Results\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Test R2:    {test_r2:.4f}\n')
    f.write(f'Test RMSE:  £{test_rmse:.2f}\n')
    f.write(f'Test MAE:   £{test_mae:.2f}\n')
    f.write(f'Overfit:    {overfit:.4f}\n')
    f.write(f'CV R2:      {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n')
    f.write(f'Features:   {X_train.shape[1]}\n\n')
    f.write('Parameters:\n')
    f.write('  n_estimators: 200\n')
    f.write('  max_depth: 6\n')
    f.write('  learning_rate: 0.05\n')
    f.write('  reg_lambda: 0.5\n\n')
    f.write('Top 10 Features:\n')
    for i, row in feat_imp.head(10).iterrows():
        f.write(f'  {row["feature"]}: {row["importance"]:.4f}\n')

print('Saved:')
print('  - model_comparison_final.csv')
print('  - feature_importance_final.csv')
print('  - model_summary_final.txt')
print('')

# STEP 8: VISUALIZATION
print('STEP 8: Creating Visualizations')
print('-' * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# R2 comparison
ax = axes[0, 0]
models = comparison['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['red', 'orange', 'yellow', 'green']
bars = ax.bar(x_pos, comparison['Test_R2'], color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Test R2 Score', fontweight='bold')
ax.set_title('Model Comparison - Test R2', fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.67, color='green', linestyle='--', linewidth=2, label='Target (0.67)')
ax.legend()

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Overfitting comparison
ax = axes[0, 1]
overfit_colors = ['red' if x > 0.15 else 'orange' if x > 0.10 else 'green' for x in comparison['Overfit']]
bars = ax.bar(x_pos, comparison['Overfit'], color=overfit_colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0.10, color='green', linestyle='--', linewidth=2, label='Target (<0.10)')
ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=2, label='Acceptable (<0.15)')
ax.set_ylabel('Overfitting (Train R2 - Test R2)', fontweight='bold')
ax.set_title('Overfitting Comparison', fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# RMSE comparison
ax = axes[0, 2]
ax.bar(x_pos, comparison['Test_RMSE'], color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Test RMSE (GBP)', fontweight='bold')
ax.set_title('RMSE Comparison', fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Feature importance
ax = axes[1, 0]
top_15 = feat_imp.head(15).sort_values('importance')
colors_feat = ['green' if f in smart_features else 'steelblue' for f in top_15['feature']]
ax.barh(range(len(top_15)), top_15['importance'], color=colors_feat, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['feature'], fontsize=9)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Top 15 Features (green = smart features)', fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Predictions vs Actual
ax = axes[1, 1]
sample_size = min(1000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test.iloc[idx], y_pred_test[idx], alpha=0.4, s=20, color='steelblue', edgecolor='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Actual Price (GBP)', fontweight='bold')
ax.set_ylabel('Predicted Price (GBP)', fontweight='bold')
ax.set_title(f'Predictions vs Actual (R2={test_r2:.4f})', fontweight='bold', pad=20)
ax.legend()
ax.grid(alpha=0.3)

# Residual plot
ax = axes[1, 2]
residuals = y_test.values - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.4, s=20, color='coral', edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Price (GBP)', fontweight='bold')
ax.set_ylabel('Residuals (GBP)', fontweight='bold')
ax.set_title('Residual Plot', fontweight='bold', pad=20)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/final_results.png', dpi=150, bbox_inches='tight')
print('Saved: final_results.png')
print('')

# FINAL SUMMARY
print('FINAL SUMMARY -BEST BALANCE')

print('')
print(f'Test R2:        {test_r2:.4f}')
print(f'Test RMSE:      £{test_rmse:.2f}')
print(f'Test MAE:       £{test_mae:.2f}')
print(f'Overfit:        {overfit:.4f}')
print(f'CV R2:          {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
print(f'Features:       {X_train.shape[1]}')
print('')

if test_r2 >= 0.67 and overfit <= 0.10:
    print('VERDICT: SUCCESS - TARGET ACHIEVED')
elif test_r2 >= 0.65 and overfit <= 0.12:
    print('VERDICT: VERY GOOD - Close to target')
elif test_r2 >= best_r2 - 0.02:
    print('VERDICT: GOOD - Best overall balance')
    print(f'  Only {best_r2 - test_r2:.4f} below best R2')
    print(f'  But {best_overfit - overfit:.4f} less overfitting')
else:
    print('VERDICT: ACCEPTABLE')


print('')

print('COMPLETE')

