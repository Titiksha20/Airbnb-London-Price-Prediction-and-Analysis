"""
Advanced Modeling Pipeline
- Fix Random Forest overfitting
- XGBoost with hyperparameter tuning
- Ensemble modeling
- Cross-validation
- Feature analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print('Advanced Modeling Pipeline')
print('')

df = pd.read_csv('final_data/features_final.csv')
print(f'Loaded: {len(df):,} rows x {len(df.columns)} columns')

y = df['price_clean']
X = df.drop('price_clean', axis=1)

X = X.select_dtypes(include=[np.number])
print(f'Numeric features: {X.shape[1]}')

nunique = X.nunique()
X = X.loc[:, nunique > 1]
print(f'After removing constants: {X.shape[1]}')

# Remove low correlation features
print('\nRemoving low correlation features...')
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
    print(f'Removed {len(low)} features (correlation < {threshold})')

print(f'Final features: {X.shape[1]}')
print('')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')
print('')

# Scale for Ridge
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# BASELINE: Ridge
print('BASELINE MODEL: Ridge Regression')

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_sc, y_train)

y_pred_ridge_train = ridge.predict(X_train_sc)
y_pred_ridge_test = ridge.predict(X_test_sc)

ridge_train_r2 = r2_score(y_train, y_pred_ridge_train)
ridge_test_r2 = r2_score(y_test, y_pred_ridge_test)
ridge_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_ridge_train))
ridge_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))

print(f'Train R2: {ridge_train_r2:.4f} | RMSE: £{ridge_train_rmse:.2f}')
print(f'Test R2:  {ridge_test_r2:.4f} | RMSE: £{ridge_test_rmse:.2f}')
print(f'Overfit:  {ridge_train_r2 - ridge_test_r2:.4f}')
print('')

# 1. FIXED RANDOM FOREST (reduced overfitting)
print('MODEL 1: Random Forest (Tuned to Reduce Overfitting)')


rf_tuned = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,           # Reduced from 20
    min_samples_split=20,   # Increased from 10
    min_samples_leaf=10,    # Added
    max_features='sqrt',    # Added
    random_state=42,
    n_jobs=-1
)

print('Training...')
rf_tuned.fit(X_train, y_train)

y_pred_rf_train = rf_tuned.predict(X_train)
y_pred_rf_test = rf_tuned.predict(X_test)

rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

print(f'Train R2: {rf_train_r2:.4f} | RMSE: £{rf_train_rmse:.2f}')
print(f'Test R2:  {rf_test_r2:.4f} | RMSE: £{rf_test_rmse:.2f}')
print(f'Overfit:  {rf_train_r2 - rf_test_r2:.4f}')
print('')

# Cross-validation
print('Cross-validating (5-fold)...')
rf_cv_scores = cross_val_score(rf_tuned, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
print(f'CV R2: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})')
print('')

# 2. XGBOOST (Initial)
print('MODEL 2: XGBoost (Initial Parameters)')
print('-' * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)

print('Training...')
xgb_model.fit(X_train, y_train)

y_pred_xgb_train = xgb_model.predict(X_train)
y_pred_xgb_test = xgb_model.predict(X_test)

xgb_train_r2 = r2_score(y_train, y_pred_xgb_train)
xgb_test_r2 = r2_score(y_test, y_pred_xgb_test)
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_xgb_train))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))

print(f'Train R2: {xgb_train_r2:.4f} | RMSE: £{xgb_train_rmse:.2f}')
print(f'Test R2:  {xgb_test_r2:.4f} | RMSE: £{xgb_test_rmse:.2f}')
print(f'Overfit:  {xgb_train_r2 - xgb_test_r2:.4f}')
print('')

# Cross-validation
print('Cross-validating (5-fold)...')
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5,
                                 scoring='r2', n_jobs=-1)
print(f'CV R2: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})')
print('')

# 3. XGBOOST HYPERPARAMETER TUNING
print('MODEL 3: XGBoost (Hyperparameter Tuning)')

print('Running GridSearchCV (this may take a few minutes)...')

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f'\nBest parameters: {grid_search.best_params_}')
print(f'Best CV R2: {grid_search.best_score_:.4f}')

xgb_best = grid_search.best_estimator_

y_pred_xgb_best_train = xgb_best.predict(X_train)
y_pred_xgb_best_test = xgb_best.predict(X_test)

xgb_best_train_r2 = r2_score(y_train, y_pred_xgb_best_train)
xgb_best_test_r2 = r2_score(y_test, y_pred_xgb_best_test)
xgb_best_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_xgb_best_train))
xgb_best_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_best_test))

print(f'\nTrain R2: {xgb_best_train_r2:.4f} | RMSE: £{xgb_best_train_rmse:.2f}')
print(f'Test R2:  {xgb_best_test_r2:.4f} | RMSE: £{xgb_best_test_rmse:.2f}')
print(f'Overfit:  {xgb_best_train_r2 - xgb_best_test_r2:.4f}')
print('')

# 4. ENSEMBLE (Voting)
print('MODEL 4: Ensemble (Voting)')


ensemble = VotingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=10.0)),
        ('rf', rf_tuned),
        ('xgb', xgb_best)
    ],
    weights=[1, 2, 2]  # Give more weight to tree models
)

print('Training ensemble...')
# Need to handle Ridge with scaling
X_train_ens = X_train.copy()
X_test_ens = X_test.copy()

ensemble_custom = VotingRegressor(
    estimators=[
        ('rf', rf_tuned),
        ('xgb', xgb_best)
    ]
)

ensemble_custom.fit(X_train_ens, y_train)

y_pred_ens_train = ensemble_custom.predict(X_train_ens)
y_pred_ens_test = ensemble_custom.predict(X_test_ens)

# Weighted with Ridge
ridge_weight = 0.2
rf_weight = 0.4
xgb_weight = 0.4

y_pred_ens_train_weighted = (
    ridge_weight * y_pred_ridge_train +
    rf_weight * y_pred_rf_train +
    xgb_weight * y_pred_xgb_best_train
)

y_pred_ens_test_weighted = (
    ridge_weight * y_pred_ridge_test +
    rf_weight * y_pred_rf_test +
    xgb_weight * y_pred_xgb_best_test
)

ens_train_r2 = r2_score(y_train, y_pred_ens_train_weighted)
ens_test_r2 = r2_score(y_test, y_pred_ens_test_weighted)
ens_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_ens_train_weighted))
ens_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ens_test_weighted))

print(f'Train R2: {ens_train_r2:.4f} | RMSE: £{ens_train_rmse:.2f}')
print(f'Test R2:  {ens_test_r2:.4f} | RMSE: £{ens_test_rmse:.2f}')
print(f'Overfit:  {ens_train_r2 - ens_test_r2:.4f}')
print('')

# RESULTS COMPARISON

print('RESULTS COMPARISON')


results = pd.DataFrame({
    'Model': ['Ridge', 'RF_Tuned', 'XGBoost', 'XGB_Best', 'Ensemble'],
    'Train_R2': [ridge_train_r2, rf_train_r2, xgb_train_r2, xgb_best_train_r2, ens_train_r2],
    'Test_R2': [ridge_test_r2, rf_test_r2, xgb_test_r2, xgb_best_test_r2, ens_test_r2],
    'Train_RMSE': [ridge_train_rmse, rf_train_rmse, xgb_train_rmse, xgb_best_train_rmse, ens_train_rmse],
    'Test_RMSE': [ridge_test_rmse, rf_test_rmse, xgb_test_rmse, xgb_best_test_rmse, ens_test_rmse],
    'Overfit': [
        ridge_train_r2 - ridge_test_r2,
        rf_train_r2 - rf_test_r2,
        xgb_train_r2 - xgb_test_r2,
        xgb_best_train_r2 - xgb_best_test_r2,
        ens_train_r2 - ens_test_r2
    ]
})

results['Test_MAE'] = [
    mean_absolute_error(y_test, y_pred_ridge_test),
    mean_absolute_error(y_test, y_pred_rf_test),
    mean_absolute_error(y_test, y_pred_xgb_test),
    mean_absolute_error(y_test, y_pred_xgb_best_test),
    mean_absolute_error(y_test, y_pred_ens_test_weighted)
]

print(results.to_string(index=False))
print('')

best_idx = results['Test_R2'].idxmax()
print(f'BEST MODEL: {results.loc[best_idx, "Model"]}')
print(f'  Test R2: {results.loc[best_idx, "Test_R2"]:.4f}')
print(f'  Test RMSE: £{results.loc[best_idx, "Test_RMSE"]:.2f}')
print(f'  Test MAE: £{results.loc[best_idx, "Test_MAE"]:.2f}')
print(f'  Overfit: {results.loc[best_idx, "Overfit"]:.4f}')
print('')

# FEATURE IMPORTANCE (XGBoost Best)
print('TOP 20 FEATURE IMPORTANCES (XGBoost Best)')


feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=False)

print(feat_imp.head(20).to_string(index=False))
print('')

# Identify low-value features
low_value = feat_imp[feat_imp['importance'] < 0.01]
print(f'Features with <1% importance: {len(low_value)}')
if len(low_value) > 0:
    print('Candidates for removal:')
    print(low_value['feature'].tolist())
print('')

# Save results
results.to_csv('results/model_comparison.csv', index=False)
feat_imp.to_csv('results/feature_importance_xgb.csv', index=False)

# VISUALIZATION
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# R2 comparison
ax = axes[0, 0]
x_pos = np.arange(len(results))
width = 0.35
ax.bar(x_pos - width/2, results['Train_R2'], width, label='Train', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, results['Test_R2'], width, label='Test', color='coral', alpha=0.7)
ax.set_ylabel('R2 Score')
ax.set_title('Model Comparison - R2')
ax.set_xticks(x_pos)
ax.set_xticklabels(results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# RMSE comparison
ax = axes[0, 1]
ax.bar(x_pos - width/2, results['Train_RMSE'], width, label='Train', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, results['Test_RMSE'], width, label='Test', color='coral', alpha=0.7)
ax.set_ylabel('RMSE (GBP)')
ax.set_title('Model Comparison - RMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Overfitting comparison
ax = axes[0, 2]
colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in results['Overfit']]
ax.bar(x_pos, results['Overfit'], color=colors, alpha=0.7)
ax.axhline(y=0.1, color='green', linestyle='--', label='Good (<0.1)')
ax.axhline(y=0.2, color='orange', linestyle='--', label='Moderate (<0.2)')
ax.set_ylabel('Overfit (Train R2 - Test R2)')
ax.set_title('Overfitting Analysis')
ax.set_xticks(x_pos)
ax.set_xticklabels(results['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Feature importance
ax = axes[1, 0]
top15 = feat_imp.head(15).sort_values('importance')
ax.barh(range(len(top15)), top15['importance'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Top 15 Features (XGBoost)')
ax.grid(axis='x', alpha=0.3)

# Predictions vs Actual (Best Model)
best_model_name = results.loc[best_idx, 'Model']
if best_model_name == 'Ridge':
    y_pred_best = y_pred_ridge_test
elif best_model_name == 'RF_Tuned':
    y_pred_best = y_pred_rf_test
elif best_model_name == 'XGBoost':
    y_pred_best = y_pred_xgb_test
elif best_model_name == 'XGB_Best':
    y_pred_best = y_pred_xgb_best_test
else:
    y_pred_best = y_pred_ens_test_weighted

ax = axes[1, 1]
sample_size = min(1000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test.iloc[idx], y_pred_best[idx], alpha=0.3, s=10, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price (GBP)')
ax.set_ylabel('Predicted Price (GBP)')
ax.set_title(f'Predictions vs Actual ({best_model_name})')
ax.grid(alpha=0.3)

# Residuals
ax = axes[1, 2]
residuals = y_test.values - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.3, s=10, color='steelblue')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Price (GBP)')
ax.set_ylabel('Residuals (GBP)')
ax.set_title(f'Residual Plot ({best_model_name})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_comparison_advanced.png', dpi=150)
print('Saved: model_comparison_advanced.png')
print('')

print('ANALYSIS COMPLETE')


