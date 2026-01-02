"""
Train and Save Final Model for Streamlit App
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb

print('Training Final Model for Deployment')
print('=' * 60)

# Load data
df = pd.read_csv('final_data/features_final.csv')
print(f'Loaded: {len(df):,} rows')

y = df['price_clean']
X = df.drop('price_clean', axis=1)

# Numeric only
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

# Train on ALL data (for deployment)
print('\nTraining on full dataset...')

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)
print('Training complete')

# Save model
with open('pkl/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\nSaved: xgb_model.pkl')

# Save feature names for reference
with open('pkl/model_features.txt', 'w') as f:
    f.write('Model Features (in order):\n\n')
    for i, feat in enumerate(X.columns, 1):
        f.write(f'{i}. {feat}\n')

print('Saved: model_features.txt')
print('\nModel ready for Streamlit app!')
print('Run: streamlit run streamlit_app.py')
