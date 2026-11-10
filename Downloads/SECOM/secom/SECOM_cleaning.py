import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# 1. Load balanced training data
train_df = pd.read_csv('secom_train_balanced.csv')

# Convert to numpy arrays and ensure y is 1D
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].to_numpy()

# 2. Initialize Random Forest estimator
rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)

# 3. Initialize Boruta selector (n_estimators='auto' or set explicitly)
boruta_selector = BorutaPy(
    estimator=rf, n_estimators='auto', max_iter=50, verbose=2, random_state=42
)

# 4. Fit Boruta
boruta_selector.fit(X, y)

# 5. Extract selected features (including support and weak support / tentative if desired)
selected_mask = boruta_selector.support_ | boruta_selector.support_weak_
selected_features = train_df.iloc[:, :-1].columns[selected_mask].tolist()
print(f'Selected {len(selected_features)} features using Boruta.')