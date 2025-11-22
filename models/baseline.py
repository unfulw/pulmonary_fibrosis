import numpy as np
from scipy.stats import loguniform, uniform
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.tabular_preprocessing import collapsed
from preprocessing.pid_split import train_ids, val_ids

X = collapsed.drop(columns=["dPercent", "y_FVC_last", "Patient"])
y = collapsed["y_FVC_last"]

# Split by patient IDs - filter using Patient column before dropping it
train_mask = collapsed["Patient"].isin(train_ids)
val_mask = collapsed["Patient"].isin(val_ids)

X_train = X[train_mask]
X_val = X[val_mask]
y_train = y[train_mask]
y_val = y[val_mask]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Initializing XGBoost model
cat_cols = ["Sex", "SmokingStatus"]
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

optim = XGBRegressor(
    tree_method="hist",
    enable_categorical=True,
    n_estimators=1000,        
    objective="reg:squarederror",
    n_jobs=-1
)

# Hyperparameter distributions for Randomized Search
param_dist = {
    "learning_rate": loguniform(1e-3, 2e-1),  # ~[0.001, 0.2]
    "max_depth":     [3,4,5,6,7,8],
    "min_child_weight": uniform(1, 5),        # [1,6)
    "subsample":     uniform(0.6, 0.4),       # [0.6,1.0)
    "colsample_bytree": uniform(0.6, 0.4),    # [0.6,1.0)
    "reg_lambda":    loguniform(1e-2, 1e2),   # [0.01,100]
    "reg_alpha":     loguniform(1e-3, 1e1),   # [0.001,10]
}

# --- CV setup ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

fit_params = {
    "eval_set": [(X_val, y_val)],    
    "verbose": False
}

search = RandomizedSearchCV(
    estimator=optim,
    param_distributions=param_dist,
    n_iter=60,                    
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=3244,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train, **fit_params)

best = search.best_estimator_
print("Best CV MAE:", -search.best_score_)
print("Best params:", search.best_params_)

# --- final eval on test set ---
y_pred = best.predict(X_val)
print("Test MAE:", mean_absolute_error(y_val, y_pred))

baseline = XGBRegressor(
    n_estimators=500,
    learning_rate=np.float64(0.005652520143262533),
    max_depth=3,
    colsample_bytree=np.float64(0.8906843740307104),
    min_child_weight=np.float64(4.581661879888676),
    reg_alpha=np.float64(0.005791400847430981),
    reg_lambda=np.float64(12.95318317933723),
    subsample=np.float64(0.8727952814245623),
    tree_method="hist",       # fast CPU training
    enable_categorical=True   # allows category dtype
)