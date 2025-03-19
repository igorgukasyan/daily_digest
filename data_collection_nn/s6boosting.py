import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import metrics
import matplotlib.pylab as plt
import pickle
# -------------------------------
# Load Data
# -------------------------------
test = pd.read_csv('./test_vectors.csv')
train = pd.read_csv('./train_vectors.csv')

# Define target variable and predictors
target = 'score'
X = train.iloc[:, :-1]
y = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# -------------------------------
# Base model with initial parameters
# -------------------------------
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ----------------------------------------------
# GROUP 1: Tune max_depth and min_child_weight
# ----------------------------------------------
param_grid1 = {
    'max_depth': [4, 5, 6, 7, 9],
    'min_child_weight': [1, 3, 5, 7]
}

grid1 = GridSearchCV(
    estimator=model,
    param_grid=param_grid1,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid1.fit(X, y)
print("Best parameters for Group 1 (max_depth, min_child_weight):", grid1.best_params_)
model.set_params(**grid1.best_params_)

# ----------------------------------------------
# GROUP 2: Tune subsample and colsample_bytree
# ----------------------------------------------
param_grid2 = {
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

grid2 = GridSearchCV(
    estimator=model,
    param_grid=param_grid2,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid2.fit(X, y)
print("Best parameters for Group 2 (subsample, colsample_bytree):", grid2.best_params_)
model.set_params(**grid2.best_params_)

# ----------------------------------------------------
# GROUP 3: Tune learning_rate and num_boost_round
# ----------------------------------------------------
param_grid3 = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [1000, 1500, 2000, 2500]
}

grid3 = GridSearchCV(
    estimator=model,
    param_grid=param_grid3,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid3.fit(X, y)
print("Best parameters for Group 3 (learning_rate, num_boost_round):", grid3.best_params_)
model.set_params(**grid3.best_params_)

# ----------------------------------------------------
# GROUP 4: Tune gamma
# ----------------------------------------------------
param_grid4 = {
    'gamma': [0, 1, 3, 5]
}

grid4 = GridSearchCV(
    estimator=model,
    param_grid=param_grid4,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid4.fit(X, y)
print("Best parameters for Group 4 (gamma):", grid4.best_params_)
model.set_params(**grid4.best_params_)

# -------------------------------
# Final Model Training & Evaluation
# -------------------------------
model.fit(X, y)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_train = model.predict(X)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Test RMSE:", rmse)
r2 = metrics.r2_score(y_test, y_pred)
print("Test R2:", r2)
r2 = metrics.r2_score(y, y_pred_train)
print("Train R2:", r2)
mae = metrics.mean_absolute_error(y_test, y_pred)
print("Test mae:", mae)
# Optional: Plot feature importance
xgb.plot_importance(model)
plt.show()

with open('xgb_model.pkl', 'wb') as f: 
    pickle.dump(model, f)