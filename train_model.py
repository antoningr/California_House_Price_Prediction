# train_model.py

import numpy as np
import pandas as pd
import datetime
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

# =======================================
# 1. Load Dataset
# =======================================
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# =======================================
# 2. Feature Engineering
# =======================================
df["Income_Age"] = df["MedInc"] * df["HouseAge"]
df["Rooms_per_Occup"] = df["AveRooms"] / (df["AveOccup"] + 1)  # avoid division by zero
df["Log_Pop"] = np.log(df["Population"] + 1)
df["Near_Coast"] = (df["Longitude"] > -118).astype(int)  # convert boolean to int for modeling

# =======================================
# 3. Feature Selection
# =======================================
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kbest = SelectKBest(score_func=f_regression, k=9)
X_kbest = kbest.fit_transform(X_scaled, y)
selected_features = X.columns[kbest.get_support()]
X = df[selected_features]

print(f"Top 9 selected features by SelectKBest: {list(selected_features)}")

# =======================================
# 4. Split Dataset
# =======================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================================
# 5. Define Models
# =======================================
models = {
    "Linear Regression"   : LinearRegression(),
    "Ridge Regression"    : Ridge(alpha=1.0),
    "Lasso Regression"    : Lasso(alpha=0.1),
    "ElasticNet"          : ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest"       : RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Extra Trees"         : ExtraTreesRegressor(n_estimators=100, random_state=42),
    "AdaBoost"            : AdaBoostRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "Decision Tree"       : DecisionTreeRegressor(random_state=42),
    "Bayesian Ridge"      : BayesianRidge(),
    "Huber Regressor"     : HuberRegressor(),
    "KNN Regressor"       : KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF Kernel)"    : SVR(kernel='rbf'),
    "XGBoost"             : XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM"            : LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    "CatBoost"            : CatBoostRegressor(verbose=0, random_state=42),
    "MLP Regressor"       : MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

models["Stacking RF+GB+XGB"] = StackingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=50, random_state=42, verbosity=0))
    ],
    final_estimator=Ridge(),
    passthrough=True,
    cv=5,
    n_jobs=-1
)

models["Stacking Linear+Tree+Cat"] = StackingRegressor(
    estimators=[
        ("lasso", Lasso(alpha=0.1)),
        ("dt", DecisionTreeRegressor(max_depth=10)),
        ("knn", KNeighborsRegressor(n_neighbors=7)),
        ("cat", CatBoostRegressor(verbose=0, random_state=42))
    ],
    final_estimator=LGBMRegressor(n_estimators=50, random_state=42, verbose=-1),
    passthrough=True,
    cv=5,
    n_jobs=-1
)

models["Stacking LR+DT+Cat"] = StackingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("dt", DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("cat", CatBoostRegressor(verbose=0, random_state=42))
    ],
    final_estimator=LGBMRegressor(n_estimators=50, random_state=42, verbose=-1),
    passthrough=True,
    cv=5,
    n_jobs=-1
)

models["Stacking LR+SVR+MLP"] = StackingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("svr", SVR(kernel='rbf')),
        ("mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=300, random_state=42))
    ],
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
    passthrough=True,
    cv=5,
    n_jobs=-1
)

# =======================================
# 6. Train and Evaluate All Models
# =======================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}:")
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2', n_jobs=-1)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2, "CV_R2_Mean": scores.mean()}
    print(f"MAE   : {mae:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"RMSE  : {np.sqrt(mse):.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"CV R² : {scores.mean():.4f}")

results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
results_df.style.background_gradient(axis=0, cmap="YlGnBu")

# =======================================
# 7. Select Best Model
# =======================================
best_model_name = results_df.index[0]
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")

# =======================================
# 8. Save the Best Model
# =======================================
now = datetime.datetime.now().strftime("%Y_%m_%d")
filename = f"best_model_{best_model_name.replace(' ', '_')}_{now}.pkl"
with open(filename, 'wb') as f:
    pickle.dump((scaler, best_model), f)
print(f"Best model saved as: {filename}")
