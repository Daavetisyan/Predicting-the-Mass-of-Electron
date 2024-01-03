import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor

# Pipeline
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
set_config(display="diagram")  # make pipeline visible
set_config(transform_output="pandas")  # make transformers output pandas dataframe

# Models
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

# Metrics and evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import shap
#shap.initjs()

# Hyperparameter Tuning
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


data = pd.read_csv("dielectron.csv")
data = pd.DataFrame(data)
important_features = ["E1","pt1","E2","pt2","M"]
data_1 = data[important_features]
target = "M"
features = ["E1","pt1","E2","pt2"]
nan_mask = data.isnull().any(axis=1)
nan_data = data[nan_mask]
nan_data = nan_data[features]
data = data.dropna(subset="M")

train_data, test_data, train_labels, test_labels = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Initialize the XGBoost Regressor model
model = XGBRegressor(objective="reg:squarederror", random_state=42)

# Train the model
model.fit(train_data, train_labels)

# Make predictions on the test set
predictions = model.predict(test_data)

# Evaluate the model
mse = mean_squared_error(test_labels, predictions)
print(f"Mean Squared Error: {mse}")

# Visualize predictions vs actual values
plt.scatter(test_labels, predictions,c=test_labels, edgecolors='black')
plt.xlabel("Actual M values")
plt.ylabel("Predicted M values")
plt.title("Actual vs Predicted M values (XGBoost)")
plt.show()

predictions_1 = model.predict(nan_data)
print(predictions_1)

scaler = StandardScaler()
scaled_train_data = scaler.fit(train_data)
#scaled_train_labels = scaler.fit(train_labels)
scaled_test_data = scaler.fit(test_data)
#scaled_test_labels = scaler.fit(test_labels)


catboost_model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, loss_function='RMSE', random_seed=42)
catboost_model.fit(train_data,train_labels)
catboost_predictions = catboost_model.predict(test_data)

plt.scatter(test_labels,catboost_predictions)
plt.xlabel("Actual M values")
plt.ylabel("Predicted M values")
plt.title("Actual vs Predicted M values (CatBoost)")
plt.show()

mse_1 = mean_squared_error(test_labels,catboost_predictions)
print(f"Mean Squared Error: {mse_1}")


regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
regressor.fit(train_data,train_labels)
gradient_predictions = regressor.predict(test_data)

plt.scatter(test_labels,gradient_predictions)
plt.xlabel("Actual M values")
plt.ylabel("Predicted M values")
plt.title("Actual vs Predicted M values (GradientBoostingRegressor)")
plt.show()

mse_2 = mean_squared_error(test_labels,gradient_predictions)
print(f"Mean Squared Error: {mse_2}")


xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Step 1: Feature scaling
    ('regressor', XGBRegressor())          # Step 2: XGBRegressor
])

xgb_pipeline.fit(train_data,train_labels)
xgb_pipeline_predictions = xgb_pipeline.predict(test_data)
xgb_mse = mean_squared_error(test_labels,xgb_pipeline_predictions)
print(f"Mean Squared Error: {xgb_mse}")

ctb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', CatBoostRegressor())
])

ctb_pipeline.fit(train_data,train_labels)
ctb_pipeline_predictions = ctb_pipeline.predict(test_data)
ctb_mse = mean_squared_error(test_labels,ctb_pipeline_predictions)
print((f"Mean Squared Error: {ctb_mse}"))

ctb_pipeline_predictions_nan = ctb_pipeline.predict(nan_data)
print(ctb_pipeline_predictions_nan)