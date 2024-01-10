import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

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



#pipeline for CatBoostRegressor
ctb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', CatBoostRegressor(iterations=250, depth=6, learning_rate=0.11, loss_function='RMSE', random_seed=42))
])

ctb_pipeline.fit(train_data,train_labels)
ctb_pipeline_predictions = ctb_pipeline.predict(test_data)
ctb_mse = mean_squared_error(test_labels,ctb_pipeline_predictions)
print((f"Mean Squared Error: {ctb_mse}"))


#predicting on nan values
ctb_pipeline_predictions_nan = ctb_pipeline.predict(nan_data)
print(ctb_pipeline_predictions_nan)