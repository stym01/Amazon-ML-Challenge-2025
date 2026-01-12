"""## Simple XGBoost Regressor"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Import metrics for regression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
# Import the XGBoost Regressor model
from xgboost import XGBRegressor

# --- Load your data ---
# IMPORTANT: Ensure your target column ("Label" in this case) contains continuous numerical values.
data = pd.read_csv('final_embedded_data.csv')
data = data.drop(columns=['sample_id','image_link'], axis=1)
print(data.shape)
X = data.iloc[:,:-1].values
y = data["price"].values

# --- Split data ---
# 'stratify' is not typically used for regression.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale features ---
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# --- Initialize and Train the XGBoost Regressor ---
# We are no longer using GridSearchCV.
# You can set hyperparameters directly here if you wish.
xgb_model = XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100,      # Example parameter
                         learning_rate=0.1,     # Example parameter
                         max_depth=5,           # Example parameter
                         eval_metric='rmse', 
                         random_state=42)

# Fit the model directly on the training data
xgb_model.fit(X_train_std, y_train)

# --- Make predictions on the test set ---
y_pred = xgb_model.predict(X_test_std)

# --- Evaluate the model ---
# Use regression metrics instead of accuracy and confusion matrix
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)*100

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}%")


