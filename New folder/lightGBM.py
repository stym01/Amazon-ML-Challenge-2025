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
import lightgbm as lgb

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
# You can set hyperparameters directly here if you wish.\

clf = lgb.LGBMRegressor()

# Train the model
clf.fit(X_train_std, y_train)

y_pred = clf.predict(X_test_std)

# --- Evaluate the model ---
# Use regression metrics instead of accuracy and confusion matrix
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)*100

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}%")






def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The calculated SMAPE value.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Handle cases where denominator might be zero to avoid division by zero
    # If both y_true and y_pred are zero, the error is 0.
    # If one is zero and the other is not, the error is 200%.
    # This implementation handles the case where both are zero by setting the contribution to 0.
    # For other cases with zero in the denominator, it effectively becomes 200%.
    
    # Create a mask for non-zero denominators
    non_zero_denominator_mask = denominator != 0
    
    # Calculate the percentage error only for non-zero denominators
    percentage_error = np.zeros_like(numerator, dtype=float)
    percentage_error[non_zero_denominator_mask] = numerator[non_zero_denominator_mask] / denominator[non_zero_denominator_mask]
    
    return np.mean(percentage_error) * 100


smape_score = smape(y_test, y_pred)
print(f"SMAPE: {smape_score:.2f}%")