import pandas as pd
import numpy as np

# 1. --- Create Sample Data (replace with your actual data) ---
# Assuming you already have 'df' and 'y_pred'
# For example:

# data = {'sample_id': [101, 102, 103, 104, 105],
#         'feature_1': [0.5, 0.2, 0.8, 0.1, 0.9]}


y_pred = np.array([1, 0, 1, 0, 1])


# 2. --- Create a new DataFrame for submission ---
# Use a dictionary where keys are the column names and values are your data
submission_df = pd.DataFrame({
    'sample_id': df['sample_id'],
    'y_pred': y_pred
})


# 3. --- Save to a CSV file ---
# The `index=False` argument prevents pandas from writing the DataFrame index as a column
submission_df.to_csv('submission.csv', index=False)

print("'submission.csv' created successfully!")