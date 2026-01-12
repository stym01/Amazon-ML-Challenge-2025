import pandas as pd

# Read only first 500 rows
df = pd.read_csv('train.csv', nrows=100)

# Save those rows to a new CSV
df.to_csv('first_100_rows.csv', index=False)