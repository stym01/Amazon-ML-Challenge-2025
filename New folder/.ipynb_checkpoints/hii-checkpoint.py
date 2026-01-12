import pandas as pd
import numpy as np
df=pd.read_csv('final_embedded_data.csv')



Q1_quality = df['Value'].quantile(0.25)
Q3_quality = df['Value'].quantile(0.75)
IQR_quality = Q3_quality - Q1_quality

lower_bound_quality = np.float64(Q1_quality - 1.5 * IQR_quality)
upper_bound_quality = np.float64(Q3_quality + 1.5 * IQR_quality)


outliers1 = df[(df['Value'] < lower_bound_quality) | (df['Value'] > upper_bound_quality)]

print(outliers1)

df_cleaned=df[(df['Value'] > lower_bound_quality) & (df['Value'] < upper_bound_quality)]


print(df_cleaned.shape)















# import pandas as pd

# df=pd.read_csv('final_embedded_data.csv')

# print(df.isnull().sum())

# # Returns a new DataFrame with null rows removed
# new_df = df.dropna()

# print(new_df.isnull().sum())

# print(new_df.shape)

