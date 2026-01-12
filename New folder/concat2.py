import pandas as pd
import numpy as np


df1 = pd.read_csv('final_embedded_data.csv')

df2 = pd.read_csv('vit_multilayer_features_raw.csv')

merged_df = pd.merge(df1, df2, on='sample_id')


merged_df = merged_df.drop('image_link',axis=1)
merged_df.head()

merged_df.to_csv('final_merged_train_raw_data.csv')