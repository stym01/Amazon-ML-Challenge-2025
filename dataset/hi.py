import pandas as pd
df=pd.read_csv("final_standardized_train_data.csv")
print(df["Unit"].value_counts())
