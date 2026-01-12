import pandas as pd
import pickle
import numpy as np
filename = '/kaggle/input/modelml/fitted_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
df1=pd.read_csv('/kaggle/working/final_embedded_data.csv')
df=pd.read_csv('/kaggle/working/final_embedded_data.csv')
df = df.drop(columns=['sample_id','image_link'], axis=1)
X_new = df.iloc[:,:].values
y_pred = loaded_model.predict(X_new)

submission_df = pd.DataFrame({
    'sample_id': df1['sample_id'],
    'price': y_pred
})

submission_df.to_csv('submission.csv', index=False)
print("'submission.csv' created successfully!")