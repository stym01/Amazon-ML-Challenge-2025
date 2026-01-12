import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb


data = pd.read_csv('final_embedded_data.csv')
data = data.drop(columns=['sample_id','image_link'], axis=1)
print(data.shape)
X = data.iloc[:,:-1].values
y = data["price"].values


sc = StandardScaler()
X_train_std = sc.fit_transform(X)


clf = lgb.LGBMRegressor()


clf.fit(X_train_std, y)

import pickle

filename = 'fitted_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)