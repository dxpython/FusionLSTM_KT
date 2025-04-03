import numpy as np
import pandas as pd


time_stamp = np.load('../dataset/time_stamp.npy')
w_228 = pd.read_csv('../dataset/W_228.csv')
v_228 = pd.read_csv('../dataset/V_228.csv')

print(time_stamp[:10])
print("=======================我是分割线==============================")
print(w_228.head())
print("=======================我是分割线==============================")
print(v_228.head())
