import numpy as np
import pandas as pd


def load_data():
    time_stamp = np.load('/root/GNNitem/dataset/time_stamp.npy')
    w_228 = pd.read_csv('/root/GNNitem/dataset/W_228.csv')
    v_228 = pd.read_csv('/root/GNNitem/dataset/V_228.csv')
    return time_stamp, w_228, v_228


if __name__ == "__main__":
    time_stamp, w_228, v_228 = load_data()
    print("Time Stamp Data:")
    print(time_stamp[:10])
    print("=======================我是分割线==============================")
    print("W_228 Data:")
    print(w_228.head())
    print("=======================我是分割线==============================")
    print("V_228 Data:")
    print(v_228.head())
