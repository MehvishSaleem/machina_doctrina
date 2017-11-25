import numpy as np
import pandas as pd
import os

def read_2_df(filename):
    return pd.read_csv(filename,
                       sep=",",
                       header=0)

def preprocess(data_dir="./data"):
    train_data = os.path.join(data_dir, "train.csv")
    test_data = os.path.join(data_dir, "test.csv")

    train_df = read_2_df(train_data)
    test_df = read_2_df(test_data)

    train_df = train_df.drop("id", axis=1)
    test_df = test_df.drop("id", axis=1)

    print(train_df.head())
    print(test_df.head())

if __name__ == '__main__':
    preprocess()