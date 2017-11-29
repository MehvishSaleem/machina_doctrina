import pandas as pd
import os

def read_and_select(cols_to_drop = []):

    train_df = pd.read_pickle(os.path.join('data', 'train_df.pkl'))
    test_df = pd.read_pickle(os.path.join('data', 'test_df.pkl'))

    print(train_df.columns.values)

    for feature_drop in cols_to_drop:
        train_df = train_df.drop(feature_drop, axis=1)
        test_df = test_df.drop(feature_drop, axis=1)

    print(train_df.head())
    print(test_df.head())

if __name__ == '__main__':
    read_and_select(['text'])