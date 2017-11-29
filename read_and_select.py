import pandas as pd
import os
from ast import literal_eval


def read_and_select(cols_to_drop = [], pick_doc2vec=True):

    train_df = pd.read_csv(os.path.join('data', 'train_df.csv'), header=0)
    test_df = pd.read_csv(os.path.join('data', 'test_df.csv'), header=0)

    if pick_doc2vec:

        d2v_train_df = pd.read_csv(os.path.join('data', 'd2v_train_df.csv'), header=0)
        d2v_test_df = pd.read_csv(os.path.join('data', 'd2v_test_df.csv'), header=0)

        train_df = pd.merge(train_df, d2v_train_df, on="id")
        test_df = pd.merge(test_df, d2v_test_df, on="id")

    print(train_df.columns.values)

    for feature_drop in cols_to_drop:
        train_df = train_df.drop(feature_drop, axis=1)
        test_df = test_df.drop(feature_drop, axis=1)

    print(train_df.head(2))
    print("\n")
    print(test_df.head(2))

    return train_df, test_df

if __name__ == '__main__':
    train_df, test_df = read_and_select(['text', 'id'])