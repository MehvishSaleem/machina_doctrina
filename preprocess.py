import pandas as pd
import os

def read_2_df(filename):
    return pd.read_csv(filename,
                       sep=",",
                       header=0)

def text_processing(text):
    return text

def preprocess(data_dir="data"):
    train_data = os.path.join(data_dir, "train.csv")
    test_data = os.path.join(data_dir, "test.csv")

    train_df = read_2_df(train_data)
    test_df = read_2_df(test_data)

    # Drop ID column
    train_df = train_df.drop("id", axis=1)
    test_df = test_df.drop("id", axis=1)

    # Convert author to one-hot encoding
    one_hot_authors = pd.get_dummies(train_df['author'])
    train_df = pd.concat([train_df, one_hot_authors], axis=1)

    # Drop Author
    train_df = train_df.drop("author", axis=1)

    # Process Text
    train_df['processed_text'] = train_df['text'].apply(text_processing)
    test_df['processed_text'] = test_df['text'].apply(text_processing)

    print(train_df.head())
    print(test_df.head())

if __name__ == '__main__':
    preprocess()