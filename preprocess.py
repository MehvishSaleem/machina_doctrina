import pandas as pd
import os
import string
from gensim.models import Doc2Vec

stop_word_list = []

def read_stop_word_list(stop_words_file, data_dir="data"):
    stop_words_path = os.path.join(data_dir, stop_words_file)

    with open(stop_words_path, 'r') as f:
        for x in f:
            stop_word_list.append(x.strip())

    print(stop_word_list)

def read_2_df(filename):
    return pd.read_csv(filename,
                       sep=",",
                       header=0)

def text_processing(text):

    text = ' '.join([w for w in text.lower().split() if not w in stop_word_list])

    for c in string.punctuation:
        # text = text.replace(c, " "+c+" ")
        text = text.replace(c, "")
    return ' '.join(text.split())

def generate_count_features(df):
    df['num_of_words'] = df['text'].apply(lambda x: len(x.split()))
    df['num_of_unique_words'] = df['text'].apply(lambda x: len(set(x.split())))
    df['num_of_characters'] = df['text'].apply(lambda x: len(x))
    df['num_of_stopwords'] = df['text'].apply(lambda x: len([y for y in x.lower().split() if y in stop_word_list]))
    df['num_of_punctuations'] = df['text'].apply(lambda x: len([y for y in x if y in string.punctuation]))
    df['num_of_uppercase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.isupper()]))
    df['num_of_titlecase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.istitle()]))

    return df

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

    # Generate Counting Based Features
    train_df = generate_count_features(train_df)
    test_df = generate_count_features(test_df)

    # Generate Word

    print(train_df.head())
    print(test_df.head())

if __name__ == '__main__':

    stop_words_file = "ranksnl_large.txt"  # Source: https://github.com/igorbrigadir/stopwords
    read_stop_word_list(stop_words_file)

    preprocess()