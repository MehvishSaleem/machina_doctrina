import pandas as pd
import os
import string

import numpy as np
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

stop_word_list = []
lemm = WordNetLemmatizer()

def read_stop_word_list(stop_words_file, data_dir="data"):
    stop_words_path = os.path.join(data_dir, stop_words_file)

    with open(stop_words_path, 'r') as f:
        for x in f:
            stop_word_list.append(x.strip())

    # print(stop_word_list)

def read_2_df(filename):
    return pd.read_csv(filename,
                       sep=",",
                       header=0)

def text_processing(text):

    text = ' '.join([w for w in text.lower().split() if not w in stop_word_list])

    for c in string.punctuation:
        # text = text.replace(c, " "+c+" ")
        text = text.replace(c, "")

    for w in text.split():
        if w != lemm.lemmatize(w):
            print(w, "->", lemm.lemmatize(w))
    return ' '.join([lemm.lemmatize(x) for x in text.split()])

def generate_count_features(df):
    df['num_of_words'] = df['text'].apply(lambda x: len(x.split()))
    df['num_of_unique_words'] = df['text'].apply(lambda x: len(set(x.split())))
    df['num_of_characters'] = df['text'].apply(lambda x: len(x))
    df['num_of_stopwords'] = df['text'].apply(lambda x: len([y for y in x.lower().split() if y in stop_word_list]))
    df['num_of_punctuations'] = df['text'].apply(lambda x: len([y for y in x if y in string.punctuation]))
    df['num_of_uppercase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.isupper()]))
    df['num_of_titlecase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.istitle()]))

    return df

# def build_model(input_dims, embedding_dims=20, optimiser="adam"):
#     model = Sequential()
#     model.add(Embedding(input_dim=input_dims, output_dim=embedding_dims))
#     model.add(GlobalAveragePooling1D())
#     model.add(Dense(3, activation="softmax"))
#     model.compile(loss="categorical_crossentropy",
#                  optimizer=optimiser,
#                  metrics=["accuracy"])
#     return model

def preprocess(data_dir="data"):
    train_data = os.path.join(data_dir, "train.csv")
    test_data = os.path.join(data_dir, "test.csv")

    train_df = read_2_df(train_data)
    test_df = read_2_df(test_data)

    # Drop ID column
    train_df = train_df.drop("id", axis=1)
    test_df = test_df.drop("id", axis=1)

    # Process Text
    train_df['processed_text'] = train_df['text'].apply(text_processing)
    test_df['processed_text'] = test_df['text'].apply(text_processing)

    # Generate Counting Based Features
    train_df = generate_count_features(train_df)
    test_df = generate_count_features(test_df)


    # .........

    # Xtrain, Xval, ytrain, yval = train_test_split(train_df["processed_text"].values, y)
    #
    # tokeniser = Tokenizer()
    # tokeniser.fit_on_texts(Xtrain)
    #
    # def tokenise(x, tokeniser, maxlen=256):
    #     return pad_sequences(
    #         sequences=tokeniser.texts_to_sequences(x),
    #         maxlen=maxlen)
    #
    # X_train_tokens, X_val_tokens, X_test_tokens = (tokenise(x, tokeniser)
    #                                                for x in (Xtrain, Xval, test_df["processed_text"].values))
    #
    # input_dim = np.max(X_train_tokens) + 1
    # embedding_dims = 15
    #
    # epochs = 50
    # model = build_model(input_dim, embedding_dims)
    #
    # data = model.fit(X_train_tokens, ytrain, batch_size=16, validation_data=(X_val_tokens, yval),
    #                  epochs=epochs, callbacks=[EarlyStopping(patience=2, monitor="val_loss")])


    # Convert author to one-hot encoding
    one_hot_authors = pd.get_dummies(train_df['author'])
    train_df = pd.concat([train_df, one_hot_authors], axis=1)

    # Drop Author
    train_df = train_df.drop("author", axis=1)

    train_df.to_pickle(os.path.join('data', 'train_df.pkl'))
    test_df.to_pickle(os.path.join('data', 'test_df.pkl'))

if __name__ == '__main__':

    stop_words_file = "ranksnl_large.txt"  # Source: https://github.com/igorbrigadir/stopwords
    read_stop_word_list(stop_words_file)

    preprocess()