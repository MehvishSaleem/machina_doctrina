import pandas as pd
import os
import string

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

stop_word_list = []
lemm = WordNetLemmatizer()

def read_stop_word_list(stop_words_file, data_dir="data"):
    global stop_word_list
    stop_words_path = os.path.join(data_dir, stop_words_file)

    with open(stop_words_path, 'r') as f:
        for x in f:
            stop_word_list.append(x.strip())

    stop_word_list = set(stop_word_list)

    # print(stop_word_list)

def read_2_df(filename):
    return pd.read_csv(filename,
                       sep=",",
                       header=0)

def text_processing(text):

    removed_stop = []

    for w in text.lower().split():
        dont_add = False
        for k in stop_word_list:
            if w == k:
                dont_add = True
                break
        if not dont_add:
            removed_stop.append(w)

    text = ' '.join(removed_stop)

    ## print("\tConverted text to lowercase ...")
    ## print("\tRemoved stop words ...")

    for c in string.punctuation:
        # text = text.replace(c, " "+c+" ")
        text = text.replace(c, "")

    ## print("\tRemoved punctuations ...")

    ## print("\tLemmatizing words ...")

    return ' '.join([lemm.lemmatize(x) for x in text.split()])

def generate_count_features(df):
    df['num_of_words'] = df['text'].apply(lambda x: len(x.split()))
    df['num_of_unique_words'] = df['text'].apply(lambda x: len(set(x.split())))
    df['num_of_characters'] = df['text'].apply(lambda x: len(x))
    df['num_of_stopwords'] = df['text'].apply(lambda x: len([y for y in x.lower().split() if y in stop_word_list]))
    df['num_of_punctuations'] = df['text'].apply(lambda x: len([y for y in x if y in string.punctuation]))
    df['num_of_uppercase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.isupper()]))
    df['num_of_titlecase'] = df['text'].apply(lambda x: len([y for y in x.split() if y.istitle()]))

    ## print("\tGenerated simple count features ...")

    # NLP based features
    df['num_of_adjectives'] = df['text'].apply(
        lambda x: len([y[0] for y in pos_tag(x.split()) if y[1] in ['JJ', 'JJR', 'JJS']]))
    df['num_of_nouns'] = df['text'].apply(
        lambda x: len([y[0] for y in pos_tag(x.split()) if y[1] in ['NN', 'NNS', 'NNP', 'NNPS']]))
    df['num_of_verbs'] = df['text'].apply(
        lambda x: len([y[0] for y in pos_tag(x.split()) if y[1] in ['VB', 'VBD', 'VBG', 'VBP', 'VBN', 'VBZ']]))

    ## print("\tGenerated NLP Based Features ...")

    return df

def generate_Doc2Vec(train_df, test_df):

    sentences = []

    for item_no, line in enumerate(train_df['processed_text'].values.tolist()):
        sentences.append(TaggedDocument(line, [item_no]))

    print("Training Doc2Vec ...")

    feat_vec_size = 30
    context_window = 50
    seed = 42
    min_count = 1
    alpha = 0.5
    max_iter = 1

    # BUILD MODEL
    model = Doc2Vec(documents=sentences,
                    alpha=alpha,  # initial learning rate
                    seed=seed,
                    min_count=min_count,  # ignore words with freq less than min_count
                    window=context_window,  # the number of words before and after to be used as context
                    size=feat_vec_size,  # is the dimensionality of the feature vector
                    iter=max_iter)

    new_train = train_df.copy()
    new_test = test_df.copy()

    colnames = ['d2v_feat_'+str(i) for i in range(feat_vec_size)]

    doc2vec_train = new_train['processed_text'].apply(lambda x: model.infer_vector(x))
    doc2vec_test = new_test['processed_text'].apply(lambda x: model.infer_vector(x))

    new_train[colnames] = pd.DataFrame(doc2vec_train.values.tolist(), index=new_train.index)
    new_test[colnames] = pd.DataFrame(doc2vec_test.values.tolist(), index=new_test.index)

    colnames += ["id"]

    return new_train[colnames], new_test[colnames]

def preprocess(data_dir="data"):
    train_data = os.path.join(data_dir, "train.csv")
    test_data = os.path.join(data_dir, "test.csv")

    train_df = read_2_df(train_data)
    test_df = read_2_df(test_data)

    print("Dataframes have been read ...\n\n")

    # Process Text
    train_df['processed_text'] = train_df['text'].apply(text_processing)
    test_df['processed_text'] = test_df['text'].apply(text_processing)

    print("Text processing done ...\n\n")

    # Generate Counting Based Features
    train_df = generate_count_features(train_df)
    test_df = generate_count_features(test_df)

    print("Count Based Features generated ...\n\n")

    # Doc2Vec Features
    d2v_train_df, d2v_test_df = generate_Doc2Vec(train_df, test_df)

    print("Doc2Vec Features added ...\n\n")

    # Convert author to one-hot encoding
    one_hot_authors = pd.get_dummies(train_df['author'])
    train_df = pd.concat([train_df, one_hot_authors], axis=1)

    print("Target class (author) has been one hot encoded ...\n\n")

    # Drop Author
    train_df = train_df.drop("author", axis=1)

    train_df.to_csv(os.path.join('data', 'train_df.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'test_df.csv'), index=False)

    d2v_train_df.to_csv(os.path.join('data', 'd2v_train_df.csv'), index=False)
    d2v_test_df.to_csv(os.path.join('data', 'd2v_test_df.csv'), index=False)

    print(train_df.head())
    print("\n")
    print(test_df.head())

    print("Processed dataframes saved as CSVs!")

if __name__ == '__main__':

    stop_words_file = "ranksnl_large.txt"  # Source: https://github.com/igorbrigadir/stopwords
    read_stop_word_list(stop_words_file)

    preprocess()