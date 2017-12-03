import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from gensim.models.word2vec import Word2Vec
import spacy
import string
import re
from read_and_select import *
from word2vec_models import *

nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stopwords = stopwords.words('english')

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs):
    texts = []
    counter = 1
    for doc in docs:
    	doc = doc.decode('utf-8')
        counter += 1
        doc = nlp(unicode(doc))
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

# Define function to preprocess text for a word2vec model
def process_text(docs):
    sentences = []
    counter = 1
    for doc in docs:
    	doc = doc.decode('utf-8')
        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
        doc = nlp(unicode(doc), disable=['tagger'])
        # Grab lemmatized form of words and make lowercase
        doc = " ".join([tok.lemma_.lower() for tok in doc])
        # Split into sentences based on punctuation
        doc = re.split("[\.?!;] ", doc)
        # Remove commas, periods, and other punctuation (mostly commas)
        doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc]
        # Split into words
        doc = [sent.split() for sent in doc]
        sentences += doc
        counter += 1
    return sentences

# Define function to create word vectors given a cleaned piece of text.
def create_average_vec_original(doc, text_dim, wordvec_model):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        if word in wordvec_model.wv.vocab:
            average = np.add(average, wordvec_model[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average
def create_average_vec(doc, text_dim, word2vec_model):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        average = np.add(average, word2vec_model.get_word_vec(word))
        num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

def load_data():

	data_dir = '../data/'
	train = pd.read_csv(data_dir + 'train.csv')
	test = pd.read_csv(data_dir + 'test.csv')
	return (train, test)

def get_original_word2vec(train, test):

	train_cleaned = cleanup_text(train['text'])

	all_text = np.concatenate((train['text'], test['text']), axis=0)
	all_text = pd.DataFrame(all_text, columns=['text'])

	train_cleaned_word2vec = process_text(all_text['text'])

	text_dim = 300
	wordvec_model = Word2Vec(train_cleaned_word2vec, size=text_dim, window=5,\
		min_count=3, workers=4, sg=1)

	# Create word vectors
	train_cleaned_vec = np.zeros((train.shape[0], text_dim), dtype="float32")
	for i in range(len(train_cleaned)):
		train_cleaned_vec[i] = create_average_vec_original(train_cleaned[i], text_dim, wordvec_model)

	return train_cleaned_vec

def get_word2vec():

	text_dim = 200

	mlp_train, mlp_test= read_and_select(['id', 'text'])
	w2v_model = w2v(mlp_train, mlp_test)
	w2v_model.train_model()
	proc_train = mlp_train['processed_text']
	proc_test = mlp_test['processed_text']

	train_vec = np.zeros((proc_train.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_train)):
		train_vec[i] = create_average_vec(proc_train[i], text_dim, w2v_model)
	test_vec = np.zeros((proc_test.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_test)):
		test_vec[i] = create_average_vec(proc_test[i], text_dim, w2v_model)
	return (train_vec, test_vec)

def main():
	train, test = load_data()
	train_cleaned_vec = get_original_word2vec(train, test)
	#get_word2vec()
if __name__ == "__main__":
    main()