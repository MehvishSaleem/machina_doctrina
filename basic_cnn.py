import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD

import spacy
import string
import re

nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stopwords = stopwords.words('english')

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs):
    texts = []
    counter = 1
    for doc in docs:
    	doc = doc.decode('utf-8')
        if counter % 1000 == 0:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(unicode(doc))
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
# Define function to preprocess text for a word2vec model
def cleanup_text_word2vec(docs):
    sentences = []
    counter = 1
    for doc in docs:
    	doc = doc.decode('utf-8')
        if counter % 1000 == 0:
            print("Processed %d out of %d documents" % (counter, len(docs)))
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
def create_average_vec(doc, text_dim, wordvec_model):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        if word in wordvec_model.wv.vocab:
            average = np.add(average, wordvec_model[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

def build_model(architecture='mlp'):
    model = Sequential()
    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_dim=300))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        inputs = Input(shape=(300,1))

        x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)

        #Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x) 
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')
    elif architecture == 'lstm':
        # LSTM network
        inputs = Input(shape=(300,1))

        x = Bidirectional(LSTM(64, return_sequences=True),
                          merge_mode='concat')(inputs)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    else:
        print('Error: Model type not found.')
    return model


def main():
	# read train data
	# read test data
	data_dir = '../data/'
	train = pd.read_csv(data_dir + 'train.csv')
	test = pd.read_csv(data_dir + 'test.csv')

	# Cleanup text and make sure it retains original shape
	print('Original training data shape: ', train['text'].shape)
	train_cleaned = cleanup_text(train['text'])
	print('Cleaned up training data shape: ', train_cleaned.shape)

	# First combine train and test text into one dataframe to make parsing text easier
	all_text = np.concatenate((train['text'], test['text']), axis=0)
	all_text = pd.DataFrame(all_text, columns=['text'])
	print('Number of total text documents:', len(all_text))

	train_cleaned_word2vec = cleanup_text_word2vec(all_text['text'])
	print('Cleaned up training data size (i.e. number of sentences): ', len(train_cleaned_word2vec))


	text_dim = 300
	print("Training Word2Vec model...")
	wordvec_model = Word2Vec(train_cleaned_word2vec, size=text_dim, window=5, min_count=3, workers=4, sg=1)
	print("Word2Vec model created.")
	print("%d unique words represented by %d dimensional vectors" % (len(wordvec_model.wv.vocab), text_dim))

	# Create word vectors
	train_cleaned_vec = np.zeros((train.shape[0], text_dim), dtype="float32")  # 19579 x 300
	for i in range(len(train_cleaned)):
		train_cleaned_vec[i] = create_average_vec(train_cleaned[i], text_dim, wordvec_model)

	print("Train word vector shape:", train_cleaned_vec.shape)

	# Transform labels into one hot encoded format.
	y_train_ohe = label_binarize(train['author'], classes=['EAP', 'HPL', 'MWS'])
	print('y_train_ohe shape: {}'.format(y_train_ohe.shape))
	print('y_train_ohe samples:')
	print(y_train_ohe[:5])

	# If using spaCy word vectors
	# X_train, X_test, y_train, y_test = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=21)
	# If using Word2Vec word vectors
	X_train, X_test, y_train, y_test = train_test_split(train_cleaned_vec, y_train_ohe, test_size=0.2, random_state=21)

	print('X_train size: {}'.format(X_train.shape))
	print('X_test size: {}'.format(X_test.shape))
	print('y_train size: {}'.format(y_train.shape))
	print('y_test size: {}'.format(y_test.shape))


	# Define keras model
	# model = build_model('mlp')
	model = build_model('cnn')
	# model = build_model('lstm')

	# If the model is a CNN then expand the dimensions of the training data
	if model.name == "CNN" or model.name == "LSTM":
		X_train = np.expand_dims(X_train, axis=2)
		X_test = np.expand_dims(X_test, axis=2)
	print('Text train shape: ', X_train.shape)
	print('Text test shape: ', X_test.shape)

	model.summary()

	# Compile the model
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	# Define number of epochs
	epochs = 50

	# Fit the model to the training data
	estimator = model.fit(X_train, y_train,
        validation_split=0.2,
        epochs=epochs, batch_size=128, verbose=1)

	score = model.evaluate(X_test, y_test)
	predictions = model.predict(X_train)
	print(predictions)
	print (score)
    #print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 
    #  (100*estimator.history, 100*estimator.history))

if __name__ == "__main__":
    main()
