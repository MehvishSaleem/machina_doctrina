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
from keras.layers import Dense, Dropout, Input, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import optimizers

import spacy
import string
import re

from read_and_select import *
from word2vec_models import *

# Need to represent a sentence through a combination of its word vector representations
def create_average_vec(doc, text_dim, word2vec_model):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        #print(word2vec_model.get_word_vec(word))
        average = np.add(average, word2vec_model.get_word_vec(word))
        num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

def build_nn_models(layers, units, dropout, model_type='mlp'):
    model = Sequential()
    # (Multi-Layer Perceptron)
    model.add(Dense(512, activation='relu', input_dim=200))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # elif architecture == 'cnn':
    #     # 1-D Convolutional Neural Network
    #     inputs = Input(shape=(300,1))

    #     x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)

    #     #Cuts the size of the output in half, maxing over every 2 inputs
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
    #     x = GlobalMaxPooling1D()(x) 
    #     outputs = Dense(3, activation='softmax')(x)

    #     model = Model(inputs=inputs, outputs=outputs, name='CNN')
    # else:
    #     print('Error: Model type not found.')
    return model


def main():

	data_dir = '../data/'
	train = pd.read_csv(data_dir + 'train.csv')
	test = pd.read_csv(data_dir + 'test.csv')

	mlp_train, mlp_test= read_and_select(['id', 'text'])
	w2v_model = w2v(mlp_train, mlp_test)
	w2v_model.train_model()

	layers = [2, 3, 5] # number of layers
	learning_rates = [0.005, 0.01, 0.05, 0.1] # learning rate
	units = [200, 300, 500] # number of units per layer
	dropout = [0, 0.2, 0.5] # use dropout or no? what's dropout rate?

	proc_train = mlp_train['processed_text']
	proc_test = mlp_test['processed_text']
	text_dim = 200


	train_vec = np.zeros((proc_train.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_train)):
		train_vec[i] = create_average_vec(proc_train[i], text_dim, w2v_model)
	test_vec = np.zeros((proc_test.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_test)):
		test_vec[i] = create_average_vec(proc_test[i], text_dim, w2v_model)

	print("Train word vector shape:", train_vec.shape)

	# Transform labels into one hot encoded format.
	y_train_ohe = label_binarize(mlp_train['author'], classes=['EAP', 'HPL', 'MWS'])
	#y_train_ohe = mlp_train[['EAP', 'HPL', 'MWS']]
	print('y_train_ohe shape: {}'.format(y_train_ohe.shape))
	print('y_train_ohe samples:')
	print(y_train_ohe[:5])

	# If using spaCy word vectors
	X_train, X_val, y_train, y_val = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=21)
	X_test = test_vec
	# If using Word2Vec word vectors
	#X_train, X_test, y_train, y_test = train_test_split(train_cleaned_vec, y_train_ohe, test_size=0.2, random_state=21)

	print('X_train size: {}'.format(X_train.shape))
	print('X_val size: {}'.format(X_val.shape))
	print('y_train size: {}'.format(y_train.shape))
	print('y_val size: {}'.format(y_val.shape))

    # Define number of epochs
	epochs = 30
	all_scores = dict()
	errors = dict()

	for l in layers:
		for u in units:
			for d in dropout:
				for lr in learning_rates:
					model = Sequential()
					model.add(Dense(units=u, input_shape=(text_dim,)))
					model.add(Activation('relu'))

					if (d==0):
						for i in range(l-1):
							model.add(Dense(units=u))
							model.add(Activation('relu'))

					else:
						for i in range(l-1):
							model.add(Dense(units=u))
							model.add(Activation('relu'))
							model.add(Dropout(d))
                    
					model.add(Dense(units=3))
					model.add(Activation('softmax'))

					sgd = optimizers.SGD(lr=lr)
                        
					model.compile(loss='categorical_crossentropy', optimizer=sgd)
					history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
					errors[lr] = history
					#score = model.evaluate(X_val, y_val)
					score = log_loss(y_val, model.predict(X_val))
					print(score)
					s = 'layers=' + str(l) + ', node number =' + str(u) + \
                            ', dropout=' + str(d) + ', lr='+str(lr)
					all_scores[s] = score

	sorted_scores = OrderedDict(sorted(all_scores.items(), key=lambda x: x[1]))
	for k, v in sorted_scores.items():
		print ("%s: %s" % (k, v))


	# # If the model is a CNN then expand the dimensions of the training data
	# if model.name == "CNN" or model.name == "LSTM":
	# 	X_train = np.expand_dims(X_train, axis=2)
	# 	X_test = np.expand_dims(X_test, axis=2)
	# print('Text train shape: ', X_train.shape)
	# print('Text test shape: ', X_test.shape)

	# model.summary()

	# # Compile the model
	# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(optimizer=sgd, loss='categorical_crossentropy')



	# # Fit the model to the training data
	# estimator = model.fit(X_train, y_train,
 #        validation_split=0.2,
 #        epochs=epochs, batch_size=128, verbose=1)


	# score = model.evaluate(X_test, y_test)
	# y_preds = model.predict(X_test)

	# print("Log Loss", log_loss(y_test, y_preds))
	# # Fit the model to the training data
	# estimator = model.fit(train_vec, y_train_ohe,
 #        epochs=epochs, batch_size=128, verbose=1)
	# predictions = model.predict(test_vec)
	# print(predictions)
	# print (score)
 #    #print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 
 #    #  (100*estimator.history, 100*estimator.history))

if __name__ == "__main__":
    main()
