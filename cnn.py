#import plaidml.keras
#plaidml.keras.install_backend()

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import optimizers


from read_and_select import *
from word2vec_models import *

# Need to represent a sentence through a combination of its word vector representations
# Simple way - do the averaging of all the vec representations appearing in doc
def create_average_vec(doc, text_dim, word2vec_model):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        average = np.add(average, word2vec_model.get_word_vec(word))
        num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

def main():

	text_dim = 200

	mlp_train, mlp_test= read_and_select(['id', 'text'])
	w2v_model = w2v(mlp_train, mlp_test)
	w2v_model.train_model()

	proc_train = mlp_train['processed_text']
	proc_test = mlp_test['processed_text']

	proc_train.to_csv('data/train_processed.csv')
	proc_test.to_csv('data/test_processed.csv')

	print("saved files")

	train_vec = np.zeros((proc_train.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_train)):
		train_vec[i] = create_average_vec(proc_train[i], text_dim, w2v_model)
	test_vec = np.zeros((proc_test.shape[0], text_dim), dtype="float32")
	for i in range(len(proc_test)):
		test_vec[i] = create_average_vec(proc_test[i], text_dim, w2v_model)

	y_train_ohe = label_binarize(mlp_train['author'], classes=['EAP', 'HPL', 'MWS'])
	X_train, X_val, y_train, y_val = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=21)
	X_test = test_vec

	X_train = np.expand_dims(X_train, axis=2)
	X_val = np.expand_dims(X_val, axis=2)

	
	layers = [2, 3] # number of layers
	learning_rates = [0.005, 0.01, 0.05] # learning rate
	filter_number = [64, 100, 128] # number of units per layer
	filter_size = [3, 4, 5]
	pool_size = [2, 5]
	dropout = [0, 0.2, 0.3] # use dropout or no? what's dropout rate?
	epochs= 50

	all_scores = dict()
	errors = dict()

	for l in layers:
		for fn in filter_number:
			for fs in filter_size:
				for ps in pool_size:
					for d in dropout:
						for lr in learning_rates:
							model = Sequential()
							inputs = Input(shape=(text_dim,1))
							x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(inputs)
							x = MaxPooling1D(pool_size=ps)(x)
							x = Dropout(d)(x)

							if (d==0):
								for i in range(l-1):
									x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(x)
									x = MaxPooling1D(pool_size=ps)(x)
							else:
								for i in range(l-1):
									x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(x)
									x = MaxPooling1D(pool_size=ps)(x)
									x = Dropout(d)(x)
                            
                            # Flatten to feed to dense layer
							x = Flatten()(x)
							outputs = Dense(3, activation='softmax')(x)
							model = Model(inputs=inputs, outputs=outputs)

							sgd = optimizers.SGD(lr=lr)
							model.compile(loss='categorical_crossentropy', optimizer=sgd)

							history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
							errors[lr] = history

							score = log_loss(y_val, model.predict(X_val))
							print(score)
							s = 'layers=' + str(l) + ', filter number =' + str(fn) + \
							    ', filter size =' + str(fs) + \
							    ', dropout=' + str(d) + ', lr='+str(lr)
							all_scores[s] = score
	sorted_scores = OrderedDict(sorted(all_scores.items(), key=lambda x: x[1]))
	for k, v in sorted_scores.items():
		print ("%s: %s" % (k, v))

if __name__ == "__main__":
    main()
