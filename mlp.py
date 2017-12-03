#import plaidml.keras
#plaidml.keras.install_backend()

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import optimizers
from get_features import *

def main():

	text_dim = 200

	train, test = load_data()
	train_vec, test_vec = get_word2vec()
	# Transform labels into one hot encoded format
	y_train_ohe = label_binarize(train['author'], classes=['EAP', 'HPL', 'MWS'])

	#Prep for training/validations sets
	X_train, X_val, y_train, y_val = train_test_split(train_vec,\
		y_train_ohe, test_size=0.2, random_state=21)

	layers = [2, 3] # number of layers
	learning_rates = [0.005, 0.01, 0.05] # learning rate
	units = [200, 300, 500] # number of units per layer
	dropout = [0.2, 0.5] # use dropout or no? what's dropout rate?

	epochs = 50

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

					score = log_loss(y_val, model.predict(X_val))
					print(score)
					s = 'layers=' + str(l) + ', node number =' + str(u) + \
					    ', dropout=' + str(d) + ', lr='+str(lr)
					all_scores[s] = score

	sorted_scores = OrderedDict(sorted(all_scores.items(), key=lambda x: x[1]))
	for k, v in sorted_scores.items():
		print ("%s: %s" % (k, v))

if __name__ == "__main__":
    main()
