#import plaidml.keras
#plaidml.keras.install_backend()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, Flatten
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

	X_train = np.expand_dims(X_train, axis=2)
	X_val = np.expand_dims(X_val, axis=2)

	layers = [2, 3] 
	filter_size = [3, 4, 5]
	pool_size = [2, 5]

	# 2 layer netrok with 100 filters of size 3
	all_scores = dict()

	for l in layers:
		for fs in filter_size:
			for ps in pool_size:
				model = Sequential()
				inputs = Input(shape=(text_dim,1))
				x = Conv1D(100, fs, strides=1, padding='same', activation='relu')(inputs)
				x = MaxPooling1D(pool_size=ps)(x)
				for i in range(l-1):
					x = Conv1D(100, fs, strides=1, padding='same', activation='relu')(x)
					x = MaxPooling1D(pool_size=ps)(x)
				x = Flatten()(x)
				outputs = Dense(3, activation='softmax')(x)
				model = Model(inputs=inputs, outputs=outputs)

				sgd = optimizers.SGD(lr=0.01)
				model.compile(loss='categorical_crossentropy', optimizer=sgd)

				history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
				    epochs=50,batch_size=128, verbose=1)
				score = log_loss(y_val, model.predict(X_val))
				s = 'layers=' + str(l) + ', filter size =' + str(fs) + \
				    ', pool size =' + str(ps)
				all_scores[s] = score

	sorted_scores = OrderedDict(sorted(all_scores.items(), key=lambda x: x[1]))
	for k, v in sorted_scores.items():
		print ("%s: %s" % (k, v))

	# Increase filter number and add dropout	
	model = Sequential()
	inputs = Input(shape=(text_dim,1))
	x = Conv1D(200, 5, strides=1, padding='same', activation='relu')(inputs)
	x = MaxPooling1D(pool_size=2)(x)
	x = Dropout(0.2)(x)
	x = Conv1D(200, 5, strides=1, padding='same', activation='relu')(x)
	x = MaxPooling1D(pool_size=2)(x)
	x = Dropout(0.2)(x)
	x = Conv1D(200, 5, strides=1, padding='same', activation='relu')(x)
	x = MaxPooling1D(pool_size=2)(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	outputs = Dense(3, activation='softmax')(x)
	model = Model(inputs=inputs, outputs=outputs)

	sgd = optimizers.SGD(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)
	epochs = 50

	history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
		epochs=epochs,batch_size=128, verbose=1)
	score = log_loss(y_val, model.predict(X_val))
	s = 'layers=' + str(3) + ', filter size =' + str(5) + \
	    'filter number =' + str(200) + ', pool size =' + str(2) + \
	    'dropout rate = 0.2'
	print(score)
	# Plot error over epochs
	plt.figure(10)
	plt.rcParams.update({'font.size': 15})
	plt.plot(history.history['loss'], label='Loss')

	plt.ylabel('Loss')
	plt.title('Training with CNN + Dropouts')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()


    



	# layers = [2, 3] # number of layers
	# learning_rates = [0.005, 0.01, 0.05] # learning rate
	# filter_number = [64, 100, 128] # number of units per layer
	# filter_size = [3, 4, 5]
	# pool_size = [2, 5]
	# dropout = [0, 0.2, 0.3] # use dropout or no? what's dropout rate?
	# epochs= 50

	# all_scores = dict()
	# errors = dict()

	# for l in layers:
	# 	for fn in filter_number:
	# 		for fs in filter_size:
	# 			for ps in pool_size:
	# 				for d in dropout:
	# 					for lr in learning_rates:
	# 						model = Sequential()
	# 						inputs = Input(shape=(text_dim,1))
	# 						x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(inputs)
	# 						x = MaxPooling1D(pool_size=ps)(x)
	# 						x = Dropout(d)(x)

	# 						if (d==0):
	# 							for i in range(l-1):
	# 								x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(x)
	# 								x = MaxPooling1D(pool_size=ps)(x)
	# 						else:
	# 							for i in range(l-1):
	# 								x = Conv1D(fn, fs, strides=1, padding='same', activation='relu')(x)
	# 								x = MaxPooling1D(pool_size=ps)(x)
	# 								x = Dropout(d)(x)
	# 						# Flatten to feed to dense layer
	# 						x = Flatten()(x)
	# 						outputs = Dense(3, activation='softmax')(x)
	# 						model = Model(inputs=inputs, outputs=outputs)

							

							
	# 						errors[lr] = history

	# 						score = log_loss(y_val, model.predict(X_val))
	# 						print(score)
	# 						s = 'layers=' + str(l) + ', filter number =' + str(fn) + \
	# 						    ', filter size =' + str(fs) + \
	# 						    ', dropout=' + str(d) + ', lr='+str(lr)
	# 						all_scores[s] = score
	# sorted_scores = OrderedDict(sorted(all_scores.items(), key=lambda x: x[1]))
	# for k, v in sorted_scores.items():
	# 	print ("%s: %s" % (k, v))

if __name__ == "__main__":
    main()
