import pandas as pd
import numpy as np
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

	data_dir = '../data/'
	train = pd.read_csv(data_dir + 'train.csv')
	test = pd.read_csv(data_dir + 'test.csv')

	mlp_train, mlp_test= read_and_select(['id', 'text'])
	w2v_model = w2v(mlp_train, mlp_test)
	w2v_model.train_model()

	# Transform labels into one hot encoded format.
	y_train_ohe = label_binarize(mlp_train['author'], classes=['EAP', 'HPL', 'MWS'])

	X_train, X_val, y_train, y_val = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=21)
	X_test = test_vec

	X_train = np.expand_dims(X_train, axis=2)
	X_val = np.expand_dims(X_val, axis=2)

	layers = [2, 3] # number of layers
	learning_rates = [0.005, 0.01, 0.05, 0.1] # learning rate
	filter_number = [64, 100, 128] # number of units per layer
	filter_size = [3, 4, 5]
	pool_size = [2, 5]
	dropout = [0, 0.2, 0.3] # use dropout or no? what's dropout rate?
	epochs=50

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

							#score = model.evaluate(X_val, y_val)
							score = log_loss(y_val, model.predict(X_val))
							print(score)
							s = 'layers=' + str(l) + ', filter number =' + str(fn) + \
							    ', filter size =' + str(fs) + \
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
