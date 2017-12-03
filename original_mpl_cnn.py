import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from get_features import *


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
    else:
        print('Error: Model type not found.')
    return model


def main():

    train, test = load_data()
    train_cleaned_vec = get_original_word2vec(train, test)
	# Transform labels into one hot encoded format
    y_train_ohe = label_binarize(train['author'], classes=['EAP', 'HPL', 'MWS'])

    #Prep for training/validations sets
    X_train, X_val, y_train, y_val = train_test_split(train_cleaned_vec,\
        y_train_ohe, test_size=0.2, random_state=21)

	# Train on the MLP
    model = build_model('mlp')

	# Compile the model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

	# Define number of epochs
    epochs = 50

	# Fit the model to the training data
    history = model.fit(X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=128, verbose=1)

    score = log_loss(y_val, model.predict(X_val))
    print('Log loss for MLP is %f'%score)
    # Plot error over epochs
    plt.figure(10)
    plt.rcParams.update({'font.size': 15})
    plt.plot(history.history['loss'], label='Loss')
    
    plt.ylabel('Loss')
    plt.title('Training with MLP')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Train on the MLP
    model = build_model('mlp')
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

    # Train on the CNN
    model = build_model('cnn')
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    history = model.fit(X_train, y_train,
        validation_split=0.2,
        epochs=epochs, batch_size=128, verbose=1)

    score = log_loss(y_val, model.predict(X_val))
    print('Log loss for CNN is %f'%score)
    # Plot error over epochs
    plt.figure(10)
    plt.rcParams.update({'font.size': 15})
    plt.plot(history.history['loss'], label='Loss')
    
    plt.ylabel('Loss')
    plt.title('Training with CNN')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
