import pandas as pd

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, Model
from keras.layers import *
from keras import losses
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


lemm = WordNetLemmatizer()
train_orig = pd.read_csv("train.csv")
train = pd.read_csv("train_df.csv", usecols=["processed_text"])
test = pd.read_csv("test_df.csv", usecols=["processed_text"])


train_text = list(pd.DataFrame(train).values)
test_text = list(pd.DataFrame(test).values)

y_train = np.zeros(shape=(19579, 3))

for idx, author in enumerate(list(train_orig.author)):
    if author == 'EAP':
        y_train[idx] = [1, 0, 0]
    elif author == 'HPL':
        y_train[idx] = [0, 1, 0]
    if author == 'MWS':
        y_train[idx] = [0, 0, 1]

vocab_size = 100000

encoded_docs = [one_hot(' '.join(t), vocab_size) for t in train_text]

max_length = 100
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


X_train, X_test, Y_train, Y_test = train_test_split(padded_docs, y_train, test_size=0.2, random_state=1)
#X_train = padded_docs
#Y_train = y_train

model = Sequential()
model.add(Embedding(vocab_size, output_dim=300, input_length=max_length))
#model.add(Bidirectional(LSTM(128, unroll=True, return_sequences=True)))
model.add(Bidirectional(LSTM(128, unroll=True)))
model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy, optimizer='rmsprop')
model.fit(X_train, Y_train, batch_size=50, epochs=10, validation_data=(X_test, Y_test))


# # FOR PREDICTIONS ON TEST DATA
#encoded_docs_test = [one_hot(' '.join(t), vocab_size) for t in test_text]
#max_length = 100
#padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
#print(padded_docs[0])
#print(padded_docs_test[0])
#loss, accuracy = model.evaluate(Y_test, Y_train, verbose=0)
#print('Accuracy: %f' % (accuracy*100))

#predictions = model.predict(padded_docs_test)
#print(predictions)


score = log_loss(Y_test, model.predict(X_test))
print("SCORE: ", score)
