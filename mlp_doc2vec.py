from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
from word2vec_models import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import log_loss

from keras.layers import Dropout

mlp_train, mlp_test = read_and_select(['id', 'text', 'processed_text'], True)

y_train_ohe = label_binarize(mlp_train['author'], classes=['EAP', 'HPL', 'MWS'])

train_vec = mlp_train.drop(['author', 'EAP', 'HPL', 'MWS'], axis=1)


train_vec = train_vec.as_matrix()
test_vec = mlp_test.as_matrix()

X_train, X_val, y_train, y_val = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=42)


learn_rate = 0.05#, 0.01, 0.5]
drop_out = 0.5#, 0.5]

model = Sequential()
model.add(Dense(units=200, activation='relu', input_shape=(310,)))

model.add(Dropout(drop_out))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=300, activation='relu'))

model.add(Dense(units=3, activation='softmax'))

#ada = Adadelta(lr=learn_rate, decay=1e-6)
model.compile(loss=losses.categorical_crossentropy, optimizer='sgd')
model.fit(X_train, y_train, epochs=200, batch_size=50)

score = log_loss(y_val, model.predict(X_val))
print("SCORE: ", score)