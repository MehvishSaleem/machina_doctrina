from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.optimizers import Adam, Adadelta
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from word2vec_models import *
from sklearn.preprocessing import label_binarize
from keras.layers import Dropout


def create_model(layers, learn_rate=0.01, neurons=20, drop_out=0.5):
    model = Sequential()
    model.add(Dense(units=200, activation='relu', input_shape=(310,)))

    for layer in range(layers-1):
        #model.add(Dropout(drop_out))
        model.add(Dense(units=neurons, activation='relu'))

    model.add(Dense(units=3, activation='softmax'))

    ada = Adadelta(lr=learn_rate, decay=1e-6)
    model.compile(loss=losses.categorical_crossentropy, optimizer=ada)

    return model


mlp_train, mlp_test = read_and_select(['id', 'text', 'processed_text'], True)

y_train_ohe = label_binarize(mlp_train['author'], classes=['EAP', 'HPL', 'MWS'])

train_vec = mlp_train.drop(['author', 'EAP', 'HPL', 'MWS'], axis=1)


train_vec = train_vec.as_matrix()
test_vec = mlp_test.as_matrix()

X_train, X_val, y_train, y_val = train_test_split(train_vec, y_train_ohe, test_size=0.2, random_state=42)

deep_network = KerasRegressor(build_fn=create_model)

epochs = [200]#, 200]
batches = [100]#, 128, 250]
learn_rate = [0.05]#, 0.01, 0.5]
neurons = [100]#, 200, 300]
drop_out = [0.5]#, 0.5]
layers = [3]

hyperparameters = dict(layers=layers,batch_size=batches, epochs=epochs, learn_rate=learn_rate, drop_out=drop_out, neurons=neurons)
grid = GridSearchCV(estimator=deep_network, param_grid=hyperparameters, n_jobs=-1)
result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
print(result.best_score_, result.best_params_)
