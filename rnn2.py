# 8 - RNN Classifier example

import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

x = np.load("3doorsdown_herewithoutyou_features.npy")
y = np.load("3doorsdown_herewithoutyou_labels.npy")

TIME_STEPS = 1
INPUT_SIZE = x.shape[1]
BATCH_SIZE = 100
BATCH_INDEX = 0
OUTPUT_SIZE = 51
CELL_SIZE = 128
LR = 0.00001

X_train = x
y_train = y.astype(int)
X_test = x
y_test = y.astype(int)
# print(y_test[0])


# data pre-processing
X_train = X_train.reshape(-1, TIME_STEPS, INPUT_SIZE) # normalize
print(len(X_train))
X_test = X_test.reshape(-1, TIME_STEPS, INPUT_SIZE) # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=OUTPUT_SIZE)
y_test = np_utils.to_categorical(y_test, nb_classes=OUTPUT_SIZE)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(40010):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    # print(X_batch)
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    # print(Y_batch)
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)




