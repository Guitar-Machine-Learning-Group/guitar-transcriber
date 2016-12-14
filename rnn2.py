import os
os.environ['KERAS_BACKEND']='tensorflow'
import glob
import timeit
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from keras import metrics

# x = np.load("3doorsdown_herewithoutyou_features.npy")
# y = np.load("3doorsdown_herewithoutyou_labels.npy")

TIME_STEPS = 1
# INPUT_SIZE = x.shape[1]
BATCH_SIZE = 256
BATCH_INDEX = 0
OUTPUT_SIZE = 51
CELL_SIZE = 32
LR = 0.001

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, 2048),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['fmeasure'])

# training
def train(X_train, y_train):
    global BATCH_INDEX, BATCH_SIZE
    step_size = (X_train.shape[0] // BATCH_SIZE + 1) * 15
    print(step_size)
    for step in range(step_size):
        X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
        Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
        cost = model.train_on_batch(X_batch, Y_batch)
        BATCH_INDEX += BATCH_SIZE
        BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

def test(X_test, y_test):
    cost, accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    print('test accuracy: ', accuracy)

if __name__ == "__main__":
    start = timeit.default_timer()
    # pre-load data
    featureFileList = glob.glob('./preprocess/features/*', recursive=True)
    labelFileList = glob.glob('./preprocess/labels/*', recursive=True)
    if len(featureFileList) != len(labelFileList):
        print("Data is not matched")
        exit(-1)
    total = len(labelFileList)
    # total = 2
    trainSize = int(total * 0.8)
    # trainSize = 3
    testX = np.array([])
    testY = np.array([]).astype(int)
    trainX = np.array([])
    trainY = np.array([]).astype(int)
    # featureFileList2 = glob.glob('./preprocess2/features/*', recursive=True)
    # labelFileList2 = glob.glob('./preprocess2/labels/*', recursive=True)


    for count in range(total):
        # if count == total-1:
        #     featureFileList = featureFileList2
        #     labelFileList = labelFileList2
        selector = np.random.randint(0, len(featureFileList))
        featureFileName = featureFileList.pop(selector)

        for i in range(len(labelFileList)):
            if labelFileList[i].split('/')[-1].split('.')[0] == featureFileName.split('/')[-1].split('.')[0]:
                labelFileName = labelFileList.pop(i)
                break
        x = np.load(featureFileName)
        y = np.load(labelFileName).astype(int)
        if count < trainSize:
            trainX = np.append(trainX, x)
            trainY = np.append(trainY, y)
        elif count < total:
            f = labelFileName.split('/')[-1].split('.')[0]
            if count == trainSize:
                # print(trainX.shape)
                # input_size = trainX.shape[1]
                train(trainX.reshape(-1, TIME_STEPS, 2048), trainY.reshape(-1, 51))
            testX = np.append(testX, x)
            testY = np.append(testY, y)
            # print("test: " + f)
            if count == total - 1:
                # output_size = testX.shape[1]
                result = test(testX.reshape(-1, TIME_STEPS, 2048), testY.reshape(-1, 51))
                # print("score :", result)
        # save_model(s)
    stop = timeit.default_timer()
    print ("Time used: %.2f s" % (stop - start))

