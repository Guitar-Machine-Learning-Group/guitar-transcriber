import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

from guitar_transcriber import GuitarTranscriber

if __name__ == "__main__":
    audio_path = './audio'
    label_path = './midi'
    window_size = 2048
    hop_size = 1024
    sampling_rate = 22050

    # extract feature and label
    gt = GuitarTranscriber(audio_path, label_path, window_size, hop_size, sampling_rate)
    #print("X_num_wins:",len(gt.X),"X_win_size:",len(gt.X[0]))
    #print("Y_num_wins (should equal to X_num_wins):",len(gt.Y),"Y_win_size(should be 51):",len(gt.Y[0]))
    
    # create out net work
    inputLen = len(gt.X)
    inputDim = len(gt.X[0])
    model = Sequential()
    model.add(LSTM(51, return_sequences=True, input_dim=inputDim, input_length=inputLen))

    # model.add(Dropout(0.2))
    # model.add(LSTM(512, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(len(chars)))
    # model.add(Activation('softmax'))
    
    # compile 
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    # train
    epochs = 10
    #print(gt.X)
    #X = np.array(gt.X).reshape((inputLen,inputDim,1))
    #Y = np.array(gt.Y).reshape((1,inputLen,51))
    model.fit(gt.X, gt.Y, batch_size=128, nb_epoch=epochs)
    '''
    # predict
    y = model.predict(gt.X)

    print(y)
    print(y[0])
    '''