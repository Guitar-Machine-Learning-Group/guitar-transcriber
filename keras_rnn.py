#'''
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
    model.add(LSTM(128, return_sequences=True, input_shape=(1, 2048)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(51))
    model.add(Activation('softmax'))

    #optimizer = RMSprop(lr=0.01)
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    # compile 
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['fmeasure'])   #built in fmeasure
    
    # train
    epochs = 20
    #print(gt.X.shape[1])
    #print(gt.X.shape[2])
    X = np.array(gt.X).reshape((inputLen,1,inputDim))
    #print(X)
    #Y = np.array(gt.Y).reshape((1,inputLen,51))
    model.fit(X, gt.Y, batch_size=128, nb_epoch=epochs)
    
    #loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)   # choose whatever test set here
    #print("loss_and_metrics") #loss and fmeasure
    
    # predict
    y = model.predict(X)

    wrong = 0.0
    correct = 0.0

    for i in range(inputLen):
    	for j in range(len(y[i])):
    	    #print("t",i,"index",j,"predict",y[i][j],"truth",gt.Y[i][j])
    	    if(y[i][j]>0.55):
    	    	y[i][j] = 1.0
    	    else:
    	    	y[i][j] = 0.0

    	    #print(i,j,y[i][j],gt.Y[i][j])

    	    #print("-"*50)
    	    #print(gt.Y[i][j])


    	#print("\n")
    	#print(y[i]==gt.Y[i])
    	eva = y[i]==gt.Y[i]

    	if(False in eva):
    		wrong += 1
    		#print("wrong\n")
    	else:
    		correct += 1
    		#print("correct\n")

    	#raw_input("")

    print(correct)
    print(wrong)
    print("accurate:",correct/inputLen)

    #print(y[0])
    #'''
















'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
'''
#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print('-' * 50)
    print(X)
    print('-' * 50)
    print(y)
    print('-' * 50)

    print("len(X)",len(X),"len(X[0])",len(X[0]),"len(X[0][0])",len(X[0][0]))
    print("len(y)",len(y),"len(y[0])",len(y[0]))
    print("maxlen",maxlen,"len(chars)",len(chars))

    model.fit(X, y, batch_size=128, nb_epoch=1)

    

    start_index = random.randint(0, len(text) - maxlen - 1)


    #break

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            print(x)
            preds = model.predict(x, verbose=0)[0]
            print(preds)

            next_index = sample(preds, diversity)
            print(next_index)

            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
'''
