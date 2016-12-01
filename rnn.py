from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
np.set_printoptions( threshold = np.inf )
import tensorflow as tf

from scoreevent import Chord, Note
from tensorflow.python.ops import rnn, rnn_cell

'''
input > weight > hidden layer 1 (activation function) > weights >
hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer..SGD..AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

flags = tf.flags
logging = tf.logging

flags.DEFINE_string( "data_path", None, \
                     "Where the training/test data is stored.")
flags.DEFINE_string( "save_path", None, \
                     "Model output directory.")
flags.DEFINE_bool(   "use_fp16", False, \
                     "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

'''
input data here for train will be 60,000 dataset contains 28*28 pixels image
'''
#song = input_data.read_data_sets( "/tmp/data/", one_hot=True )
song_x      = np.load("3doorsdown_herewithoutyou_features.npy")
song_y      = np.load("3doorsdown_herewithoutyou_labels.npy")
song_test_x = song_x
song_test_y = song_y
input_nodes = song_x.shape[1]

n_classes   = 51
batch_size  = 1
rnn_size    = 128
chunk_size  = song_x.shape[1]
n_chunks    = 1
hm_epochs   = 3

x = tf.placeholder( 'float', [ None, n_chunks, chunk_size ] )
y = tf.placeholder( 'float' )

def recurrent_neural_network_model( x ):

	layer = { 'weights': tf.Variable( tf.random_normal( \
		                              [ rnn_size, n_classes ] ) ),
              'biases':  tf.Variable( tf.random_normal( \
              	                      [ n_classes ] ) ) }

	x = tf.transpose( x, [ 1, 0, 2 ] )
	x = tf.reshape( x, [ -1, chunk_size ] )
	x = tf.split( 0, n_chunks, x )

	lstm_cell       = rnn_cell.BasicLSTMCell( rnn_size, state_is_tuple = True )
	outputs, states = rnn.rnn( lstm_cell, x, dtype = tf.float32 )

	output          = tf.matmul( outputs[-1], \
    	                         layer['weights'] ) + layer['biases']

	return output

def train_recurrent_neural_network( x ):

	prediction = recurrent_neural_network_model( x )
	cost       = tf.reduce_mean( \
	                 tf.nn.softmax_cross_entropy_with_logits( prediction, y ) )
	optimizer  = tf.train.AdamOptimizer().minimize( cost )

	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

		for epoch in range( hm_epochs ):

			epoch_loss = 0
			
			idx = 0

			while idx < song_x.shape[0]:

				start       = idx
				end         = idx + batch_size

				batch_x     = np.array( song_x[start:end] )
				batch_x     = batch_x.reshape( \
					                  ( batch_size, n_chunks, chunk_size ) )
				batch_y     = np.array( song_y[start:end] )

				_, c        = sess.run( [ optimizer, cost ], \
				                          feed_dict = { x: batch_x, \
				                                        y: batch_y } )
				epoch_loss += c
				idx        += batch_size
			

			print( "Epoch %i completed out of %i, loss: %f" \
			       %( epoch + 1, hm_epochs, epoch_loss ) )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )

		print( "Accuracy: %f" %accuracy.eval( \
			{ x: song_test_x.reshape( ( -1, n_chunks, chunk_size ) ), \
		      y: song_test_y } ) )
		
if __name__ == "__main__":
	train_recurrent_neural_network( x )
