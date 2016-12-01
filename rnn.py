from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

from glob import glob
from time import time
from math import floor
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

flags.DEFINE_string( "data_path",  "./preprocess/", \
                     "Where the training/test data is stored." )
flags.DEFINE_string( "save_path",  "./models/neural-network/", \
                     "Model output directory." )
flags.DEFINE_string( "save_name",  "nn-model", \
                     "Model output name.")
flags.DEFINE_bool(   "save",       False, \
                     "save result model." )
flags.DEFINE_string( "load_path",  "./models/neural-network/", \
                     "Model load directory." )
flags.DEFINE_string( "load_name",  "nn-model", \
                     "Model load name.")
flags.DEFINE_bool(   "load",       False, \
                     "Load exist model." )
flags.DEFINE_bool(   "labels",     True, \
                     "Whether exist label for feature." )
flags.DEFINE_float(  "train_rate", 0.8, \
                     "How many input for train." )
flags.DEFINE_bool(   "self_test",  False, \
                     "Test by training data." )

FLAGS = flags.FLAGS

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
	                 tf.nn.sigmoid_cross_entropy_with_logits( prediction, y ) )
	optimizer  = tf.train.AdamOptimizer().minimize( cost )

	if FLAGS.save:
		saver      = tf.train.Saver()

	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

		print("-----------------------------")
		print("| --- Holding the Place --- |")
		print("-----------------------------")

		spent = 0

		for epoch in range( hm_epochs ):

			timer_sta  = time()
			epoch_loss = 0
			idx        = 0

			while idx < song_x.shape[0]:

				start       = idx
				end         = idx + batch_size

				batch_x     = np.array( song_x[start:end] )

				if end > song_x.shape[0]:

					batch_x = batch_x.reshape( \
					                  ( song_x.shape[0] - start, \
					                    n_chunks, chunk_size ) )
				else:
					batch_x     = batch_x.reshape( \
						                  ( batch_size, n_chunks, chunk_size ) )
				batch_y     = np.array( song_y[start:end] )

				_, c        = sess.run( [ optimizer, cost ], \
				                          feed_dict = { x: batch_x, \
				                                        y: batch_y } )
				epoch_loss += c
				idx        += batch_size

			timer_end  = time()

			spent += timer_end - timer_sta
			timer  = spent * ( hm_epochs - epoch - 1 ) / ( epoch + 1 )

			print( "\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K" + \
				   "Epoch %i completed out of %i (%.1f%%), loss: %f\n" \
			       %( epoch + 1, hm_epochs, 100 * ( epoch + 1 ) / hm_epochs, \
			       	  epoch_loss ) + \
			       "Estimate time remains: %ih %im %is\n" \
			       %( timer // 3600, ( timer % 3600 ) // 60, timer % 60 ) + \
			       "Already spent:         %ih %im %is" \
			       %( spent // 3600, ( spent % 3600 ) // 60, spent % 60 ) )

		print()

		if FLAGS.save:
			saver.save( sess, FLAGS.save_path + FLAGS.save_name )
			print( "Model successfully saved in '" + FLAGS.save_path + "'" + \
			       " as '" + FLAGS.save_name + "'\n" )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )

		print( "Accuracy: %.1f" %( 100 * accuracy.eval( \
			{ x: song_test_x.reshape( ( -1, n_chunks, chunk_size ) ), \
		      y: song_test_y } ) ) )
		
if __name__ == "__main__":

	features = []

	if os.path.isdir( FLAGS.data_path + "features/" ):

		features.extend( glob( "%s/*.npy" %( FLAGS.data_path + "features/" ) ) )

	if not len( features ):
		raise LookupError( "Can not find any feature data to process" )

	if FLAGS.labels:

		for name in features:

			if not os.path.isfile( FLAGS.data_path + "labels/" + \
				                   name.split('/')[-1] ):

				raise LookupError( "Can not find labels for" + \
				                   name.split('/')[-1] )

	if FLAGS.self_test:

		num_train = len( features )

	else:

		num_train = floor( FLAGS.train_rate * len( features ) )

	print( "Total %i songs found, will use %i songs for training.\n" \
		   %( len( features ), num_train ) + \
	       "Train rate: %.1f%%\n" %( 100 * num_train / len( features ) ) )

	song_x = np.load( features[0] )
	song_y = np.load( FLAGS.data_path + "labels/" + features[0].split('/')[-1] )

	for i in range( 1, num_train ):

		song_x = np.vstack( ( song_x, np.load( features[i] ) ) )
		song_y = np.vstack( ( song_y, np.load( FLAGS.data_path + "labels/" + \
			                                   features[i].split('/')[-1] ) ) )

	if FLAGS.self_test:

		num_train = 0

	for i in range( num_train, len( features ) ):

		if i == num_train:

			song_test_x = np.load( features[i] )
			song_test_y = np.load( FLAGS.data_path + "labels/" + \
				                   features[i].split('/')[-1] )
		else:

			song_test_x = np.vstack( ( song_test_x, np.load( features[i] ) ) )
			song_test_y = np.vstack( ( song_test_y, \
				                       np.load( FLAGS.data_path + "labels/" + \
				                                features[i].split('/')[-1] ) ) )

	input_nodes = song_x.shape[1]
	n_classes   = 51
	rnn_size    = 1024
	batch_size  = 128
	chunk_size  = song_x.shape[1]
	n_chunks    = 1
	hm_epochs   = 100

	print(song_x.shape)

	print( "RNN size:         %i\n" %rnn_size + \
	       "Chunk size:       %i\n" %chunk_size + \
	       "Number of chunks: %i\n" %n_chunks + \
	       "Batch size:       %i\n" %batch_size + \
	       "Number of epochs: %i\n" %hm_epochs )

	x = tf.placeholder( 'float', [ None, n_chunks, chunk_size ] )
	y = tf.placeholder( 'float' )

	train_recurrent_neural_network( x )