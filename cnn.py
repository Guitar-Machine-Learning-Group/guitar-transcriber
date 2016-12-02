from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import tensorflow as tf

from glob import glob
from time import time
from math import floor
from scoreevent import Chord, Note

'''
input > weight > hidden layer 1 (activation function) > weights >
hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer..SGD..AdaGrad)

backpropagation

feed forward + backprop = epoch

Result around 50%

skyu0221@guitar-vmware:~/guitar-transcriber$ python3 nn.py --save=True
Total 44 songs found, will use 35 songs for training.
Train rate: 79.5%

Number of hidden layers: 3
Number of nodes in hidden layer: 
[100, 100, 100]
Batch size:              200
Number of epochs:        15

Epoch 15 completed out of 15 (100.0%), loss: 29090.942720
Estimate time remains: 0h 0m 0s
Already spent:         0h 1m 6s

Model successfully saved in './models/neural-network/' as 'nn-model'

Accuracy: 59.6
'''
flags = tf.flags
logging = tf.logging

flags.DEFINE_string( "data_path",  "./preprocess/", \
                     "Where the training/test data is stored." )
flags.DEFINE_string( "save_path",  "./models/convolutional-neural-network/", \
                     "Model output directory." )
flags.DEFINE_string( "save_name",  "nn-model", \
                     "Model output name.")
flags.DEFINE_bool(   "save",       False, \
                     "save result model." )
flags.DEFINE_string( "load_path",  "./models/convolutional-neural-network/", \
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
flags.DEFINE_bool(   "train",      True, \
                     "Train data or not." )

FLAGS = flags.FLAGS

def conv2d( x, w ):

	return tf.nn.conv2d( x, w, strides =[ 1, 1, 1, 1 ], padding = 'SAME' )

def maxpool2d( x ):

	return tf.nn.max_pool( x, ksize = [ 1, 8, 8, 1 ], \
		                    strides = [ 1, 8, 8, 1 ], padding = 'SAME' )

def convolutional_neural_network_model( x ):

	weights = { \
	    'w_conv1': tf.Variable( tf.random_normal( [ 5, 5, 1, 32 ] ) ), \
	    'w_conv2': tf.Variable( tf.random_normal( [ 5, 5, 32, 64 ] ) ), \
	    'w_fc':    tf.Variable( tf.random_normal( [ 32*32*64, 1024 ] ) ), \
	    'out':     tf.Variable( tf.random_normal( [ 1024, n_classes ] ) ) }

	biases  = { \
	    'b_conv1': tf.Variable( tf.random_normal( [ 32 ] ) ), \
	    'b_conv2': tf.Variable( tf.random_normal( [ 64 ] ) ), \
	    'b_fc':    tf.Variable( tf.random_normal( [ 1024 ] ) ), \
	    'out':     tf.Variable( tf.random_normal( [ n_classes ] ) ) }

	x = tf.reshape( x, shape = [ -1, 2048, 2048, 1 ] )

	conv1 = tf.nn.relu( conv2d( x, weights[ 'w_conv1' ] ) + \
		                biases[ 'b_conv1' ] )
	conv1 = maxpool2d( conv1 )

	conv2 = tf.nn.relu( conv2d( conv1, weights[ 'w_conv2' ] ) + \
		                biases[ 'b_conv2'] )
	conv2 = maxpool2d( conv2 )

	fc     = tf.reshape( conv2, [ -1, 32*32*64 ] )
	fc     = tf.nn.relu( tf.matmul( fc, weights[ 'w_fc' ] ) + \
		                 biases[ 'b_fc' ] )
	fc     = tf.nn.dropout( fc, keep_rate )

	output = tf.matmul( fc, weights[ 'out' ] ) + biases[ 'out' ]

	return output

def train_convolutional_neural_network( x ):

	prediction = convolutional_neural_network_model( x )
	cost       = tf.reduce_mean( \
	                 tf.nn.sigmoid_cross_entropy_with_logits( prediction, y ) )
	optimizer  = tf.train.AdamOptimizer().minimize( cost )

	if FLAGS.save or FLAGS.load:
		saver      = tf.train.Saver()

	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

		if FLAGS.load:
			new_saver = tf.train.get_checkpoint_state( FLAGS.load_path )
			if new_saver and new_saver.model_checkpoint_path:
				saver.restore( sess, new_saver.model_checkpoint_path )
				print( "Model successfully load from '" + FLAGS.load_path + \
					   "'" + " file '" + FLAGS.load_name + "'\n" )

			else:
				print( "Model cannot find. No model loaded." )

		if FLAGS.train:
			print("-----------------------------")
			print("| --- Holding the Place --- |")
			print("-----------------------------")

			spent = 0

		for epoch in range( hm_epochs ):

			timer_sta  = time()
			epoch_loss = 0
			idx        = 0

			while idx < song_x.shape[0]:

				start = idx
				end   = idx + batch_size

				if end > song_x.shape[0]:
					start = song_x.shape[0] - batch_size
					end   = song_x.shape[0]

				batch_x     = np.transpose( np.array( song_x[start:end] ) )
				batch_x     = batch_x.reshape( [ 1, -1 ] )
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

		if FLAGS.train:
			print()

		if FLAGS.save:
			if not os.path.exists( FLAGS.save_path ):
				os.makedirs( FLAGS.save_path )

			saver.save( sess, FLAGS.save_path + FLAGS.save_name )
			print( "Model successfully saved in '" + FLAGS.save_path + "'" + \
			       " as '" + FLAGS.save_name + "'\n" )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )

		print( "Accuracy: %.1f\n" \
			   %( accuracy.eval( { x: song_test_x, \
			   	                   y: song_test_y } ) * 100 ) )
		
if __name__ == "__main__":

	features = []

	if os.path.isdir( FLAGS.data_path + "features/" ):
		features.extend( glob( "%s/*.npy" %( FLAGS.data_path + "features/" ) ) )

	if not len( features ):
		raise LookupError( "Can not find any feature data to process" )

	if os.name == 'nt':
		split_symbol = '\\'

	else:
		split_symbol = "/"

	if FLAGS.labels:

		for name in features:

			if not os.path.isfile( FLAGS.data_path + "labels/" + \
				                   name.split( split_symbol )[-1] ):
				raise LookupError( "Can not find labels for " + \
				                   name.split( split_symbol )[-1] )

	random.shuffle( features )

	if FLAGS.self_test:
		num_train = len( features )

	elif not FLAGS.train:
		num_train = 0

	else:
		num_train = floor( FLAGS.train_rate * len( features ) )

	print( "Total %i songs found, will use %i songs for training.\n" \
		   %( len( features ), num_train ) + \
	       "Train rate: %.1f%%\n" %( 100 * num_train / len( features ) ) )

	if FLAGS.train:
		song_x = np.load( features[0] )
		song_y = np.load( FLAGS.data_path + "labels/" + \
			              features[0].split( split_symbol )[-1] )

	else:
		song_x = None
		song_y = None

	for i in range( 1, num_train ):

		song_x = np.vstack( ( song_x, np.load( features[i] ) ) )
		song_y = np.vstack( ( song_y, np.load( FLAGS.data_path + "labels/" + \
			                         features[i].split( split_symbol )[-1] ) ) )

	if FLAGS.self_test:
		num_train = 0

	for i in range( num_train, len( features ) ):

		if i == num_train:
			song_test_x = np.load( features[i] )
			song_test_y = np.load( FLAGS.data_path + "labels/" + \
				                   features[i].split( split_symbol )[-1] )
		else:
			song_test_x = np.vstack( ( song_test_x, np.load( features[i] ) ) )
			song_test_y = np.vstack( ( song_test_y, \
				                       np.load( FLAGS.data_path + "labels/" + \
				                     features[i].split( split_symbol )[-1] ) ) )
	# Train first 3 songs, test first 3 songs
	# 3 hl, 100 nd, 100 batch, 100 epoch

	if FLAGS.train:
		input_nodes  = song_x.shape[1]
		hm_epochs    = 15

	else:
		input_nodes  = song_test_x.shape[1]
		hm_epochs    = 0

	n_classes    = 51
	batch_size   = 2048
	keep_rate    = 0.8
	layer_levels = 2

	print( "Number of hidden layers: %i\n" %layer_levels + \
	       "Batch size:              %i\n" %batch_size + \
	       "Number of epochs:        %i\n" %hm_epochs )

	x         = tf.placeholder( 'float', [ None, input_nodes * batch_size ] )
	y         = tf.placeholder( 'float' )
	keep_prob = tf.placeholder( tf.float32 )

	train_convolutional_neural_network( x )
