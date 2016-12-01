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
flags.DEFINE_bool(   "train",      True, \
                     "Train data or not." )

FLAGS = flags.FLAGS

def neural_network_model( data ):

	hidden_layer = []
	layer_result = []
	layer_nodes.append( input_nodes )

	for hidden_layer_idx in range( layer_levels ):

		hidden_layer.append( 
			{ 'weights': tf.Variable( tf.random_normal( \
				                      [ layer_nodes[hidden_layer_idx - 1], \
				                        layer_nodes[hidden_layer_idx] ] ) ), \
			  'biases':  tf.Variable( tf.random_normal( \
			  	                      [ layer_nodes[hidden_layer_idx] ] ) ) } )

	output_layer = \
		{ 'weights': tf.Variable( tf.random_normal( \
	                              [ layer_nodes[-2], n_classes ] ) ), \
	      'biases':  tf.Variable( tf.random_normal( \
	                              [ n_classes ] ) ) }

	layer_result.append( tf.add( tf.matmul( data, \
		                                    hidden_layer[0]['weights'] ), \
			                     hidden_layer[0]['biases'] ) )

	layer_result[0] = tf.nn.relu( layer_result[0] )

	for hidden_layer_idx in range( 1, layer_levels ):

		layer_result.append( tf.add( \
			tf.matmul( layer_result[hidden_layer_idx - 1], \
				       hidden_layer[hidden_layer_idx]['weights'] ), \
			hidden_layer[hidden_layer_idx]['biases'] ) )

		layer_result[hidden_layer_idx] = \
		    tf.nn.relu( layer_result[hidden_layer_idx] )

	output = tf.matmul( layer_result[-1], \
		                output_layer['weights'] ) + output_layer['biases']

	return output

def train_neural_network( x ):

	prediction = neural_network_model( x )
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

				start       = idx
				end         = idx + batch_size

				batch_x     = np.array( song_x[start:end] )
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

	if FLAGS.labels:

		for name in features:

			if not os.path.isfile( FLAGS.data_path + "labels/" + \
				                   name.split('/')[-1] ):
				raise LookupError( "Can not find labels for" + \
				                   name.split('/')[-1] )

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
			              features[0].split('/')[-1] )

	else:
		song_x = None
		song_y = None

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
	# Train first 3 songs, test first 3 songs
	# 3 hl, 100 nd, 100 batch, 100 epoch

	if FLAGS.train:
		input_nodes  = song_x.shape[1]
		hm_epochs    = 100

	else:
		input_nodes  = song_test_x.shape[1]
		hm_epochs    = 0

	layer_levels = 3
	n_classes    = 51
	layer_nodes  = [ 100 ] * layer_levels
	batch_size   = 100

	print( "Number of hidden layers: %i\n" %layer_levels + \
	       "Number of nodes in hidden layer: \n%s\n" %layer_nodes + \
	       "Batch size:              %i\n" %batch_size + \
	       "Number of epochs:        %i\n" %hm_epochs )

	x = tf.placeholder( 'float', [ None, input_nodes ] )
	y = tf.placeholder( 'float' )

	train_neural_network( x )
