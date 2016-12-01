from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import numpy as np
import tensorflow as tf

from glob import glob
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

	if FLAGS.save:
		saver      = tf.train.Saver()

	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

		print("----- Holding the Place -----")

		for epoch in range( hm_epochs ):

			epoch_loss = 0
			
			idx = 0

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
			

			print( "\x1b[1A\x1b[2K" + \
				   "Epoch %i completed out of %i (%.1f%%), loss: %f" \
			       %( epoch + 1, hm_epochs, 100 * ( epoch + 1 ) / hm_epochs, \
			       	  epoch_loss ) )

		print()

		if FLAGS.save:
			saver.save( sess, FLAGS.save_path + FLAGS.save_name )
			print( "Model successfully saved in '" + FLAGS.save_path + "'" + \
			       " as '" + FLAGS.save_name + "'\n" )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )

		print( "Accuracy: %f\n" %accuracy.eval( { x: song_test_x, \
		                                        y: song_test_y } ) )
		
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

	print(song_x.shape)

	input_nodes  = song_x.shape[1]
	layer_levels = 3
	n_classes    = 51
	layer_nodes  = [ 50, 50, 50 ]
	batch_size   = 100
	hm_epochs    = 100

	x = tf.placeholder( 'float', [ None, input_nodes ] )
	y = tf.placeholder( 'float' )

	start_time = datetime.datetime.now()
	train_neural_network( x )
	end_time   = datetime.datetime.now()
	print( "Time spent (H:M:S): %s" %( end_time - start_time ) )