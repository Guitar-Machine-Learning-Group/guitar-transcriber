from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

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

flags.DEFINE_string( "data_path", None, \
                     "Where the training/test data is stored.")
flags.DEFINE_string( "save_path", None, \
                     "Model output directory.")
flags.DEFINE_bool(   "use_fp16", False, \
                     "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

song_x       = np.load("3doorsdown_herewithoutyou_features.npy")
song_y       = np.load("3doorsdown_herewithoutyou_labels.npy")
song_test_x  = song_x
song_test_y  = song_y
input_nodes  = song_x.shape[1]

layer_levels = 3
n_classes    = 51
layer_nodes  = [ 2000, 2000, 2000 ]
batch_size   = 1000
hm_epochs    = 10

x = tf.placeholder( 'float', [ None, input_nodes ] )
y = tf.placeholder( 'float' )

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
	                 tf.nn.softmax_cross_entropy_with_logits( prediction, y ) )
	optimizer  = tf.train.AdamOptimizer().minimize( cost )

	hm_epochs  = 10

	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

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
			

			print( "Epoch %i completed out of %i, loss: %f" \
			       %( epoch + 1, hm_epochs, epoch_loss ) )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )

		print( "Accuracy: %f" %accuracy.eval( { x: song_test_x, \
		                                        y: song_test_y } ) )
		
if __name__ == "__main__":
	train_neural_network( x )
