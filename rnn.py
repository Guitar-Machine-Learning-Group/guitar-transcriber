from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import reader as input_data

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
mnist = input_data.read_data_sets( "/tmp/data/", one_hot=True )

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes   = 10
batch_size  = 100

x = tf.placeholder( 'float', [ None, 784 ] )
y = tf.placeholder( 'float' )

class Config( object ):

	init_scale    = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers    = 2
	num_steps     = 20
	hidden_size   = 200
	max_epoch     = 4
	max_max_epoch = 13
	keep_prob     = 1.0
	lr_decay      = 0.5
	batch_size    = 20
	vocab_size    = 10000

def neural_network_model( data ):

	hidden_1_layer = { 'weights': tf.Variable( tf.random_normal( \
	                                         [ 784, n_nodes_hl1 ] ) ), \
	                   'biases':  tf.Variable( tf.random_normal( \
	                                           [n_nodes_hl1] ) ) }
	hidden_2_layer = { 'weights': tf.Variable( tf.random_normal( \
	                                         [ n_nodes_hl1, n_nodes_hl2 ] ) ), \
	                   'biases':  tf.Variable( tf.random_normal( \
	                                           [n_nodes_hl2] ) ) }
	hidden_3_layer = { 'weights': tf.Variable( tf.random_normal( \
	                                         [ n_nodes_hl2, n_nodes_hl3 ] ) ), \
	                   'biases':  tf.Variable( tf.random_normal( \
	                                           [n_nodes_hl3] ) ) }
	output_layer   = { 'weights': tf.Variable( tf.random_normal( \
	                                         [ n_nodes_hl3, n_classes ] ) ), \
	                   'biases':  tf.Variable( tf.random_normal( \
	                                           [n_classes] ) ) }

	l1     = tf.add( tf.matmul( data, hidden_1_layer['weights'] ), \
	                 hidden_1_layer['biases'] )
	l1     = tf.nn.relu(l1)

	l2     = tf.add( tf.matmul( l1,   hidden_2_layer['weights'] ), \
	                 hidden_2_layer['biases'] )
	l2     = tf.nn.relu(l2)

	l3     = tf.add( tf.matmul( l2,   hidden_3_layer['weights'] ), \
	                 hidden_3_layer['biases'] )
	l3     = tf.nn.relu(l3)

	output = tf.matmul( l3, output_layer['weights'] ) + output_layer['biases']

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

			for _ in range( int( mnist.train.num_examples / batch_size ) ):

				epoch_x, epoch_y = mnist.train.next_batch( batch_size )
				_, c             = sess.run( [ optimizer, cost ], \
				                             feed_dict = { x: epoch_x, \
				                                           y: epoch_y } )
				epoch_loss      += c

			print( "Epoch %i completed out of %i, loss: %f" \
			       %( epoch + 1, hm_epochs, epoch_loss ) )

		correct  = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
		accuracy = tf.reduce_mean( tf.cast( correct, 'float' ) )
		print( "Accuracy: %f" %accuracy.eval( { x: mnist.test.images, \
		                                        y: mnist.test.labels } ) )

def main(_):

	if not FLAGS.data_path:
		raise ValueError( "Must set --data_path to valid data directory" )

	# Reader.py not start yet
	input_data = reader.read( FLAGS.data_path )

	config                 = Config
	eval_config            = Config
	eval_config.batch_size = 1
	eval_config.num_steps  = 1

	with tf.Graph().as_default():

		initializer = tf.random_uniform_initializer( -config.init_scale,
		                                              config.init_scale )
	with tf.Session() as sess:

		output = sess.run()
		print(output)
		
if __name__ == "__main__":
	#tf.app.run()
	train_neural_network( x )
