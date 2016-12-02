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
from tensorflow.python.ops import rnn, rnn_cell

flags = tf.flags
logging = tf.logging

flags.DEFINE_string( "data_path",  "./preprocess/", \
                     "Where the training/test data is stored." )
flags.DEFINE_string( "save_path",  "./models/recurrent-neural-network/", \
                     "Model output directory." )
flags.DEFINE_string( "save_name",  "rnn-model", \
                     "Model output name.")
flags.DEFINE_bool(   "save",       False, \
                     "save result model." )
flags.DEFINE_string( "load_path",  "./models/recurrent-neural-network/", \
                     "Model load directory." )
flags.DEFINE_string( "load_name",  "rnn-model", \
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

def findStateIndex(state,stateArray):
	for i in range(len(stateArray)):
		if np.array_equal(state, stateArray[i]):
			return i
	return None


if __name__ == "__main__":

	'''
		loading feature prepare (start)
	'''

	features = []

	if os.path.isdir( FLAGS.data_path + "features/" ):
		features.extend( glob( "%s/*.npy" %( FLAGS.data_path + "features/" ) ) )

	if not len( features ):
		raise LookupError( "Can not find any feature data to process" )

	if os.name == 'nt':
		split_symbol = '\\'

	else:
		split_symbol = '/'

	if FLAGS.labels:

		for name in features:

			if not os.path.isfile( FLAGS.data_path + "labels/" + \
				                   name.split( split_symbol )[-1] ):
				raise LookupError( "Can not find labels for " + \
				                   name.split( split_symbol )[-1] )

	# random.shuffle( features )

	if FLAGS.self_test:
		num_train = len( features )

	elif not FLAGS.train:
		num_train = 0

	else:
		num_train = floor( FLAGS.train_rate * len( features ) )

	print( "Total %i songs found, will use %i songs for training.\n" \
		   %( len( features ), num_train ) + \
	       "Train rate: %.1f%%\n" %( 100 * num_train / len( features ) ) )

	'''
		loading feature prepare (end)
	'''



	# load feature from files
	f = np.load(features[0])
	l = np.load( FLAGS.data_path + "labels/" + \
			              features[0].split( split_symbol )[-1] )
	# find average amptitude
	f_avg = np.average(f, axis=1).reshape((len(f),1))
	# the sencond preprocess of feature
	s_X = f>f_avg

	# for element in f[0]:
	# 	print(element)
	helper_test = l
	# print(f_avg)
	# print(helper_test)
	# print(len(helper_test))
	# print(np.vstack({tuple(row) for row in helper_test}))
	# print(len(np.vstack({tuple(row) for row in helper_test})))

	# setup HMM state transition probability
	trans_prob = np.ones((2,2))
	# setup HMM emission probability
	num_state  = len(np.vstack({tuple(row) for row in l}))
	emi_prob = np.ones((2,2))

