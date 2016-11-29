from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

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

def main():

	if not FLAGS.data_path:
		raise ValueError( "Must set --data_path to valid data directory" )

	config = Config

if __name__ == "__main__":
	tf.app.run()
