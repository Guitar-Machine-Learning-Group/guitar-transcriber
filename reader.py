# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 2048
      if self.one_hot:
        fake_label = [1] + [0] * 50
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=2048):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_FEATURES = '3doorsdown_herewithoutyou_features.npy'
  TRAIN_LABELS   = '3doorsdown_herewithoutyou_labels.npy'

  TEST_FEATURES  = '3doorsdown_herewithoutyou_features.npy'
  TEST_LABELS    = '3doorsdown_herewithoutyou_labels.npy'

  train_features = numpy.load( TRAIN_FEATURES )
  train_labels   = numpy.load( TRAIN_LABELS )
  test_features  = numpy.load( TEST_FEATURES )
  test_labels    = numpy.load( TEST_LABELS )

  train_features = train_features.reshape( ( train_features.shape[0], 2048, 1, 1 ) )
  train_labels   = train_labels.reshape(   ( train_labels.shape[0],   51,   1, 1 ) )
  test_features  = test_features.reshape(  ( test_features.shape[0],  2048, 1, 1 ) )
  test_labels    = test_labels.reshape(    ( test_labels.shape[0],    51,   1, 1 ) )

  if not 0 <= validation_size <= len( train_features ):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format( len( train_features ), validation_size ) )

  validation_features = train_features[:validation_size]
  validation_labels   = train_labels[:validation_size]
  train_features      = train_features[validation_size:]
  train_labels        = train_labels[validation_size:]

  train = DataSet(train_features, train_labels, dtype=dtype, reshape=reshape)

  validation = DataSet(validation_features,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)

  test = DataSet(test_features, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)