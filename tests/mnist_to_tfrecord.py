"""
Reference : https://www.tensorflow.org/tutorials/load_data/tfrecord
Author : jwlim
"""

import os
import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecords(X, Y, output_path, usage):
    def _np_to_tfrecord(x: np.ndarray, y: np.uint8) -> tf.train.Example:
        x_shape = np.shape(x)
        feature = {
            'height': _int64_feature(x_shape[0]),
            'width': _int64_feature(x_shape[1]),
            'depth': _int64_feature(x_shape[2]),
            'label': _int64_feature(y),
            'image_raw': _bytes_feature(x.tobytes())
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    N = np.shape(X)[0]

    record_file = os.path.join(output_path, f'{usage}.tfrecords')
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in range(N):
            features = _np_to_tfrecord(X[i], Y[i])
            writer.write(features.SerializeToString())


def _preprocess_mnist(x: np.ndarray, y: np.uint8):
    x = np.expand_dims(x, axis=3)
    x = np.tile(x, (1, 1, 1, 1))
    return x, y


def take_tfrecord_mnist(x, y, file_path_with_name, usage):
    x, y = _preprocess_mnist(x, y)
    write_tfrecords(x, y, file_path_with_name, usage)


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # file_path_with_name = 'gs://caplab-jeffrey/mnist/data/'
    take_tfrecord_mnist(x_train, y_train, 'gs://caplab-jeffrey/mnist/', 'train')
    take_tfrecord_mnist(x_test, y_test, 'gs://caplab-jeffrey/mnist/', 'test')