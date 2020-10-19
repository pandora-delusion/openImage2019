#-*-coding:utf-8-*-

import keras
import tensorflow as tf

import backend

class UpSampleLayer(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == "channels_first":
            source = tf.transpose(source, (0, 2, 3, 1))
            output = backend.resize_image(source, (target_shape[2], target_shape[3]),
                                          method="nearest")
            output = tf.transpose(output, (0, 3, 1, 2))
        else:
            output = backend.resize_image(source, (target_shape[1], target_shape[2]),
                                          method="nearest")
        return output

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == "channels_first":
            shape = (input_shape[0][0], input_shape[0][1], input_shape[1][2:])
        else:
            shape = (input_shape[0][0], input_shape[1][1:3], input_shape[0][3])

        return shape


