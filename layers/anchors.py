#-*-coding:utf-8-*-

import numpy as np
import keras
import utils.anchors as uanchors
import backend

class Anchors(keras.layers.Layer):

    def __init__(self, size, stride, ratios, scales, **kwargs):

        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        self.num_anchors = len(ratios)*len(scales)
        self.anchors = keras.backend.variable(uanchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales))
        super(Anchors, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs_shape = keras.backend.shape(inputs)

        if keras.backend.image_data_format() == "channels_first":
            anchors = backend.shift(inputs_shape[2:], self.stride, self.anchors)
        else:
            anchors = backend.shift(inputs_shape[1:3], self.stride, self.anchors)

        anchors = keras.backend.expand_dims(anchors, axis=0)
        anchors = keras.backend.tile(anchors, (inputs_shape[0], 1, 1))
        # anchors (batch_number, anchors_number, 4)
        return anchors

    def compute_output_shape(self, input_shape):

        if None in input_shape[1:]:
            return input_shape[0], None, 4
        else:
            if keras.backend.image_data_format() == "channels_first":
                number = input_shape[2]*input_shape[3]*self.num_anchors
            else:
                number = input_shape[1]*input_shape[2]*self.num_anchors
            return input_shape[0], number, 4

