#-*-coding:utf-8-*-
import keras
import backend
import tensorflow as tf

class BBox(keras.layers.Layer):

    def __init__(self, prior_scaling=None, **kwargs):

        if prior_scaling is None:
            prior_scaling = [0.1, 0.1, 0.2, 0.2]

        self.prior_scaling = prior_scaling

        super(BBox, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        # [batch_size, width*height*all_pyramid_anchor_number, 4]
        return backend.bbox_transform(anchors, regression,
                                      prior_scaling=self.prior_scaling)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(BBox, self).get_config()
        config.update({
            "prior_scaling": self.prior_scaling,
        })

        return config

class ClipBox(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        image, bboxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image),
                                   keras.backend.floatx())
        if keras.backend.image_data_format() == "channels_first":
            height = shape[2]
            width = shape[3]
        else:
            height = shape[1]
            width = shape[2]
        xmin = tf.clip_by_value(bboxes[:, :, 0], 0, width)
        ymin = tf.clip_by_value(bboxes[:, :, 1], 0, height)
        xmax = tf.clip_by_value(bboxes[:, :, 2], 0, width)
        ymax = tf.clip_by_value(bboxes[:, :, 3], 0, height)

        return keras.backend.stack([xmin, ymin, xmax, ymax],
                                   axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]