#-*-coding:utf-8-*-

import tensorflow as tf
import keras
import keras.callbacks

def shift(input_shape, stride, anchors):

    shift_x = (keras.backend.arange(0, input_shape[1], dtype=keras.backend.floatx()) + \
        keras.backend.constant(0.5, dtype=keras.backend.floatx()))*stride
    shift_y = (keras.backend.arange(0, input_shape[0], dtype=keras.backend.floatx()) + \
        keras.backend.constant(0.5, dtype=keras.backend.floatx()))*stride

    # shift_x *= stride
    # shift_y *= stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.flatten(shift_x)
    shift_y = keras.backend.flatten(shift_y)

    shifts = keras.backend.transpose(keras.backend.stack([shift_x, shift_y, shift_x, shift_y],
                                                         axis=0))
    anchors_number = keras.backend.shape(anchors)[0]

    shifts_number = keras.backend.shape(shifts)[0]

    shifts = keras.backend.reshape(shifts, [shifts_number, 1, 4])
    anchors = keras.backend.reshape(anchors, [1, anchors_number, 4])

    shift_anchors = anchors + keras.backend.cast(shifts, dtype=keras.backend.floatx())

    shift_anchors = keras.backend.reshape(shift_anchors, [shifts_number*anchors_number, 4])

    return shift_anchors

def resize_image(images, size, method='nearest',
                 align_corners=False, preserve_aspect_ratio=False):
    """

    :param images: [batch, height, width, channels]
    :param size: (new_height, new_width)
    :param method: ['nearest', 'bilinear', 'bicubic', 'area']
    :param align_corners: False
    :param preserve_aspect_ratio: False
    :return:
    """
    methods = {
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "area": tf.image.ResizeMethod.AREA,
    }

    return tf.image.resize_images(images, size, methods[method], align_corners,
                                  preserve_aspect_ratio)

def bbox_transform(anchors, deltas, prior_scaling=None):
    if prior_scaling is None:
        prior_scaling = [0.1, 0.1, 0.2, 0.2]

    anchor_width = anchors[:, :, 2] - anchors[:, :, 0]
    anchor_height = anchors[:, :, 3] - anchors[:, :, 1]

    anchor_cx = (anchors[:, :, 0] + anchors[:, :, 2])/2.0
    anchor_cy = (anchors[:, :, 1] + anchors[:, :, 3])/2.0

    px = deltas[:, :, 0]*anchor_width*prior_scaling[0]+anchor_cx
    py = deltas[:, :, 1]*anchor_height*prior_scaling[1]+anchor_cy
    pw = tf.exp(deltas[:, :, 2]*prior_scaling[2])*anchor_width
    ph = tf.exp(deltas[:, :, 3]*prior_scaling[3])*anchor_height

    xmin = px-pw/2.0
    xmax = px+pw/2.0
    ymin = py-ph/2.0
    ymax = py+ph/2.0

    # [batch_size, width*height*all_pyramid_anchor_number, 4]
    return keras.backend.stack([xmin, ymin, xmax, ymax], axis=2)

class RedirectModel(keras.callbacks.Callback):

    def __init__(self, callback, model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

class ParallelModelCheckpoint(keras.callbacks.ModelCheckpoint):

    def __init__(self, model, filepath, monitor="val_loss", verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode="auto", period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose,
                                                      save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        print("save single model!!!")
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

    def on_batch_end(self, batch, logs=None):
        if batch % 100000 == 0:
            self.on_epoch_end(batch, None)
        super(ParallelModelCheckpoint, self).on_batch_end(batch, logs)


# class LearningRateIter(keras.callbacks.LearningRateScheduler):
#
#     def _scheduler(self, base_lr):
#         base_lr = base_lr
#
#         def _temp(idx):
#             return base_lr /
#
#         return _temp
#
#     def __init__(self, base_lr, verbose=0):
#         self.batch_length = 10000
#         super(LearningRateIter, self).__init__(self._scheduler(base_lr), verbose=verbose)
#
#     def on_batch_end(self, batch, logs=None):
#         if batch > self.batch_length:
#             self.batch_length += 10000
#             self.on_epoch_end(batch, None)





