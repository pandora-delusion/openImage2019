#-*-coding:utf-8-*-

import keras
import tensorflow as tf


def _set_value(matrix, nd, val):
    val_diff = val - tf.gather_nd(matrix, nd)
    # indices = []
    # for i in range(nd.get_shape()[1]):
    #     indices.append(nd[:, i])
    shape = tf.shape(matrix, out_type=tf.int64)
    # diff_matrix = tf.SparseTensor(indices=nd, values=val_diff, dense_shape=shape)
    diff_matrix = tf.sparse_to_dense(sparse_indices=nd, output_shape=shape, sparse_values=val_diff)
    return matrix + diff_matrix


def focal_loss_initializer(alphas=0.25, gamma=2.0):

    def _focal(y_true, y_pred):

        labels = y_true[:, :, :-1]
        states = y_true[:, :, -1]
        predictions = y_pred

        indices = tf.where(keras.backend.not_equal(states, -1))
        labels = tf.gather_nd(labels, indices)
        predictions = tf.gather_nd(predictions, indices)

        alphas_factor = tf.ones_like(labels, dtype=keras.backend.floatx())*(1.0 - alphas)
        # alphas_factor = tf.ones_like(labels, dtype=keras.backend.floatx())*(1.0 - alphas)
        l_indices = tf.where(keras.backend.equal(labels, 1))
        update = tf.ones((keras.backend.shape(l_indices)[0], ))*alphas
        # alphas_factor= tf.scatter_nd_update(alphas_factor, l_indices, update)
        alphas_factor =_set_value(alphas_factor, l_indices, update)
        del update,

        # diff = 1-predictions
        # focal_factor = tf.where(keras.backend.equal(labels, 1), diff, predictions)
        update = 1.0-tf.gather_nd(predictions, l_indices)
        focal_factor = _set_value(predictions, l_indices, update)

        # focal_factor = tf.scatter_nd_update(focal_factor, l_indices, update)
        del l_indices, update

        # focal_weight = alphas_factor*focal_factor**gamma
        focal_factor = focal_factor**gamma
        focal_weight = alphas_factor*focal_factor
        del alphas_factor, focal_factor
        # cls_loss = focal_weight*keras.backend.binary_crossentropy(labels, predictions)
        ##############################################################
        # cls_loss = keras.backend.binary_crossentropy(labels, predictions)
        # cls_loss = cls_loss*focal_weight

        positive = tf.where(keras.backend.equal(states, 1))
        positive = keras.backend.cast(keras.backend.shape(positive)[0], keras.backend.floatx())
        positive = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), positive)

        predictions = tf.transpose(predictions)
        labels = tf.transpose(labels)
        focal_weight = tf.transpose(focal_weight)

        def _batch_crossentropy(args):
            prediction, label, focal_weight = args
            temp = keras.backend.binary_crossentropy(label, prediction)
            temp = focal_weight*temp
            temp = keras.backend.sum(temp)
            return temp

        res = tf.map_fn(fn=_batch_crossentropy, elems=[predictions, labels, focal_weight],
                        dtype=keras.backend.floatx(), parallel_iterations=32)
        res = keras.backend.sum(res)

        return res/positive

    return _focal

def smooth_l1_initializer(sigma=3.0):

    sigma_squared = sigma**2

    def _smooth_l1(true, pred):

        results = pred
        targets = true[:, :, :-1]
        states = true[:, :, -1]

        indices = tf.where(keras.backend.equal(states, 1))
        results = tf.gather_nd(results, indices)
        targets = tf.gather_nd(targets, indices)
        # states = tf.gather_nd(targets, indices)

        difference = keras.backend.abs(results - targets)
        loss = tf.where(keras.backend.less(difference, 1.0/sigma_squared), 0.5*sigma_squared*keras.backend.pow(difference, 2),
                 difference-0.5/sigma_squared)

        positive = keras.backend.maximum(1.0, keras.backend.cast(keras.backend.shape(indices)[0],
                                                                 dtype=keras.backend.floatx()))
        positive = keras.backend.cast(positive, dtype=keras.backend.floatx())
        return keras.backend.sum(loss)/positive

    return _smooth_l1
