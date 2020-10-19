#-*-coding:utf-8-*-

import keras
import tensorflow as tf

def filter(boxes, classification,
           class_specific_filter=True,
           nms=True,
           score_threshold=0.05,
           max_detections=300,
           nms_threshold=0.5):
    """

    :param boxes: [width*height*all_pyramid_anchor_number, 4]
    :param classification: [width*height*pyramid_anchor_number, num_classes]
    :param class_specific_filter:
    :param nms:
    :param score_threshold:
    :param max_detections:
    :param nms_threshold:
    :return:
    """

    def _filter(scores, labels):

        indices = tf.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes = tf.gather_nd(boxes, indices) # (M, 4)
            # filtered_scores = keras.backend.gather(scores, indices)[:, 0]
            filtered_scores = tf.gather_nd(scores, indices) # (M, 1)

            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold) # (k, 1)
            indices = keras.backend.gather(indices, nms_indices) # (k, 1)

        labels = tf.gather_nd(labels, indices) # (k, 1)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)
        # (k, 2)
        return indices

    if class_specific_filter:
        all_indices = []

        for cls in range(classification.shape[1]):
            scores = classification[:, cls] # (width*height*all_pyramid_anchor_number, )
            labels = cls*tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter(scores, labels))
        # (k, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter(scores, labels)

    # (k, 1)
    scores = tf.gather_nd(classification, indices)
    # (k, 1)
    labels = indices[:, 1]
    # print("scores: ", keras.backend.shape(scores))

    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)

    pad_size = keras.backend.maximum(0, max_detections-keras.backend.shape(scores)[0])

    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(tf.pad(labels, [[0, pad_size]], constant_values=-1), dtype='int32')

    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]

class Filter(keras.layers.Layer):

    def __init__(self, nms=True,
                 class_specific_filter=True,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 max_detections=300,
                 parallel_iterations=32,
                 **kwargs):
        self.nms = nms
        self.class_specific_filter=class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations

        super(Filter, self).__init__(**kwargs)

    def _filter(self, args):
        # boxes, classification = args
        # [width*height*all_pyramid_anchor_number, 4]
        boxes = args[0]
        # [width*height*all_pyramid_anchor_number, num_classes]
        classification = args[1]

        return filter(boxes, classification,
                      nms=self.nms,
                      class_specific_filter=self.class_specific_filter,
                      score_threshold=self.score_threshold,
                      max_detections=self.max_detections,
                      nms_threshold=self.nms_threshold)

    def call(self, inputs, **kwargs):
        boxes = inputs[0]
        classification = inputs[1]
        return tf.map_fn(self._filter, elems=[boxes, classification],
                         dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'],
                         parallel_iterations=self.parallel_iterations)

    def compute_output_shape(self, input_shape):

        return [(input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs)+1)*[None]

    def get_config(self):
        config = super(Filter, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            "score_threshold": self.score_threshold,
            "max_detections": self.max_detections,
            "parallel_iterations": self.parallel_iterations,
        })
        return config

