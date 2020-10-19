#-*-coding:utf-8-*-
import keras
import layers
import tensorflow as tf

from utils.anchors import AnchorConfig
from utils.setting import parse_anchor_parameter
import initializers

def create_pyramid_features(c3, c4, c5, channels=256):
    r5 = keras.layers.Conv2D(channels, kernel_size=1, strides=1, padding="same", name="c5_reduced")(c5)
    u5 = layers.UpSampleLayer(name="r5_upsampled")([r5, c4])
    r5 = keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="same",name="r5")(r5)

    r4 = keras.layers.Conv2D(channels, kernel_size=1, strides=1, padding="same", name="c4_reduced")(c4)
    r4 = keras.layers.Add(name="r4_merged")([u5, r4])
    u4 = layers.UpSampleLayer(name="r4_upsampled")([r4, c3])
    r4 = keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="same", name="r4")(r4)

    r3 = keras.layers.Conv2D(channels, kernel_size=1, strides=1, padding="same", name="c3_reduced")(c3)
    r3 = keras.layers.Add(name="r3_merged")([u4, r3])
    r3 = keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding="same", name="r3")(r3)

    r6 = keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding="same", name="r6")(c5)

    r7 = keras.layers.Activation('relu', name="r6_relu")(r6)
    r7 = keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding="same", name="r7")(r7)

    return [r3, r4, r5, r6, r7]

def regression_task(num_values, num_anchors,
                    pyramid_feature_size=256,
                    regression_feature_size=256,
                    name="regression_task"):
    options = {
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
        "kernel_initializer": keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        "bias_initializer": "zeros",
    }

    if keras.backend.image_data_format() == "channels_first":
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation="relu",
            name="pyramid_regression_{}".format(i),
            **options
        )(outputs)
    outputs = keras.layers.Conv2D(num_anchors*num_values, name="pyramid_regression", **options)(outputs)
    if keras.backend.image_data_format() == "channels_first":
        size = (3, 2, 1)
    else:
        size = (2, 1, 3)
    outputs = keras.layers.Permute(size, name="pyramid_regression_permute")(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name="pyramid_regression_reshape")(outputs)
    # outputs (batch_number, width*height*num_anchors, 4)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

    # def _regression_task(feature, pyramid_idx=0):
    #     outputs = feature
    #     for i in range(4):
    #         outputs = keras.layers.Conv2D(filters=regression_feature_size,
    #                                       activation="relu",
    #                                       name="pyramid_{}_regression_{}".format(pyramid_idx, i),
    #                                       **options)(outputs)
    #     outputs = keras.layers.Conv2D(num_anchors*num_values, name="pyramid_{}_regression"
    #                                   .format(pyramid_idx), **options)(outputs)
    #     if keras.backend.image_data_format() == "channels_first":
    #         size = (3, 2, 1)
    #     else:
    #         size = (2, 1, 3)
    #     outputs = keras.layers.Permute(size, name="pyramid_{}_regression_permute"
    #                                    .format(pyramid_idx))(outputs)
    #     outputs = keras.layers.Reshape((-1, num_values), name="pyramid_{}_regression_reshape"
    #                                    .format(pyramid_idx))(outputs)
    #
    #     return outputs
    # return _regression_task

def classification_task(num_classes, num_anchors,
                        pyramid_feature_size=256,
                        prior_probability=0.01,
                        classification_feature_size=256,
                        name="classification_task"):
    options = {
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
    }

    if keras.backend.image_data_format() == "channels_first":
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(filters=classification_feature_size,
                                      activation="relu",
                                      name="pyramid_classification_{}".format(i),
                                      kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                                      bias_initializer="zeros",
                                      **options)(outputs)
    outputs = keras.layers.Conv2D(filters=num_classes * num_anchors,
                                  kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                                  bias_initializer=initializers.PriorProbability(probability=prior_probability),
                                  name="pyramid_classification",
                                  **options)(outputs)
    if keras.backend.image_data_format() == "channels_first":
        resize = (3, 2, 1)
    else:
        resize = (2, 1, 3)
    ### Permute,Reshape都不包含batch轴
    outputs = keras.layers.Permute(resize, name="pyramid_classification_permute")(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name="pyramid_classification_reshape")(outputs)
    outputs = keras.layers.Activation("sigmoid", name="pyramid_classification_sigmoid")(outputs)

    # output (batch_number, width*height*anchor_number, num_classes)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    # def _classification_task(feature, pyramid_idx):
    #
    #     outputs = feature
    #     for i in range(4):
    #         outputs = keras.layers.Conv2D(filters=classification_feature_size,
    #                                       activation="relu",
    #                                       name="pyramid_{}_classification_{}".format(pyramid_idx, i),
    #                                       kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #                                       bias_initializer="zeros",
    #                                       **options)(outputs)
    #     outputs = keras.layers.Conv2D(filters=num_classes*num_anchors,
    #                                   kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #                                   bias_initializer=initializers.PriorProbability(probability=prior_probability),
    #                                   name="pyramid_{}_pyramid_classification".format(pyramid_idx),
    #                                   **options)(outputs)
    #     if keras.backend.image_data_format() == "channels_first":
    #         resize = (3, 2, 1)
    #     else:
    #         resize = (2, 1, 3)
    #     outputs = keras.layers.Permute(resize, name="pyramid_{}_classification_permute"
    #                                    .format(pyramid_idx))(outputs)
    #     outputs = keras.layers.Reshape((-1, num_classes), name="pyramid_{}_classification_reshape"
    #                                    .format(pyramid_idx))(outputs)
    #     outputs = keras.layers.Activation("sigmoid", name="pyramid_{}_classification_sigmoid"
    #                                       .format(pyramid_idx))(outputs)
    #     return outputs
    #
    # return _classification_task

def detections(num_classes, num_anchors):
    regression = regression_task(4, num_anchors)
    classification = classification_task(num_classes, num_anchors)
    return {
        "regression": regression,
        "classification": classification,
    }

##############################################################################
def build_pyramid(models, features):
    res = []
    for name, model in models.items():
        res.append(keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features]))
    return res

def retinanet(inputs, backbone_layers, num_classes,
              num_anchors=None,
              create_pyramid_features=create_pyramid_features,
              tasks=detections,
              name="retinanet"):

    if num_anchors is None:
        num_anchors = AnchorConfig.default.num_anchors()

    c3, c4, c5 = backbone_layers

    features = create_pyramid_features(c3, c4, c5)

    # [regression, classification]
    # regression: (batch_number, width*height*all_pyramid_anchor_number, 4)
    # classification: (batch_number, width*height*all_pyramid_anchor_number, number_classes)
    pyramids = build_pyramid(tasks(num_classes, num_anchors), features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)

def build_anchors(anchor_config, features):

    anchors = []
    for i, f in enumerate(features):
        anchors.append(layers.Anchors(size=anchor_config.sizes[i],
                                      stride=anchor_config.strides[i],
                                      ratios=anchor_config.ratios,
                                      scales=anchor_config.scales,
                                       name="anchors_{}".format(i))(f))

    # (batch_number, all_pyramid_anchor_numbers, 4)
    return keras.layers.Concatenate(axis=1, name="anchors")(anchors)

def retinanet_bbox(model, nms=True,
                   class_specific_filter=True,
                   name="retinanet-bbox",
                   anchor_config=None):

    if anchor_config is None:
        anchor_config = parse_anchor_parameter()

    features = [model.get_layer(name).output for name in ["r3", "r4", "r5", "r6", "r7"]]
    anchors = build_anchors(anchor_config, features)

    regression = model.outputs[0]
    # [batch_size, width*height*all_pyramid_anchor_number, num_classes]
    classification = model.outputs[1]

    # [batch_size, width*height*all_pyramid_anchor_number, 4]
    boxes = layers.BBox(name="boxes")([anchors, regression])
    # [batch_size, width*height*all_pyramid_anchor_number, 4]
    boxes = layers.ClipBox(name="clipped_boxes")([model.inputs[0], boxes])

    detections_task = layers.Filter(nms=nms,
                               class_specific_filter=class_specific_filter,
                               name="filtered_detection")([boxes, classification])

    return keras.models.Model(inputs=model.inputs, outputs=detections_task, name=name)




