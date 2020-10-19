#-*-coding:utf-8-*-

import keras
from keras.applications import mobilenet_v2
from .retinanet import retinanet
from . import Model
from utils.image import preprocess_image

def mobilenet_retinanet(num_classes, backbone="mobilenet224_1.0", inputs=None, **kwargs):

    alpha = float(backbone.split("_")[1])

    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    backbone = mobilenet_v2.MobileNetV2(input_tensor=inputs, alpha=alpha, include_top=False,
                                        pooling=None, weights=None)

    # layer_names = ["conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
    layer_names = ["block_11_project_BN", "block_14_project_BN", "block_16_project_BN"]

    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)

class MobileNet(Model):

    def net(self, *args, **kwargs):
        return mobilenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def preprocess_image(self, inputs):
        return preprocess_image(inputs, mode="tf")

