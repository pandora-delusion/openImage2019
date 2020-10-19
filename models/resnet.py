#-*-coding:utf-8-*-

import keras
from keras.utils import get_file
import keras_resnet
import keras_resnet.models

from . import Model
from . import retinanet
from utils.image import preprocess_image


def resnet_retinanet(num_classes, backbone="resnet50", inputs=None, **kwargs):

    if inputs is None:
        if keras.backend.image_data_format() == "channels_first":
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))
    print("backbone: %s" % backbone)

    if backbone == "resnet50":
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == "resnet101":
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == "resnet152":
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError("model ({}) is invalid".format(backbone))

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


class ResNet(Model):

    def __init__(self, backbone):
        super(ResNet, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def net(self, *args, **kwargs):
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def preprocess_image(self, inputs):

        return preprocess_image(inputs, mode="caffe")

