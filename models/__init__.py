#-*-coding:utf-8-*-

import layers
import losses
import initializers

class Model(object):

    def __init__(self, backbone):
        self.backbone = backbone

        self.custom_objects = {
            "UpSampleLayer": layers.UpSampleLayer,
            "PriorProbability": initializers.PriorProbability,
            "BBox": layers.BBox,
            "Filter": layers.Filter,
            "Anchors": layers.Anchors,
            "ClipBox": layers.ClipBox,
            "_smooth_l1": losses.smooth_l1_initializer(),
            "_focal": losses.focal_loss_initializer(),
        }

    def net(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess_image(self, inputs):
        raise NotImplementedError

def model(backbone):

    if 'resnet' in backbone:
        from .resnet import ResNet as m
    elif "mobilenet" in backbone:
        from .mobilenet import MobileNet as m
    else:
        raise NotImplementedError
    return m(backbone)

def load_model(filepath, backbone_name):
    import keras.models
    return keras.models.load_model(filepath, custom_objects=model(backbone_name).custom_objects, compile=False)

def predict_model(m, nms=True, class_specific_filter=True, anchor_config=None):
    from .retinanet import retinanet_bbox
    return retinanet_bbox(model=m, nms=nms, class_specific_filter=class_specific_filter, anchor_config=anchor_config)



