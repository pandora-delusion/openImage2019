# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

from easydict import EasyDict
import os

__C = EasyDict()

cfg = __C

__C.FILE = EasyDict()
# __C.FILE.BASE_DIR = os.path.abspath("..") + "/"
__C.FILE.BASE_DIR = "/media/user/disk2/delusion/openImage2019/"
__C.FILE.CLASS_HIERARCHY_FILE = "datasets/challenge-2019-label500-hierarchy.json"
__C.FILE.TRAIN_BBOX_DIR = "data/labels/train/"
__C.FILE.VAL_BBOX_DIR="data/labels/validation/"
__C.FILE.TRAIN_BBOX_FILE = "datasets/challenge-2019-train-detection-bbox.csv"
__C.FILE.VAL_BBOX_FILE = "datasets/challenge-2019-validation-detection-bbox.csv"
__C.FILE.BBOX_DIR = "data/labels/"
__C.FILE.IMAGES_TRAIN_DIR = "data/oid/train/"
__C.FILE.IMAGES_VALIDATION_DIR = "data/oid/validation/"
__C.FILE.TRAIN_INFO = "data/train_info.txt"
__C.FILE.TENSORBOARD_DIR = "log/"
__C.FILE.SNAPSHOT_PATH = "checkpoint/"

__C.BBOX = EasyDict()
__C.BBOX.CSV_IMAGEID = "ImageID"
__C.BBOX.CSV_SOURCE = "Source"
__C.BBOX.CSV_LABELNAME = "LabelName"
__C.BBOX.CSV_CONFIDENCE = "Confidence"
__C.BBOX.CSV_XMIN = "XMin"
__C.BBOX.CSV_XMAX = "XMax"
__C.BBOX.CSV_YMIN = "YMin"
__C.BBOX.CSV_YMAX = "YMax"
__C.BBOX.CSV_ISOCCLUDED = "IsOccluded"
__C.BBOX.CSV_ISTRUNCATES = "IsTruncated"
__C.BBOX.CSV_ISGROUPOF = "IsGroupOf"
__C.BBOX.CSV_ISDEPICTION = "IsDepiction"
__C.BBOX.CSV_ISINSIDE = "IsInside"

__C.ANCHOR = EasyDict()
__C.ANCHOR.RATIOS = "0.5 1 2"
__C.ANCHOR.SCALES = "0 1 2"
__C.ANCHOR.SIZES = "16 32 64 128 256"
__C.ANCHOR.STRIDES = "4 8 16 32 64"
__C.ANCHOR.PYRAMID_LEVELS = "3 4 5 6 7"

__C.TRAIN = EasyDict()
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.IMAGE_MIN_SIDE = 448
__C.TRAIN.IMAGE_MAX_SIDE = 448
__C.TRAIN.IMAGENET_RESNET50_WEIGHTS = "data/weights/ResNet-50-model.keras.h5"
__C.TRAIN.IMAGENET_RESNET152_WEIGHTS = "data/weights/ResNet-152-model.keras.h5"
__C.TRAIN.IMAGENET_MOBILENET224_1_0 = "data/weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
__C.TRAIN.GPU = "0,1,2,3"
__C.TRAIN.MULTI_GPU = 4
__C.TRAIN.LEARNING_RATE = 1e-5
__C.TRAIN.EPOCHS = 1
__C.TRAIN.WORKERS = 32
__C.TRAIN.MULTIPROCESSING = True
__C.TRAIN.MAX_QUEUE_SIZE = 32

__C.VAL = EasyDict()
__C.VAL.IOU_THRESHOLD = 0.5
__C.VAL.SCORE_THRESHOLD = 0.05

__C.TEST = EasyDict()
__C.TEST.IMAGE_DIR = "data/oid/test/"

