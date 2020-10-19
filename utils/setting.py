#-*-coding:utf-8-*-
from config.fileconfig import cfg
from utils.anchors import AnchorConfig

import numpy as np
import keras

def parse_anchor_parameter():

    ratio = np.array(list(map(float, cfg.ANCHOR.RATIOS.split(" "))), keras.backend.floatx())
    scales = np.array(list(map(lambda x: 2**(float(x)/3.0), cfg.ANCHOR.SCALES.split(" "))), keras.backend.floatx())

    sizes = list(map(int, cfg.ANCHOR.SIZES.split(" ")))
    strides = list(map(int, cfg.ANCHOR.STRIDES.split(" ")))
    pyramid_levels = list(map(int, cfg.ANCHOR.PYRAMID_LEVELS.split(" ")))

    return AnchorConfig(sizes, strides, ratio, scales, pyramid_levels)

if __name__ == "__main__":
    print("hello")