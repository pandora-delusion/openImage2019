#-*-coding:utf-8-*-

import keras

import numpy as np
import math

class PriorProbability(keras.initializers.Initializer):

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            "probability": self.probability
        }

    def __call__(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)* -math.log((1-self.probability)/self.probability)