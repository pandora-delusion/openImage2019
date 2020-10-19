#-*-coding:utf-8-*-

import tensorflow as tf

tf.enable_eager_execution()

data = tf.ones((2, 3), dtype=tf.int32)

data = tf.assign_add(data, 1)

print(data)