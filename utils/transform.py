#-*-coding:utf-8-*-

import numpy as np

def transform_coordinate(transform, aabb):
    xmin, xmax, ymin, ymax = aabb

    points = transform.dot([
        [xmin, xmax, xmin, xmax],
        [ymin, ymax, ymax, ymin],
        [1, 1, 1, 1],
    ])
    # points[0, :] += int(_x)
    # points[1, :] += int(_y)

    min_corner = points[0:2, :].min(axis=1)
    max_corner = points[0:2, :].max(axis=1)

    return [min_corner[0], max_corner[0], min_corner[1], max_corner[1]]

def rotation(angle):
    """
    构造一个二维旋转矩阵
    :param angle:
    :return:
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def random_rotation(min, max, default=np.random):
    return rotation(default.uniform(min, max))

def translation(translation):
    """
    构造一个二维平移矩阵
    :param translation:
    :return:
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

def random_vector(min, max, default=np.random):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    return default.uniform(min, max)

def random_translation(min, max, default=np.random):
    return translation(random_vector(min, max, default))

def shear(angle):
    """
    构造一个二维shear矩阵
    :param angle:
    :return:
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])

def random_shear(min, max, default=np.random):
    return shear(default.uniform(min, max))

def scaling(factor):
    """
    构造二维缩放矩阵
    :param factor:
    :return:
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])

def random_scaling(min, max, default=np.random):
    return scaling(random_vector(min, max, default))

def random_flip(flip_x, flip_y, default=np.random):
    _x = default.uniform(0, 1) < flip_x
    _y = default.uniform(0, 1) < flip_y
    _x = int(_x)
    _y = int(_y)
    return np.array([
        [(-1)**_x, 0, _x],
        [0, (-1)**_y, _y],
        [0, 0, 1]
    ])

def random_transform(default=np.random,
                     min_rotation=0,
                     max_rotation=0,
                     min_translation=(0, 0),
                     max_translation=(0, 0),
                     min_shear=0,
                     max_shear=0,
                     min_scaling=(1, 1),
                     max_scaling=(1, 1),
                     flip_x_chance=0,
                     flip_y_chance=0):
    res= np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, default),
        random_translation(min_translation, max_translation, default),
        random_shear(min_shear, max_shear, default),
        random_scaling(min_scaling, max_scaling, default),
        random_flip(flip_x_chance, flip_y_chance, default)
    ])
    return res

def random_transform_generator(default=np.random, **kwargs):
    # if not default:
    #     default = np.random.RandomState()
    #
    # while True:
    #     yield random_transform(default=default, **kwargs)
    def _generator(width, height):
        res = random_transform(default, **kwargs)
        if 'flip_x_chance' in kwargs:
            res[0, 2] *= width
        if 'flip_y_chance' in kwargs:
            res[1, 2] *= height

        return res
    return _generator



