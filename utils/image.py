#-*-coding:utf-8-*-

from __future__ import division
import numpy as np
import cv2
from PIL import Image
import piexif
import warnings

warnings.filterwarnings("error", category=UserWarning)

def read_image_bgr(path):
    """
    读取图片为BGR的格式
    :param path:
    :return:
    """
    # warnings.filterwarnings("error")
    try:
        image = np.asarray(Image.open(path).convert("RGB"))
    except UserWarning:
        print("rewrite exif warning file: {}".format(path))
        piexif.remove(path)
        image = np.asarray(Image.open(path).convert("RGB"))
    # warnings.filterwarnings("default")
    return image[:, :, ::-1].copy()

class TransformParameters:

    def __init__(self,
                 fill_mode='nearest',
                 interpolation="linear",
                 cval=0,
                 relative_translation=True):
        # 边界填充模式
        self.fill_mode = fill_mode

        self.cval = cval
        # 插值模式
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == "constant":
            return cv2.BORDER_CONSTANT
        elif self.fill_mode == "nearest":
            return cv2.BORDER_REPLICATE
        elif self.fill_mode == "reflect":
            return cv2.BORDER_REFLECT_101
        elif self.fill_mode == "wrap":
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        # 最邻近插值
        if self.interpolation == "nearest":
            return cv2.INTER_NEAREST
        # 线性插值
        elif self.interpolation == "linear":
            return cv2.INTER_LINEAR
        # 三次样条插值
        elif self.interpolation == "cubic":
            return cv2.INTER_CUBIC
        # 区域插值
        elif self.interpolation == "area":
            return cv2.INTER_AREA
        # lanczos插值
        elif self.interpolation == "lanczos4":
            return cv2.INTER_LANCZOS4

def apply_transform(matrix, image, params=TransformParameters()):
    # matrix, _, _ = matrix
    res = cv2.warpAffine(image, matrix[:2, :],
                         dsize=(image.shape[1], image.shape[0]),
                         flags=params.cvInterpolation(),
                         borderMode=params.cvBorderMode(),
                         borderValue=params.cval)
    return res

def preprocess_image(x, mode="caffe"):
    x = x.astype(np.float32)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def compute_resize_scales(image_shape, min_side=800, max_side=1333):
    width, height, _ = image_shape
    small_side = min(width, height)

    scale = min_side/small_side

    large_side = max(width, height)

    if large_side*scale > max_side:
        scale = max_side/large_side

    return scale

def resize_image(img, min_side=800, max_side=1333):

    scale = compute_resize_scales(img.shape, min_side=min_side, max_side=max_side)

    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def clip(image):

    return np.clip(image, 0, 255).astype(np.uint8)

def adjust_constrast(image, factor):
    """
    调整图像的明暗对比
    :param image:
    :param factor:
    :return:
    """
    mean = image.mean(axis=0).mean(axis=0)
    return clip((image-mean)*factor + mean)

def adjust_brightness(image, delta):
    """
    调整图像的亮度
    :param image:
    :param delta:
    :return:
    """
    return clip(image+delta*255)

def adjust_hue(image, delta):
    """
    调整图像的颜色，delta在-1和1之间
    :param image:
    :param delta:
    :return:
    """
    image[..., 0] = np.mod(image[..., 0]+delta*180, 180)
    return image

def adjust_saturation(image, factor):
    """
    调整色彩饱和度
    :param image:
    :param factor:
    :return:
    """
    image[..., 1] = np.clip(image[..., 1]*factor, 0, 255)
    return image

class VisualEffect:

    def __init__(self,
                 constrast_factor,
                 brightness_delta,
                 hue_delta,
                 saturation_factor):
        self.constrast_factor = constrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):

        if self.constrast_factor:
            image = adjust_constrast(image, self.constrast_factor)
        if self.brightness_delta:
            image = adjust_brightness(image, self.brightness_delta)

        if self.hue_delta or self.saturation_factor:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

def uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])

def random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-0.1, 0.1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
):
    def _generate():
        while True:
            yield VisualEffect(
                constrast_factor=uniform(contrast_range),
                brightness_delta=uniform(brightness_range),
                hue_delta=uniform(hue_range),
                saturation_factor=uniform(saturation_range),
            )

    return _generate()

if __name__ == "__main__":
    read_image_bgr("/media/user/disk2/delusion/oid/train/train_0/0a0a00b2fbe89a47.jpg")