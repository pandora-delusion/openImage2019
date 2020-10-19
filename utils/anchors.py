#-*-coding:utf-8-*-

import keras
import numpy as np
from utils.compute_overlap import compute_overlap

class AnchorConfig:

    def __init__(self, sizes, strides, ratios, scales, pyramid_levels):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.pyramid_levels = pyramid_levels

    def num_anchors(self):
        return len(self.ratios)*len(self.scales)

AnchorConfig.default = AnchorConfig(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    pyramid_levels =  [3, 4, 5, 6, 7]
)

def guess_shapes(image_shape, pyramid_levels):

    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2**x-1) // (2**x) for x in pyramid_levels]
    return image_shapes

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    :param base_size:
    :param ratios:
    :param scales:
    :return: (xmin, ymin, xmax, ymax)
    """
    if ratios is None:
        ratios = AnchorConfig.default.ratios

    if scales is None:
        scales = AnchorConfig.default.scales

    num_anchors = len(ratios)*len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size*np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2]*anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2]*np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2]*0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3]*0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):

    shift_x, shift_y = np.mgrid[0:shape[1], 0:shape[0]]
    shift_x = (shift_x + 0.5)*stride
    shift_y = (shift_y + 0.5)*stride

    shift = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel())).transpose()

    anchors_num = anchors.shape[0]
    shifts_num = shift.shape[0]
    # (shifts_num, anchors_num, 4)
    anchors = (anchors.reshape((1, anchors_num, 4)) + shift.reshape((1, shifts_num, 4)).transpose((1, 0, 2)))

    anchors = anchors.reshape((anchors_num*shifts_num, 4))

    return anchors

def anchors_for_shape(image_shape,
                      anchor_params=None,
                      shapes_callback=None):
    if anchor_params is None:
        anchor_params = AnchorConfig.default

    pyramid_levels = anchor_params.pyramid_levels

    if shapes_callback is None:
        shapes_callback = guess_shapes

    image_shapes = shapes_callback(image_shape, pyramid_levels)

    all_anchors = np.zeros((0, 4))
    for idx in range(len(pyramid_levels)):
        anchors = generate_anchors(anchor_params.sizes[idx],
                                   anchor_params.ratios,
                                   anchor_params.scales)
        shift_anchors = shift(image_shapes[idx], anchor_params.strides[idx],
                              anchors)
        all_anchors = np.append(all_anchors, shift_anchors, axis=0)

    return all_anchors

def bbox_for_anchors(image_shape, annotations):
    """
    xmin, xmax, ymin, ymax -> xmin, ymin, xmax, ymax
    :param image_shape:
    :param annotations:
    :return:
    """
    height = image_shape[0]
    width = image_shape[1]
    bbox = np.zeros((annotations.shape[0], 4))
    bbox[:, 0] = annotations[:, 0]
    bbox[:, 1] = annotations[:, 2]
    bbox[:, 2] = annotations[:, 1]
    bbox[:, 3] = annotations[:, 3]

    return bbox

def get_gt_indices(anchors, annotations, positive_overlap=0.5,
                   negative_overlap=0.4):
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    best_overlap_each_anchor_indices = np.argmax(overlaps, axis=1) # (M, )
    best_overlap_each_anchor = overlaps[np.arange(overlaps.shape[0]), best_overlap_each_anchor_indices] # (M, )

    positive_indices = best_overlap_each_anchor >= positive_overlap
    ignore_indices = (best_overlap_each_anchor > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, best_overlap_each_anchor_indices

def anchor_targets_bbox(anchors, image_group, annotations_group,
                        labels_group, num_classes,
                        negative_overlap=0.4,
                        positive_overlap=0.5):
    assert len(image_group) == len(annotations_group) == len(labels_group)
    assert len(annotations_group)>0

    batch_size = len(image_group)

    regression = np.zeros((batch_size, anchors.shape[0], 4+1), dtype=keras.backend.floatx())
    classification = np.zeros((batch_size, anchors.shape[0], num_classes+1), dtype=keras.backend.floatx())

    for idx, (image, annotations, labels) in enumerate(zip(image_group, annotations_group, labels_group)):
        # xmin, xmax, ymin, ymax -> xmin, ymin, xmax, ymax
        bbox = bbox_for_anchors(image.shape, annotations[:, 1:5])
        positive_indices, ignore_indices, best_overlap_indices = get_gt_indices(anchors, bbox, positive_overlap,
                                                                                negative_overlap)
        classification[idx, ignore_indices, -1] = -1
        classification[idx, positive_indices, -1] = 1

        regression[idx, ignore_indices, -1] = -1
        regression[idx, positive_indices, -1] = 1

        p_idx = best_overlap_indices[positive_indices]
        anchor_indices = np.where(positive_indices)[0]

        for i, a_idx in enumerate(anchor_indices):
            classification[idx, a_idx, labels[p_idx[i]]] = 1

        regression[idx, :, :-1] = bbox_transform(anchors, bbox[best_overlap_indices, :])
    return regression, classification

def bbox_transform(anchors, gt_boxes, prior_scaling=None):
    if prior_scaling is None:
        prior_scaling = [0.1, 0.1, 0.2, 0.2]

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    anchor_cx = (anchors[:, 2] + anchors[:, 0])/2.0
    anchor_cy = (anchors[:, 3] + anchors[:, 1])/2.0

    bbox_cx = (gt_boxes[:, 2]+gt_boxes[:, 0])/2.0
    bbox_cy = (gt_boxes[:, 3]+gt_boxes[:, 1])/2.0

    bbox_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    bbox_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

    target_cx = (bbox_cx-anchor_cx)/anchor_widths/prior_scaling[0]
    target_cy = (bbox_cy-anchor_cy)/anchor_heights/prior_scaling[1]
    target_width = np.log(bbox_widths/anchor_widths)/prior_scaling[2]
    target_height = np.log(bbox_heights/anchor_heights)/prior_scaling[3]

    targets = np.stack((target_cx, target_cy,
                        target_width, target_height), axis=-1)

    return targets