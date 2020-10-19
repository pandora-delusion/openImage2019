#-*-coding:utf-8-*-

import progressbar
import keras
import numpy as np
from utils.compute_overlap import compute_overlap
import random

def get_detections(generator, model, score_threshold=0.05, max_detection=100, save_path=None):

    all_detections = [[None for i in range(generator.num_classes())
                            if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix="Running network: "):
        image = generator.load_image(i)
        image = generator.preprocess_image(image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == "channels_first":
            image = image.transpose((2, 0, 1))

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        boxes /= scale

        indices = np.where(scores[0, :]> score_threshold)[0]

        sorted_idx = np.argsort(-scores)[:max_detection]

        image_bboxes = boxes[0, indices[sorted_idx], :]
        image_scores = scores[sorted_idx]
        image_labels = labels[0, indices[sorted_idx]]

        detections = np.concatenate([image_bboxes, np.expand_dims(image_scores, axis=1),
                                     np.expand_dims(image_labels, axis=1)], axis=1)

        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = detections[detections[:, -1] == label, :-1]

    return all_detections

def handle_detections(generator, model, score_threshold=0.05, iou_threshold=0.5, idx=0, number=10000):
    all_annotations = dict.fromkeys(generator.idx2class, 0)
    all_scores = dict.fromkeys(generator.idx2class, [])
    false_positive = dict.fromkeys(generator.idx2class, [])
    true_positive = dict.fromkeys(generator.idx2class, [])

    limit = generator.size()
    if idx+number>=limit:
        number = limit
    else:
        number += idx

    for i in range(idx, number):
        # image = generator.load_image(i)
        image, gt_labels, gt_boxes = generator[i]
        image = generator.preprocess_image(image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == "channels_first":
            image = image.transpose((2, 0, 1))

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        boxes /= scale

        indices = np.where(scores[0, :]>score_threshold)[0]
        boxes = boxes[0, indices, :]
        scores = scores[0, indices]
        labels = labels[0, indices]

        sorted_indices = np.argsort(-scores)
        boxes = boxes[sorted_indices, :]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        detected_annotations = []

        for idx in range(len(sorted_indices)):
            all_scores[labels[idx]].append(scores[idx])

            if gt_boxes.shape[0] == 0:
                false_positive[labels[idx]].append(1)
                true_positive[labels[idx]].append(0)
                continue

            # annotations = annotations.astype(np.float64)
            box = np.expand_dims(boxes[idx], axis=0).astype(np.float64)
            overlaps = compute_overlap(box, gt_boxes)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]
            # print(max_overlap)

            if max_overlap > iou_threshold and assigned_annotation not in detected_annotations:
                false_positive[labels[idx]].append(0)
                true_positive[labels[idx]].append(1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positive[labels[idx]].append(1)
                true_positive[labels[idx]].append(0)

        for label in gt_labels:
            all_annotations[label] += 1

    return all_annotations, all_scores, true_positive, false_positive

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(generator, model, score_threshold=0.05, iou_threshold=0.5, idx=0, number=10000):

    annotations, scores, true_positive, false_positive = handle_detections(generator, model,
                                                                           score_threshold,
                                                                           iou_threshold,
                                                                           idx,
                                                                           number)
    average_precisions = {}
    for label in scores:
        if len(scores[label]) <= 0:
            continue

        label_scores = np.array(scores[label])
        label_false = np.array(false_positive[label])
        label_true = np.array(true_positive[label])

        indices = np.argsort(-label_scores)
        label_false = label_false[indices]
        label_true = label_true[indices]

        # print(np.where(label_true>0)[0])

        label_false = np.cumsum(label_false)
        label_true = np.cumsum(label_true)

        if annotations[label] <= 0:
            recall = np.zeros_like(label_true)
        else:
            recall = label_true/annotations[label]
        precision = label_true/np.maximum(label_true+label_false, np.finfo(np.float64).eps)

        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision, annotations[label]

    return average_precisions

# import time
#
# def detections(generator, model, score_threshold=0.16, max_detections=10, number=100):
#
#     indices = random.sample(range(generator.size()), number)
#
#     image_groups = []
#     bboxe_groups = []
#     score_groups = []
#     label_groups = []
#
#     for idx in indices:
#
#         start = time.time()
#
#         image, gt_labels, gt_boxes = generator[idx]
#         image_for_process = generator.preprocess_image(image.copy())
#         image_for_process, scale = generator.resize_image(image_for_process)
#         # print(scale)
#
#         if keras.backend.image_data_format() == "channels_first":
#             image_for_process = image_for_process.transpose((2, 0, 1))
#
#         boxes, scores, labels = model.predict_on_batch(np.expand_dims(image_for_process, axis=0))[:3]
#
#         boxes /= scale
#
#         _idx = np.where(scores>score_threshold)[1]
#
#         boxes = boxes[0, _idx].astype(np.float64)
#         scores = scores[0, _idx]
#         labels = labels[0, _idx]
#
#         overlap = compute_overlap(boxes, boxes)
#         fathers = []
#         boxes_number = boxes.shape[0]
#         for i in range(0, boxes_number):
#             for j in range(i, boxes_number):
#                 father, _ = generator.judge_similar(labels[i], labels[j])
#                 if father != -1 and overlap[i, j] > 0.5:
#                     fathers.append(father)
#         fathers = set(fathers)
#         # print(fathers)
#
#         sons = list(set(range(boxes_number)) - fathers)
#         sons.sort()
#         boxes = boxes[sons]
#         scores = scores[sons]
#         labels = labels[sons]
#
#         # scores = scores[_idx]
#
#         scores_sort = np.argsort(-scores)[:max_detections]
#
#         image_bboxes = boxes[scores_sort, :]
#         image_scores = scores[scores_sort]
#         image_labels = labels[scores_sort]
#
#         image_groups.append(image)
#         bboxe_groups.append(image_bboxes)
#         score_groups.append(image_scores)
#         label_groups.append(image_labels)
#
#         print(time.time()-start)
#
#     return image_groups, bboxe_groups, score_groups, label_groups

def detections(generator, image, model, score_threshold=0.16, max_detections=300):

    image_for_process = generator.preprocess_image(image.copy())
    image_for_process, scale = generator.resize_image(image_for_process)

    if keras.backend.image_data_format() == "channels_first":
        image_for_process = image_for_process.transpose((2, 0, 1))

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image_for_process, axis=0))[:3]

    boxes /= scale

    _idx = np.where(scores>score_threshold)[1]

    boxes = boxes[0, _idx].astype(np.float64)
    scores = scores[0, _idx]
    labels = labels[0, _idx]

    overlap = compute_overlap(boxes, boxes)
    fathers = []
    boxes_number = boxes.shape[0]
    for i in range(0, boxes_number):
        for j in range(i, boxes_number):
            father, _ = generator.judge_similar(labels[i], labels[j])
            if father != -1 and overlap[i, j] > 0.5:
                fathers.append(father)
    fathers = set(fathers)

    sons = list(set(range(boxes_number)) - fathers)
    sons.sort()
    boxes = boxes[sons]
    scores = scores[sons]
    labels = labels[sons]

    # scores = scores[_idx]

    scores_sort = np.argsort(-scores)[:max_detections]

    image_bboxes = boxes[scores_sort, :]
    image_scores = scores[scores_sort]
    image_labels = labels[scores_sort]

    height = image.shape[0]
    width = image.shape[1]

    # image_bboxes[:, 0:3:2] /= width
    # image_bboxes[:, 1:4:2] /= height

    return image_bboxes, image_scores, image_labels