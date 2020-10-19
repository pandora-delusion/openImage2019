#-*-coding:utf-8-*-

import os
import keras
import tensorflow as tf

from config.fileconfig import cfg
from preprocessing.loader import ValLoader
from utils.image import preprocess_image
import models
from backend.evaluate import evaluate

def create_generator():

    common_args = {
        "batch_size":1,
        "image_min_side": cfg.TRAIN.IMAGE_MIN_SIDE,
        "image_max_side": cfg.TRAIN.IMAGE_MAX_SIDE,
        "preprocess_image": preprocess_image,
    }

    validation_generator = ValLoader(**common_args)

    return validation_generator

def main(filename):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = tf.ConfigProto()
    session = tf.Session(config=config)

    keras.backend.tensorflow_backend.set_session(session)

    generator = create_generator()

    with tf.device("/cpu:0"):
        model = models.load_model(cfg.FILE.BASE_DIR + "checkpoint/" + filename, backbone_name="resnet50")

    model = models.predict_model(model)

    average_precisions = evaluate(generator, model, iou_threshold=cfg.VAL.IOU_THRESHOLD,
                                  score_threshold=cfg.VAL.SCORE_THRESHOLD, number=10000)

    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print("{:.0f} instances of class".format(num_annotations),
              generator.label_to_name(label), "with average precision: {:.4f}".format(average_precision))

        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) <= 0:
        print("No test instances found.")
        return

    print("mAP using the weighted average of precisions among classes: {:.4f}"
          .format(sum([a*b for a, b in zip(total_instances, precisions)])/sum(total_instances)))
    print("mAP: {:.4f}".format(sum(precisions)/sum(x >0 for x in total_instances)))

def draw():
    from config.fileconfig import cfg
    from backend.evaluate import detections
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU

    config = tf.ConfigProto()
    session = tf.Session(config=config)

    keras.backend.tensorflow_backend.set_session(session)

    from utils.setting import parse_anchor_parameter
    anchor_config = parse_anchor_parameter()

    num_anchors = anchor_config.num_anchors()
    from config.fileconfig import cfg

    backbone = models.model("resnet50")

    generator = create_generator()
    num_classes = generator.num_classes()
    # model = backbone.net(num_classes, num_anchors=num_anchors)

    with tf.device("/gpu:0"):
        model = models.load_model(cfg.FILE.BASE_DIR + "checkpoint/resnet50_01_alpha.h5",
                              backbone_name="resnet50")
    # model = keras.utils.multi_gpu_model(model, gpus=cfg.TRAIN.MULTI_GPU)

    model.summary()

    model = models.predict_model(model)

    # ################################################
    test_dir = cfg.FILE.BASE_DIR + cfg.TEST.IMAGE_DIR

    image_generator = imageGenerator(test_dir)
    #
    # import csv
    #
    # csvfile = open("submission.csv", "w")
    # writer = csv.writer(csvfile)
    # writer.writerow(["ImageId", "PredictionString"])
    # for id, image in image_generator:
    #     image_bboxes, image_scores, image_labels = detections(generator, image, model)
    #
    #     box_number = image_bboxes.shape[0]
    #
    #     strings = []
    #
    #     print(id)
    #
    #     for idx in range(box_number):
    #         label = int(image_labels[idx])
    #         label = generator.idx2class[label]
    #         strings.append("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
    #                        .format(label, image_scores[idx], image_bboxes[idx, 0],
    #                                image_bboxes[idx, 1], image_bboxes[idx][2], image_bboxes[idx, 3]))
    #     predict_string = " ".join(strings)
    #     writer.writerow([id, predict_string])
    #
    # csvfile.close()
    # ##############################################

    from backend.evaluate import detections

    number = 10

    from utils.visualization import plt_bboxes

    for i in range(number):
        id, image = next(image_generator)
        print("id: ", id)
        image_boxes, image_scores, image_labels = detections(generator, image, model)

        labels = []
        for label in image_labels:
            labels.append(generator.label_to_name(label))

        print(" ".join(labels))

        plt_bboxes(image, image_labels, image_scores, image_boxes)

from utils.image import read_image_bgr

def imageGenerator(image_dir):

    with os.scandir(image_dir) as scanner:
        for entry in scanner:
            filename = entry.name
            filepath = image_dir + filename
            image = read_image_bgr(filepath)
            id = filename.replace(".jpg", "")
            yield id, image

if __name__ == "__main__":
    import pandas as pd
    #
    # # data = pd.read_csv("submission_s.csv", header=None, names=["ImageId", "PredictionString"])
    # # data = data
    # # data = pd.read_csv("submission_c.csv")
    # # head = data.head(100000-1)
    # # write_file = open("submission.csv", "w", newline="")
    # # head.to_csv(write_file, index=False)
    # data = pd.read_csv("submission.csv")
    # print(data)
    draw()