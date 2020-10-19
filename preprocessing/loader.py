#-*-coding:utf-8-*-

import numpy as np
import random
import os
import time

import  keras

from config.fileconfig import cfg
from utils.image import read_image_bgr, apply_transform, TransformParameters, \
    preprocess_image, resize_image
from utils.transform import transform_coordinate
from utils.setting import parse_anchor_parameter
from utils.anchors import anchors_for_shape, guess_shapes, anchor_targets_bbox, bbox_for_anchors
from utils.datasets import get_multi_labels_hierarchy_and_classes, handle_multi_labels
import models

class ImageLoader(keras.utils.Sequence):

    def __init__(self, transform_generator=None,
                 visual_effect_generator=None,
                 batch_size=cfg.TRAIN.BATCH_SIZE,
                 group_method="random",
                 image_min_side=cfg.TRAIN.IMAGE_MIN_SIDE,
                 image_max_side=cfg.TRAIN.IMAGE_MAX_SIDE,
                 transform_parameters=None,
                 preprocess_image=preprocess_image,
                 handle_targets=anchor_targets_bbox,
                 compute_shapes=guess_shapes,
                 multi_label_handler=handle_multi_labels):

        self.transform_generator = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.batch_size = batch_size
        self.group_method = group_method
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()
        self.preprocess_image = preprocess_image
        self.compute_shapes=compute_shapes
        self.handle_targets = handle_targets
        self.multi_label_handler = multi_label_handler
        # self.analyze_length()
        # self.groups_images()
        #
        # if self.shuffle_groups:
        #     self.on_epoch_end()
        #
        # self.load_class()
        self.length = self.get_length()
        self.groups = self.groups_images()
        self.class2idx, self.idx2class, self.class2name, self.label_hierarchy = self.load_classes()

    def groups_images(self):
        order = list(range(self.length))
        if self.group_method == "random":
            random.shuffle(order)

        return [[order[x % self.length] for x in range(i, i + self.batch_size)]
                                            for i in range(0, self.length, self.batch_size)]
    def size(self):
        raise NotImplementedError

    def has_label(self, label):
        raise NotImplementedError

    def has_class(self, cls):
        raise NotImplementedError

    def class_to_label(self, name):
        raise NotImplementedError

    def label_to_class(self, label):
        raise NotImplementedError

    def get_length(self):
        raise NotImplementedError

    def load_classes(self):
        raise NotImplementedError

    def load_data(self, idx):
        raise NotImplementedError

    def num_classes(self):
        return len(self.class2idx)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        group = self.groups[item]
        inputs, target = self.load_image_and_annotation(group)
        return inputs, target

    def load_image_item(self, file):
        return read_image_bgr(file)

    def load_annotations_item(self, key, id):
        select_file = "%s/%s.txt" % (key, id)
        return np.loadtxt(select_file, ndmin=2, dtype=np.float64)

    def random_visual_effect_group_item(self, image, annotations):
        visual_effect = next(self.visual_effect_generator)
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, images_group, annotations_group):
        assert(len(images_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            return images_group, annotations_group

        for idx in range(len(images_group)):
            images_group[idx], annotations_group[idx] = self.random_visual_effect_group_item(images_group[idx],
                                                                                             annotations_group[idx])

        return images_group, annotations_group

    def calculate_annotations(self, annotations, width, height):
        annotations[:, 1:3] *= width
        annotations[:, 3:] *= height

    def calculate_annotation_group(self, annotation_group, image_group):
        for idx, image in enumerate(image_group):
            height = image.shape[0]
            width = image.shape[1]
            self.calculate_annotations(annotation_group[idx], width, height)

    def random_transform_group_item(self, image, annotations):
        height = image.shape[0]
        width = image.shape[1]
        if self.transform_generator is not None:
            transform = self.transform_generator(width, height)
        else:
            return image, annotations

        image = apply_transform(transform, image, self.transform_parameters)
        for idx in range(annotations.shape[0]):
            annotations[idx, 1:5] = transform_coordinate(transform, annotations[idx, 1:5])

        return image, annotations

    def random_transform_group(self, images_group, annotations_group):
        assert len(images_group) == len(annotations_group)

        for idx in range(len(images_group)):
            images_group[idx], annotations_group[idx] = self.random_transform_group_item(images_group[idx],
                                                                                         annotations_group[idx])

        return images_group, annotations_group

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_item(self, image, annotations):

        image = self.preprocess_image(image)

        image, image_scale = self.resize_image(image)

        annotations[:, 1:5] *= image_scale

        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, images_group, annotations_group):
        assert len(images_group) == len(annotations_group)

        for idx in range(len(images_group)):
            images_group[idx], annotations_group[idx] = self.preprocess_group_item(images_group[idx],
                                                                                   annotations_group[idx])
        return images_group, annotations_group

    def compute_inputs(self, image_group):

        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        image_batch = np.zeros((self.batch_size, ) + max_shape, dtype=keras.backend.floatx())

        for image_idx, image in enumerate(image_group):
            image_batch[image_idx, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == "channels_first":
            image_batch = image_batch.transpose((0, 3, 1, 2))
        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = parse_anchor_parameter()
        return anchors_for_shape(image_shape, anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, images_group, annotations_group):
        max_shape = tuple(max(image.shape[x] for image in images_group) for x in range(3))
        anchors = self.generate_anchors(max_shape)
        labels_group = self.multi_label_handler(annotations_group, self.label_hierarchy)

        # for labels in labels_group:
        #     for label in labels:
        #         print(" ".join([self.class2name[self.idx2class[int(x)]] for x in list(label)]))
        # anchors是width在前
        batches = self.handle_targets(anchors, images_group, annotations_group,
                                                                labels_group, self.num_classes())
        return list(batches)

    def load_image_and_annotation(self, group):

        images_group, annotations_group = self.load_data(group)

        self.calculate_annotation_group(annotations_group, images_group)

        images_group, annotations_group = self.random_visual_effect_group(images_group, annotations_group)

        images_group, annotations_group = self.random_transform_group(images_group, annotations_group)

        images_group, annotations_group = self.preprocess_group(images_group, annotations_group)

        # from utils.visualization import plt_bboxes
        # plt_bboxes(images_group[0][:, :, ::-1], annotations_group[0][:, 0], np.ones((annotations_group[0].shape[0])),
        #            annotations_group[0][:, 1:5])

        inputs = self.compute_inputs(images_group)

        targets = self.compute_targets(images_group, annotations_group)

        return inputs, targets

class TrainLoader(ImageLoader):

    def __init__(self, shuffle_groups=False, **kwargs):
        self._idx_dict = {}
        self.o2n = None
        self.shuffle_groups = shuffle_groups
        super(TrainLoader, self).__init__(**kwargs)

    def fix_label_error(self, annotations):
        for i in range(annotations.shape[0]):
            annotations[i, 0] = self.o2n[int(annotations[i, 0])]

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def load_data(self, idx):
        file_infos = map(self.search_file, idx)
        images_group = []
        annotations_group = []
        for file, key in file_infos:
            images_group.append(self.load_image_item(file))
            id = file.split("/")[-1]
            id = id.split(".")[0]
            val = "%strain_%s" % (cfg.FILE.BASE_DIR+cfg.FILE.TRAIN_BBOX_DIR, key)
            res = self.load_annotations_item(val, id)
            self.fix_label_error(res)
            annotations_group.append(res[0:5])
        return images_group, annotations_group

    def search_file(self, item):
        idxs = list(self._idx_dict.keys())
        idxs.sort(reverse=True)
        select_key = None
        relative_idx = 0
        for i in idxs:
            if item >= i:
                select_key = self._idx_dict[i]
                relative_idx = item - i
                break
        select_dir = "{0}/train_{1}".format(cfg.FILE.IMAGES_TRAIN_DIR, select_key)
        select_dir = cfg.FILE.BASE_DIR + select_dir
        select_file = None
        with os.scandir(select_dir) as scanner:
            for idx, entry in enumerate(scanner):
                if idx == relative_idx:
                    select_file = entry.name
                    break

        if not select_file:
            raise RuntimeError("search image failed, can not find %s group files in file %s." % (item, self._idx_dict[i]))
        select_file = "{0}/{1}".format(select_dir, select_file)
        # print("{}-{}".format(select_key, select_file))
        return select_file, select_key

    def load_classes(self):
        class2idx, idx2class,class2name, hi = get_multi_labels_hierarchy_and_classes()
        from utils.datasets import old2new
        self.o2n = old2new(class2idx)

        return class2idx, idx2class, class2name, hi

    def get_length(self):
        # print(cfg.FILE.BASE_DIR)
        file = open(cfg.FILE.BASE_DIR + cfg.FILE.TRAIN_INFO)
        lines = file.readlines()
        # lines.pop(-1)
        acc = 0
        for line in lines:
            idx, length = line.split(" ")
            length.replace("\\n", "")
            length = int(length)
            self._idx_dict[acc] = idx
            acc += length
        file.close()
        # print(self._idx_dict)
        return acc

class ValLoader(ImageLoader):

    def __init__(self, **kwargs):
        dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_VALIDATION_DIR
        self.length = len(os.listdir(dir))

        super(ValLoader, self).__init__(**kwargs)

        print(self.label_hierarchy)

    def load_classes(self):
        class2idx, idx2class, class2name, hi = get_multi_labels_hierarchy_and_classes()
        return class2idx, idx2class, class2name, hi

    def get_length(self):
        val_dir = cfg.FILE.BASE_DIR + cfg.FILE.VAL_BBOX_DIR
        count = 0
        with os.scandir(val_dir) as scanner:
            for _ in scanner:
                count += 1
        # print("length: %s"%count)
        return count

    def search_file(self, item):
        dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_VALIDATION_DIR
        select_file = None
        with os.scandir(dir) as scanner:
            for idx, entry in enumerate(scanner):
                if idx == item:
                    select_file = entry.name

        if not select_file:
            raise RuntimeError("search image failed, can not find validation image %s" % item)
        return select_file

    def load_data(self, idx):
        file_infos = map(self.search_file, idx)
        images_group = []
        annotations_group = []
        image_dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_VALIDATION_DIR
        annotation_dir = cfg.FILE.BASE_DIR + cfg.FILE.VAL_BBOX_DIR
        annotation_dir = annotation_dir[:-1]
        for file in file_infos:
            images_group.append(self.load_image_item(image_dir+file))
            id = file.replace(".jpg", "")
            print("id: ", id)
            res = self.load_annotations_item(annotation_dir, id)
            annotations_group.append(res[:, 0:5])
        return images_group, annotations_group

    def __getitem__(self, item):
        images_group, annotations_group = self.load_data([item])
        # print(annotations_group[0])
        self.calculate_annotation_group(annotations_group, images_group)

        # for idx in range(len(annotations_group)):
        #     annotations_group[idx] = bbox_for_anchors([0, 0], annotations_group[idx])
        #
        # image = images_group[0]
        # annotations = annotations_group[0]
        image = images_group[0]
        labels = annotations_group[0][:, 0]
        boxes = bbox_for_anchors([0, 0], annotations_group[0][:, 1:5])

        return image, labels, boxes

    def size(self):
        return self.length

    def num_classes(self):
        return len(self.class2idx)

    def has_label(self, label):
        return label in self.idx2class

    def has_class(self, cls):
        return cls in self.class2idx

    def class_to_label(self, name):
        return self.class2idx[name]

    def label_to_class(self, label):
        return self.idx2class[label]

    def label_to_name(self, label):
        cls = self.label_to_class(label)
        return self.class2name[cls]

    def load_image(self, image_index):
        file = self.search_file(image_index)
        img_dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_VALIDATION_DIR
        return self.load_image_item(img_dir+file)

    def load_annotations(self, image_index):
        file = self.search_file(image_index)
        file = file.replace(".jpg", "")
        ann_dir = cfg.FILE.BASE_DIR + cfg.FILE.VAL_BBOX_DIR
        ann_dir = ann_dir[:-1]
        return self.load_annotations_item(ann_dir, file)[:, 0:5]

    def judge_similar(self, a, b):
        """
        返回顺序：父 子
        :param a:
        :param b:
        :return:
        """
        a_h = self.label_hierarchy[a]
        b_h = self.label_hierarchy[b]
        if b in a_h:
            return b, a
        if a in b_h:
            return a, b
        return -1, -1

if __name__ == "__main__":
    from utils.image import random_visual_effect_generator
    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05))
    from utils.transform import random_transform_generator
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )
    x = TrainLoader(transform_generator=transform_generator,
                    visual_effect_generator=visual_effect_generator)
    item = 0
    # group = x.groups[item]
    # x.load_image_and_annotation(group)
    xx = x[item]
    #
    # y = ValLoader(shuffle_groups=False)
    # yy = y[6]
    # from utils.image import read_image_bgr, apply_transform
    # image = read_image_bgr("/media/user/disk2/delusion/openImage2019/data/oid/train/train_0/0a0a5cb609c10f09.jpg")
    # image.astype(np.float32)
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    # matrix = np.array([
    #     [-1, 0, image.shape[1]],
    #     [0, -1, image.shape[0]],
    #     [0, 0, 1],
    # ], dtype=np.float32)
    # output = apply_transform(matrix[:2, :], image)
    # plt.imshow(output)
    # plt.show()


