import matplotlib.pyplot as plt
import random

def plt_bboxes(img, classes, scores, bboxes, figsize=(10, 10), linewidth=1.5):
    """
    :param img: (H x W x C)
    :param classes: numpy (N x 1)
    :param scores: numpy (N x 1)
    :param bboxes: numpy (N x 4)
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]

    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            xmin = int(bboxes[i, 0])
            ymin = int(bboxes[i, 1])
            xmax = int(bboxes[i, 2])
            ymax = int(bboxes[i, 3])
            # xmin = int(bboxes[i, 0])
            # xmax = int(bboxes[i, 1])
            # ymin = int(bboxes[i, 2])
            # ymax = int(bboxes[i, 3])

            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin-2,
                           "{:s} | {:.3f}".format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()

if __name__ == "__main__":
    import os
    import random
    from config.fileconfig import cfg
    import matplotlib.image as mimage
    import numpy as np
    from utils.transform import random_transform_generator
    from utils.image import apply_transform
    gen = random_transform_generator(
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
    # image_path = os.listdir(cfg.FILE.BASE_DIR + cfg.FILE.BBOX_DIR)
    image_path = os.listdir("/media/user/disk2/delusion/openImage2019/data/labels/train/train_0")
    for i in range(1):
        file_id = random.choice(image_path)
        labels_np = np.loadtxt(cfg.FILE.BASE_DIR + cfg.FILE.BBOX_DIR + "train_0/" + file_id, ndmin=2)
        img = mimage.imread(cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_TRAIN_DIR + "train_0/" + file_id.replace("txt", "jpg"))
        print(type(img))
        plt_bboxes(img, labels_np[:, 0], np.ones((labels_np.shape[0], )), labels_np[:, 1:5])
        res = apply_transform(next(gen), img)
        # plt_bboxes(res, labels_np[:, 0], np.ones((labels_np.shape[0], )), labels_np[:, 1:5])


