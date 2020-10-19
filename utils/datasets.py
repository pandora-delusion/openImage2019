# -*- encoding: utf8 -*-

import pandas as pd
import json
import numpy as np
import os
import logging
import logging.config

from config.fileconfig import cfg


def load_class(classes):
    """
    从cvs文件中加载种类信息，返回两个字典，一个是idx:class，一个是class:idx
    """
    class2idx = dict(((cls, idx) for idx, cls in enumerate(classes)))

    return class2idx

def handle_class():
    class2Name = read_description()
    classes = class2Name.keys()
    class2idx = load_class(classes)
    return class2idx, class2Name

def analyse_hierarchy_old(json_dir):
    json_file = open(cfg.FILE.BASE_DIR + json_dir)
    json_dict = json.load(json_file)
    labelName = cfg.BBOX.CSV_LABELNAME
    subcategory = "Subcategory"
    classes = [json_dict[labelName]]

    def _handle_list(json_list):
        for json_dict in json_list:
            classes.append(json_dict[labelName])

        for js_dict in json_list:
            sub = js_dict.get(subcategory, None)
            if sub:
                _handle_list(sub)

    _handle_list(json_dict[subcategory])
    return classes

def old_idx_class_map():
    classes = analyse_hierarchy_old(cfg.FILE.CLASS_HIERARCHY_FILE)
    class2idx = load_class(classes)
    return class2idx

def old2new(new):
    o2n = {}
    old = old_idx_class_map()
    for cls in new:
        o2n[old[cls]] = new[cls]
    return o2n

def analyse_hierarchy(json_dir, class2idx):
    json_file = open(cfg.FILE.BASE_DIR + json_dir)
    json_dict = json.load(json_file)
    labelName = cfg.BBOX.CSV_LABELNAME
    subcategory = "Subcategory"
    hierarchy = {}

    stack = []

    def _analyse_hierarchy(json_dict):
        cls = json_dict[labelName]
        if subcategory in json_dict:
            if cls in class2idx:
                stack.append(cls)
            for json_obj in json_dict[subcategory]:
                _analyse_hierarchy(json_obj)

            if cls in class2idx:
                stack.pop(-1)

        if cls in class2idx:
            temp = []
            temp.extend(stack)
            if cls in hierarchy:
                hierarchy[cls].extend(temp)
            else:
                hierarchy[cls] = temp
    _analyse_hierarchy(json_dict)
    return hierarchy

def multi_label_handler(annotations, hierarchy):
    size = annotations.shape[0]
    labels = []
    for i in range(size):
        temp = int(annotations[i, 0])
        label = [temp]
        label.extend(hierarchy[temp])
        label = np.array(label, dtype=int)
        labels.append(label)

    return labels

def handle_multi_labels(annotations_group, hierarchy):
    labels_group = []
    for annotations in annotations_group:
        labels = multi_label_handler(annotations, hierarchy)
        labels_group.append(labels)

    return labels_group

def handle_train_bboxes(data, image_dir, class2idx, bbox_dir):
    image_scanner = os.scandir(image_dir)
    checkout_name = image_dir.split("\\")[-1]
    with image_scanner as scanner:
        for idx, entry in enumerate(scanner):
            image_id = entry.name.split(".")[0]
            bboxes = data[data[cfg.BBOX.CSV_IMAGEID] == image_id]
            # 有些图片可能还没有,先跳过
            if bboxes.shape[0] <= 0:
                continue
            bboxes_info = np.zeros((bboxes.shape[0], 10))
            label_list = bboxes[cfg.BBOX.CSV_LABELNAME].to_list()
            label_id_list = [class2idx[label] for label in label_list]
            bboxes_info[:, 0] += label_id_list
            bboxes_info[:, 1:] = bboxes.to_numpy()[:, 4:]
            temp_file = bbox_dir + checkout_name + "\\" + image_id + ".txt"
            np.savetxt(temp_file, bboxes_info)
            # logging.info("bbox number: %s", bboxes.shape[0], {idx:idx, image_id:image_id})

COLUMNS = [cfg.BBOX.CSV_LABELNAME, cfg.BBOX.CSV_XMIN, cfg.BBOX.CSV_XMAX, cfg.BBOX.CSV_YMIN,
           cfg.BBOX.CSV_YMAX, cfg.BBOX.CSV_ISOCCLUDED, cfg.BBOX.CSV_ISTRUNCATES,
           cfg.BBOX.CSV_ISGROUPOF, cfg.BBOX.CSV_ISDEPICTION, cfg.BBOX.CSV_ISINSIDE]

from multiprocessing import Process
def handle_bboxes(img_dir, class2idx, bbox_dir):
    data_gen = pd.read_csv(cfg.FILE.BASE_DIR + cfg.FILE.TRAIN_BBOX_FILE,
                       iterator=True, chunksize=10000,
                       converters={cfg.BBOX.CSV_IMAGEID: str,
                                   cfg.BBOX.CSV_LABELNAME: str,
                                   cfg.BBOX.CSV_XMIN: float,
                                   cfg.BBOX.CSV_XMAX: float,
                                   cfg.BBOX.CSV_YMIN: float,
                                   cfg.BBOX.CSV_YMAX: float,
                                   cfg.BBOX.CSV_ISOCCLUDED: int,
                                   cfg.BBOX.CSV_ISTRUNCATES: int,
                                   cfg.BBOX.CSV_ISGROUPOF: int,
                                   cfg.BBOX.CSV_ISDEPICTION: int,
                                   cfg.BBOX.CSV_ISINSIDE: int,
                                   })
    image_scanner = os.scandir(img_dir)
    checkout_name = img_dir.split("\\")[-1]
    logging.basicConfig(level=logging.DEBUG,
                        filename=cfg.FILE.BASE_DIR + "checkout\\" + checkout_name + ".log",
                        filemode="a",
                        format="%(idx)d - %(image_id)s")

    with image_scanner as scanner:
        for idx, entry in enumerate(scanner):
            image_id = entry.name.split(".")[0]
            temp = pd.DataFrame(columns=COLUMNS)
            for aslice in data_gen:
                bboxes = aslice[aslice[cfg.BBOX.CSV_IMAGEID] == image_id]
                if bboxes.shape[0] <= 0:
                    continue
                temp = temp.append(bboxes[COLUMNS])
            bboxes_info = np.zeros((temp.shape[0], 10))
            # label_list = temp[cfg.BBOX.CSV_LABELNAME].to_list()
            label_id_list = [class2idx[label] for label in temp[cfg.BBOX.CSV_LABELNAME]]
            bboxes_info[:, 0] += label_id_list
            bboxes_info[:, 1:] = temp.values[:, 1:]
            temp_file = bbox_dir + checkout_name + "\\" + image_id + ".txt"
            np.savetxt(temp_file, bboxes_info)
            logging.info(idx=idx, image_id=image_id)

def split_csv(csv_file):
    import os
    data = pd.read_csv(cfg.FILE.BASE_DIR + csv_file)
    chksize = int(data.shape[0]/float(os.cpu_count()) + 1)
    data_gen = pd.read_csv(cfg.FILE.BASE_DIR + csv_file,
                           iterator=True, chunksize=chksize)
    for idx, iter in enumerate(data_gen):
        iter.to_csv(cfg.FILE.BASE_DIR + "datasets/bbox/challenge-2019-val-detection-bbox_%s.csv" % idx)

def read_description():
    data = pd.read_csv(cfg.FILE.BASE_DIR + "/datasets/challenge-2019-classes-description-500.csv")
    class2Name = {}
    class2Name["/m/061hd_"] = "Infant bed"
    for _, row in data.iterrows():
        class2Name[row["/m/061hd_"]] = row["Infant bed"]
    return class2Name

def get_multi_labels_hierarchy_and_classes():
    class2idx, class2Name = handle_class()
    hi = analyse_hierarchy(cfg.FILE.CLASS_HIERARCHY_FILE, class2idx)
    class2idx = dict(((key, idx) for idx, key in enumerate(hi.keys())))
    idx2class = dict(((idx, key) for idx, key in enumerate(hi.keys())))
    hidx = {}
    for key in hi:
        hidx[class2idx[key]] = [class2idx[x] for x in hi[key]]
    return class2idx, idx2class, class2Name, hidx

def delete_empty_annotations():

    def delete_pic():
        dir_path = cfg.FILE.BASE_DIR + cfg.FILE.VAL_BBOX_DIR
        empty_file = []
        with os.scandir(dir_path) as scanner:
            for entry in scanner:
                file_path = dir_path + entry.name
                length = os.path.getsize(file_path)
                if length <= 0:
                    empty_file.append(entry.name)
        for path in empty_file:
            os.remove(dir_path+"/"+path)
        print(empty_file)
        image_dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_VALIDATION_DIR
        print(image_dir)
        for path in empty_file:
            os.remove(image_dir+"/"+path.replace("txt", "jpg"))
        print("删除%s个文件" % len(empty_file))

    # for file in files:
    #     delete_pic(file)
    delete_pic()

def delete_image(idx):
    dirs = "train_%s"%idx

    image_dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_TRAIN_DIR + dirs
    with os.scandir(image_dir) as img_scan:
        for entry in img_scan:
            ann_dir = cfg.FILE.BASE_DIR + cfg.FILE.BBOX_DIR + dirs
            isExist = False
            image_name = entry.name.replace(".jpg", "")
            with os.scandir(ann_dir) as ann_scan:
                for it in ann_scan:
                    ann_name = it.name.replace(".txt", "")
                    if image_name == ann_name:
                        isExist = True
                        break
            if not isExist:
                del_path = image_dir + "/" + entry.name
                os.remove(del_path)
                print("删除图片：%s"%del_path)


def delete_bad_image(idx):
    from utils.image import read_image_bgr
    dirs = "train_%s" % idx

    image_dir = cfg.FILE.BASE_DIR + cfg.FILE.IMAGES_TRAIN_DIR + dirs
    ann_dir = cfg.FILE.BASE_DIR + cfg.FILE.BBOX_DIR + "train/" + dirs
    print(image_dir)
    number = 0

    with os.scandir(image_dir) as img_scan:
        for entry in img_scan:
            image_name = entry.name
            # print(image_name)

            try:
                read_image_bgr("{}/{}".format(image_dir, image_name))
            except OSError:
                number += 1
                os.remove("{}/{}".format(image_dir, image_name))
                os.remove("{}/{}".format(ann_dir, image_name.replace("jpg", "txt")))

    print("{}: {}".format(dirs, number))
    return number

if __name__ == "__main__":
    # idx2class, class2idx = load_class(handle_class(cfg.FILE.CLASS_HIERARCHY_FILE))
    # handle_train_bboxes(cfg.FILE.TRAIN_BBOX_FILE, cfg.FILE.IMAGES_TRAIN_DIR, class2idx, cfg.FILE.BBOX_DIR)
    # trains = ["I:\\openimage\\train\\train_0", "I:\\openimage\\train\\train_1"]
    # print(len(idx2class))
    # class2idx, class2Name = handle_class()
    # hi = analyse_hierarchy(cfg.FILE.CLASS_HIERARCHY_FILE, class2idx)
    # print(hi)

    # file = open(cfg.FILE.BASE_DIR + "/datasets/classes.txt", "w")
    # class2idx, idx2class, class2Name, hi = get_multi_labels_hierarchy_and_classes()
    # print(class2idx)
    # print(len(class2idx))
    # print(hi)

    # delete_empty_annotations()
    # delete_bad_image(0)
    # delete_image('1')
    # split_csv(cfg.FILE.VAL_BBOX_FILE)

    from multiprocessing import Pool

    idxes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']
    with Pool(len(idxes)) as pool:
        pool.map(delete_bad_image, idxes)
