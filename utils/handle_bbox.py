# -*- encoding: utf-8 -*-
# @TIME    : 2019/8/7 21:31
# @Author  : 成昭炜
# @File    : handle_bbox.py
import pandas as pd
import os
from multiprocessing import Semaphore, JoinableQueue, Process
import numpy as np
import time

from config.fileconfig import cfg
from utils.datasets import get_multi_labels_hierarchy_and_classes

# COLUMNS = [cfg.BBOX.CSV_XMIN, cfg.BBOX.CSV_XMAX, cfg.BBOX.CSV_YMIN,
#            cfg.BBOX.CSV_YMAX, cfg.BBOX.CSV_ISOCCLUDED, cfg.BBOX.CSV_ISTRUNCATES,
#            cfg.BBOX.CSV_ISGROUPOF, cfg.BBOX.CSV_ISDEPICTION, cfg.BBOX.CSV_ISINSIDE]

COLUMNS = [cfg.BBOX.CSV_XMIN, cfg.BBOX.CSV_XMAX, cfg.BBOX.CSV_YMIN,
           cfg.BBOX.CSV_YMAX, cfg.BBOX.CSV_ISGROUPOF]

class2idx, _, _, _ = get_multi_labels_hierarchy_and_classes()

def handle_bbox_with_multiprocess(csv_dir, images_dir, save_dir):
    # os.mkdir(cfg.FILE.BASE_DIR + cfg.FILE.BBOX_DIR + img_dir)

    csv_path = os.listdir(cfg.FILE.BASE_DIR + csv_dir)
    csv_length = len(csv_path)
    print(csv_length)
    # pool = Pool(processes=csv_length)

    # print("主进程：创建Queue")

    queue1 = JoinableQueue(csv_length)
    queue2 = JoinableQueue(csv_length)
    handles = [Semaphore(1) for _ in range(csv_length)]
    # print("csv_length: ", csv_length)
    def handle_bbox(csv_file, handle):
        # print(cfg.FILE.BASE_DIR + csv_dir + csv_file)
        data = pd.read_csv(cfg.FILE.BASE_DIR + cfg.FILE.TRAIN_BBOX_DIR + csv_file,
                           converters={
                               cfg.BBOX.CSV_IMAGEID: str,
                               cfg.BBOX.CSV_LABELNAME: str,
                               cfg.BBOX.CSV_XMIN: float,
                               cfg.BBOX.CSV_XMAX: float,
                               cfg.BBOX.CSV_YMIN: float,
                               cfg.BBOX.CSV_YMAX: float})
        while True:
            handle.acquire()
            image_id = queue1.get()
            temp = data[data[cfg.BBOX.CSV_IMAGEID] == image_id]
            length = temp.shape[0]
            results = np.zeros((length, len(COLUMNS) + 1))
            if length > 0:
                classes = [class2idx[label] for label in temp[cfg.BBOX.CSV_LABELNAME].values]
                results[:, 0] += classes
                values = temp[COLUMNS].values
                for idx in range(length):
                    results[idx, 1:] += values[idx]
            queue2.put(results)
            queue1.task_done()
    processes = []
    for idx, csv_file in enumerate(csv_path):
        p = Process(target=handle_bbox, args=(csv_file, handles[idx]))
        p.daemon = True
        p.start()
        processes.append(p)

    print("数据加载完成")

    print("主进程开始执行")

    image_scanner = os.scandir(images_dir)
    with image_scanner as scanner:
        for idx, entry in enumerate(scanner):
            # start = time.time()
            image_id = entry.name.split(".")[0]
            print(image_id)
            for _ in range(csv_length):
                queue1.put(image_id)
            res = queue2.get()
            queue2.task_done()
            for __ in range(csv_length-1):
                temp_ = queue2.get()
                queue2.task_done()
                res = np.concatenate((res, temp_), axis=0)
            temp_file = cfg.FILE.BASE_DIR + save_dir + "/" + image_id + ".txt"
            np.savetxt(temp_file, res)
            # end = time.time()
            # print("数据处理耗时： %s" %(end - start))

            for handle in handles:
                handle.release()
    print("主进程结束")

def handle(dir_list):
    for images_dir in dir_list:
        name = images_dir.split("/")[-1]
        handle_bbox_with_multiprocess(cfg.FILE.TRAIN_BBOX_DIR, images_dir)

def handle_val_bbox():
    handle_bbox_with_multiprocess(cfg.FILE.TRAIN_BBOX_DIR,
                                  "/media/user/disk2/delusion/oid/validation",
                                  "data/labels/validation")



if __name__ == "__main__":
    handle_val_bbox()

