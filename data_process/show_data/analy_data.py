# -*- coding: utf-8 -*-
# @time     :21-1-16 下午6:56
import json
import numpy as np
import matplotlib.pyplot as plt

def json_data(file):
    with open(file, 'r') as d:
        data = json.load(d)
    annotations = data['annotations']
    bbox = {}
    for annotation in annotations:
        bb_xywh = annotation['bbox']
        label_id = annotation['category_id']
        if label_id not in bbox:
            bbox[label_id] = [bb_xywh, ]
        else:
            bbox[label_id].append(bb_xywh)

    return bbox

def bbox_plot(info):
    fig = plt.figure()
    for key in info:
        bb = np.asarray(info[key])
        w_h = bb[:, 2:]
        plt.scatter(w_h[:, 0], w_h[:, 1])
    plt.show()



cam1_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/CAM1/train'     #gray
cam2_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/CAM2/train'     #gray
cam3_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/CAM3/train'     #rgb

cam1_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/cam1_train.json'
cam2_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/cam2_train.json'
cam3_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/train.json'

bb_info = json_data(cam3_json)
bbox_plot(bb_info)
