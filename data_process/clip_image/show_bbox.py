# -*- coding: utf-8 -*-
# @time     :21-1-18 下午10:28

import cv2
import shutil
import os
import json
import numpy as np

def show_bbox(img, bboxes, labels=None):
    for i in range(len(bboxes)):
        x1 = bboxes[i][0]
        y1 = bboxes[i][1]
        x2 = bboxes[i][2]
        y2 = bboxes[i][3]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        # if labels!=None:
        cv2.putText(img, str(bboxes[i][4]),(int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
    img = cv2.resize(img, (512, 375))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return

def get_json_info(path):
    with open(path, 'r') as r:
        data_info = json.load(r)
    names_info = {}
    for info in data_info:
        name = info['name']
        w = info['image_width']
        h = info['image_height']
        label = info['category']
        bbox = info['bbox']
        bbox.append(float(label))
        if name not in names_info:
            names_info[name] = {'w': w, 'h':h, 'bb':[bbox,]}
        else:
            names_info[name]['bb'].append(bbox)
    return names_info
json_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/test_split.json'
data = get_json_info(json_path)

img_dirt = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/test_split'
files = os.listdir(img_dirt)
for file in files:
    img_path = os.path.join(img_dirt, file)
    img = cv2.imread(img_path)
    info = data[file]
    bb = info['bb']

    show_bbox(img, bb)