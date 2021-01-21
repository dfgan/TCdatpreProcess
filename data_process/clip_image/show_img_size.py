# -*- coding: utf-8 -*-
# @time     :21-1-18 下午8:45

import os
import cv2
import json

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

train_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/train_split.json'
data_info = get_json_info(train_json)
w_list = []
h_list = []
for name in data_info:
    w, h = data_info[name]['w'], data_info[name]['h']
    w_list.append(w)
    h_list.append(h)
print(set(w_list))
print(set(h_list))