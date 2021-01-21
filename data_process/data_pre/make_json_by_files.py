# -*- coding: utf-8 -*-
# @time     :21-1-14 下午2:45

import os
import numpy as np
import random
import json



def make_images(names, start_id):
    out = []
    label_dict = {}
    for name in names:
        start_id += 1
        mid = {}
        size = names[name]
        mid['file_name'] = name
        mid['width'] = size[0]
        mid['height'] = size[1]
        mid['id'] = start_id
        out.append(mid)
        label_dict[name] = start_id
    return out, label_dict

def make_categories(labels):
    categorys = []
    label_dict = {}
    ind = 1
    for label in labels:
        mid = {}
        mid['supercategory'] = 'none'
        mid['id'] = ind
        mid['name'] = str(label)
        label_dict[label] = ind
        ind += 1
        categorys.append(mid)

    return categorys, label_dict

def make_annotations(info, label_dict, image_dict):
    annotations = []
    index = 0
    for name in image_dict:
        data = info[name]
        for bb in data['annotations']:
            mid = {}
            mid['segmentation'] = []
            mid['iscrowd'] = 0
            mid['image_id'] = image_dict[name]
            bbox_old = bb['bbox']
            x = int(bbox_old[0])
            y = int(bbox_old[1])
            w = int(bbox_old[2] - bbox_old[0])
            h = int(bbox_old[3] - bbox_old[1])
            mid['bbox'] = [x, y, w, h]
            mid['area'] = w * h
            mid['ignore'] = 0
            index += 1
            mid['id'] = index
            mid['category_id'] = label_dict[bb['category']]
            annotations.append(mid)
    return annotations

train_img_dir = '/swdata/df/cz_data/data/tile_round1_train_20201231/color/train_split'
train_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/color/train_split.json'
with open(train_json, 'r') as r:
    data_info = json.load(r)

all_files = os.listdir(train_img_dir)
data_dict = {}
names = {}
labels = []
for info in data_info:
    name = info['name']
    if name not in all_files:
        continue
    height = info['image_height']
    width = info['image_width']
    category = info['category']
    bbox = info['bbox']
    if name not in names:
        names[name] = [width, height]
    if category not in labels:
        labels.append(category)
    if name in data_dict:
        data_dict[name]['annotations'].append({'category':category, 'bbox':bbox})
    else:
        data_dict[name] = {'height':height, 'width':width, 'annotations':[{'category':category, 'bbox':bbox},]}

test_dict = {}
train_dict = {}
for name in names:
    if name in all_files:
        train_dict[name] = names[name]

#train
images, images_dict = make_images(train_dict, 2021000000)
categories, labels_dict = make_categories(labels)
annotations = make_annotations(data_dict, labels_dict, images_dict)
train_out = {}
train_out['images'] = images
train_out['annotations'] = annotations
train_out['categories'] = categories
print(len(annotations))
train_out = json.dumps(train_out, ensure_ascii=False, indent=4)
with open('/swdata/df/cz_data/data/tile_round1_train_20201231/color/train.json', 'w+', encoding='utf-8') as f:
    f.write(train_out)
