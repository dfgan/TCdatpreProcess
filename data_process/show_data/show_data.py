# -*- coding: utf-8 -*-
# @time     :21-1-10 下午1:45

import os
import numpy as np
import random
import json
import matplotlib.pyplot as plt

train_img_dir = '/swdata/df/cz_data/data/tile_round1_train_20201231/train_imgs'
train_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/train_annos.json'

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
            mid['area'] = 0
            mid['iscrowd'] = 0
            mid['image_id'] = image_dict[name]
            bbox_old = bb['bbox']
            bbox = []
            for i in bbox_old:
                bbox.append(int(i))
            mid['bbox'] = bbox
            mid['ignore'] = 0
            index += 1
            mid['id'] = index
            mid['category_id'] = label_dict[bb['category']]
            annotations.append(mid)
    return annotations

with open(train_json, 'r') as r:
    data_info = json.load(r)

data_dict = {}
names = {}
labels = []
label_bbox = {}
for info in data_info:
    name = info['name']
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

    if category in label_bbox:
        label_bbox[category].append(bbox)
    else:
        label_bbox[category] = [bbox, ]


# test_dict = {}
# train_dict = {}
# for name in names:
#     k = random.random()
#     if k > 0.9:
#         test_dict[name] = names[name]
#     else:
#         train_dict[name] = names[name]


# #train
# images, images_dict = make_images(train_dict, 2020000000)
# categories, labels_dict = make_categories(labels)
# annotations = make_annotations(data_dict, labels_dict, images_dict)
# train_out = {}
# train_out['images'] = images
# train_out['annotations'] = annotations
# train_out['categories'] = categories
# print(len(annotations))
# train_out = json.dumps(train_out, ensure_ascii=False, indent=4)
# with open('train.json', 'w+', encoding='utf-8') as f:
#     f.write(train_out)
#
# #test
# images, images_dict = make_images(test_dict, 2021000000)
# annotations = make_annotations(data_dict, labels_dict, images_dict)
# test_out = {}
# test_out['images'] = images
# test_out['annotations'] = annotations
# test_out['categories'] = categories
# print(len(annotations))
# test_out = json.dumps(test_out, ensure_ascii=False, indent=4)
# with open('test.json', 'w+', encoding='utf-8') as f:
#     f.write(test_out)

def bbox2xywh(data):
    out = {}
    for key in data:
        bboxes = data[key]
        new_bbox = []
        for bbox in bboxes:
            x = (bbox[0]+bbox[2]) / 2
            y = (bbox[1]+bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            new_bbox.append([x, y, w, h])
        out[key] = new_bbox
    return out

l_w = [2, 2.5, 3, 3.5, 4, 4.5, 5]
l_h = [4, 4.5, 5, 5.5, 6, 6.5, 7]
out = bbox2xywh(label_bbox)
name = []
i = 0
rates = []
fig, ax = plt.subplots()
for label in out:
    name.append(label)
    locat = np.asarray(out[label])
    x = locat[:, 2]
    y = locat[:, 3]
    rate = x / y
    rates.append(rate)
    ax.annotate(str(label), (l_w[i], l_h[i]))
    i += 1
    ax.scatter(x, y)
plt.show()

for rate in rates:
    data = sorted(rate)
    plt.scatter(np.arange(len(data)), data)
plt.show()