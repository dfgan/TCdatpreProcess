# -*- coding: utf-8 -*-
# @time     :21-1-12 下午4:14

import os
import cv2
import shutil
import json

all_images = '/swdata/df/cz_data/data/tile_round1_train_20201231/test_images'
label = './test.json'

with open(label, 'r') as R:
    data = json.load(R)
images_info = data['images']
annotaions = data['annotations']
labels = data['categories']
# print(data)


infos = {}
id_name = {}
for image in images_info:
    name = image['file_name']
    id = image['id']
    if name not in infos:
        infos[name] = {'id':id, 'bbox':[], 'label':[]}
        id_name[id] = name
    else:
        print(name)

for annot in annotaions:
    id = annot['image_id']
    bbox = annot['bbox']

    label = annot['category_id']
    infos[id_name[id]]['bbox'].append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    infos[id_name[id]]['label'].append(label)

for name in infos:
    img_path = os.path.join(all_images, name)
    img = cv2.imread(img_path)
    bboxes = infos[name]['bbox']
    labels = infos[name]['label']
    for bbox_index in range(len(bboxes)):
        cv2.rectangle(img, (bboxes[bbox_index][0], bboxes[bbox_index][1]), (bboxes[bbox_index][2], bboxes[bbox_index][3]), (255, 0, 255), thickness=1)
        tet = labels[bbox_index]
        cv2.putText(img, str(tet), (bboxes[bbox_index][0], bboxes[bbox_index][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    save_path = os.path.join('./save', name)
    cv2.imwrite(save_path, img)
    print(name)
