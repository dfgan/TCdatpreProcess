# -*- coding: utf-8 -*-
# @time     :21-1-13 下午3:55
from argparse import ArgumentParser
import cv2
import numpy as np
import os
import json
from mmdet.apis import inference_detector, init_detector, show_result_pyplot



json_file = '/swdata/df/cz_data/data/tile_round1_train_20201231/test.json'
path = '/swdata/df/cz_data/data/tile_round1_testA_20201231/testA_imgs'

config1 = '/swdata/df/cz_data/mmdetection/df_use/cam1/cam1.py'
checkpoint1 = '/swdata/df/cz_data/mmdetection/df_use/cam1/epoch_12.pth'

config2 = '/swdata/df/cz_data/mmdetection/df_use/cam2/cam2.py'
checkpoint2 = '/swdata/df/cz_data/mmdetection/df_use/cam2/epoch_12.pth'

config3 = '/swdata/df/cz_data/mmdetection/df_use/cam3/cam3.py'
checkpoint3 = '/swdata/df/cz_data/mmdetection/df_use/cam3/epoch_12.pth'

with open(json_file, 'r') as d:
    data = json.load(d)

categories = data['categories']
labels = {}
for categorie in categories:
    id = categorie['id']
    name = categorie['name']
    labels[id] = int(name)

files = os.listdir(path)

min_score = 1.
out = []
done_files = []

model1 = init_detector(config1, checkpoint1, device='cuda:1')
for file in files:
    split_name = file.split('_')[-1].split('.')[0]
    if split_name != 'CAM1':
        continue
    done_files.append(file)
    print((len(done_files)*1.)/len(files))
    file_path = os.path.join(path, file)
    results = inference_detector(model1, file_path)
    for result_inx in range(len(results)):
        out_single = {}
        name = file
        # print(name)
        result = results[result_inx]
        if len(result) == 0:
            continue
        categorie = labels[result_inx+1]
        # label_id = labels[result_inx]
        for bboxes in result:
            bbox = bboxes[:4]
            score = bboxes[4]
            if score < min_score:
                min_score = score
                print(min_score)
            if score < 0.01:
                continue
            # bbox = bbox.astype(np.int)
            bbox_single = []
            for bb in bbox:
                bbox_single.append(int(bb))
            # print(name)
            out_single['name'] = name
            out_single['category'] = categorie
            out_single['bbox'] = bbox_single
            out_single['score'] = np.float(score)
            out.append(out_single)

model2 = init_detector(config2, checkpoint2, device='cuda:1')
for file in files:
    split_name = file.split('_')[-1].split('.')[0]
    if split_name != 'CAM2':
        continue
    done_files.append(file)
    print((len(done_files)*1.)/len(files))
    file_path = os.path.join(path, file)
    results = inference_detector(model2, file_path)
    for result_inx in range(len(results)):
        out_single = {}
        name = file
        # print(name)
        result = results[result_inx]
        if len(result) == 0:
            continue
        categorie = labels[result_inx+1]
        # label_id = labels[result_inx]
        for bboxes in result:
            bbox = bboxes[:4]
            score = bboxes[4]
            if score < min_score:
                min_score = score
                print(min_score)
            if score < 0.01:
                continue
            # bbox = bbox.astype(np.int)
            bbox_single = []
            for bb in bbox:
                bbox_single.append(int(bb))
            # print(name)
            out_single['name'] = name
            out_single['category'] = categorie
            out_single['bbox'] = bbox_single
            out_single['score'] = np.float(score)
            out.append(out_single)

model3 = init_detector(config3, checkpoint3, device='cuda:1')
for file in files:
    split_name = file.split('_')[-1].split('.')[0]
    if split_name != 'CAM3':
        continue
    done_files.append(file)
    print((len(done_files)*1.)/len(files))
    file_path = os.path.join(path, file)
    results = inference_detector(model3, file_path)
    for result_inx in range(len(results)):
        out_single = {}
        name = file
        # print(name)
        result = results[result_inx]
        if len(result) == 0:
            continue
        categorie = labels[result_inx+1]
        # label_id = labels[result_inx]
        for bboxes in result:
            bbox = bboxes[:4]
            score = bboxes[4]
            if score < min_score:
                min_score = score
                print(min_score)
            if score < 0.01:
                continue
            bbox_single = []
            for bb in bbox:
                bbox_single.append(int(bb))
            out_single['name'] = name
            out_single['category'] = categorie
            out_single['bbox'] = bbox_single
            out_single['score'] = np.float(score)
            out.append(out_single)

print(min_score)
result_out = json.dumps(out, ensure_ascii=False, indent=4)
with open('result.json', 'w+', encoding='utf-8') as f:
    f.write(result_out)

