# -*- coding: utf-8 -*-
# @time     :21-1-20 上午9:25

from argparse import ArgumentParser
import cv2
import numpy as np
import os
import json
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def crop_fun(img):
    h, w, _ = img.shape
    h_time = (h - crop_size[0]) // crop_strid[0]
    h_leave = (h - crop_size[0]) % crop_strid[0]
    w_time = (w - crop_size[1]) // crop_strid[1]
    w_leave = (w - crop_size[1]) % crop_strid[1]

    if h_leave > 300:
        h_iter = h_time + 1
    else:
        h_iter = h_time

    if w_leave > 500:
        w_iter = w_time + 1
    else:
        w_iter = w_time

    crop_imgs = []
    offset_imgs = []
    for i in range(h_iter):
        for j in range(w_iter):
            if i == h_iter - 1:
                h_offset = h - crop_size[0]
            else:
                h_offset = i * crop_strid[0]

            if j == w_iter - 1:
                w_offset = w - crop_size[1]
            else:
                w_offset = j * crop_strid[1]

            img_c = img[h_offset:(h_offset+crop_size[0]), w_offset:(w_offset+crop_size[1]), :]
            crop_imgs.append(img_c)
            offset_imgs.append([h_offset, w_offset])
    return crop_imgs, offset_imgs

def result_offet(result, offset):
    out = []
    for i in range(len(result)):
        if len(result[i]) == 0:
           out.append(result[i])
        else:
            mid = result[i] + np.asarray([offset[1], offset[0], offset[1], offset[0], 0.])
            out.append(mid)
    return out


json_file = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/test.json'
path = '/swdata/df/cz_data/data/tile_round1_testA_20201231/testA_imgs'
config = '/swdata/df/cz_data/mmdetection/df_use/gray.py'
checkpoint = '/swdata/df/cz_data/mmdetection/df_use/gray_crop_base4/epoch_12.pth'

crop_size = [1500, 2048]
crop_strid = [750, 1024]

with open(json_file, 'r') as d:
    data = json.load(d)

categories = data['categories']
labels = {}
for categorie in categories:
    id = categorie['id']
    name = categorie['name']
    labels[id] = int(name)
min_score = 1.
out = []
files = os.listdir(path)

done_files = []
model = init_detector(config, checkpoint, device='cuda:0')
for file in files:
    done_files.append(file)
    print((len(done_files)*1.)/len(files))
    file_path = os.path.join(path, file)
    end_flag = file.split('_')[-1].split('.')[0]
    if end_flag == 'CAM3':
        continue
    crop_imgs, offset_imgs = crop_fun(cv2.imread(file_path))
    outs = []
    for img_id in range(len(crop_imgs)):
        result = inference_detector(model, crop_imgs[img_id])
        result = result_offet(result, offset_imgs[img_id])
        outs.append(result)

    results = [[], ]*6
    for out in outs:
        for label_i in range(6):
            results[label_i].extend(out[label_i])

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

print(min_score)
result_out = json.dumps(out, ensure_ascii=False, indent=4)
with open('result1.json', 'w+', encoding='utf-8') as f:
    f.write(result_out)
