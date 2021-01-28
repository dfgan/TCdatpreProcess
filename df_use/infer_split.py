# -*- coding: utf-8 -*-
# @time     :21-1-20 上午9:25

from argparse import ArgumentParser
import cv2
import numpy as np
import os
import json
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def nms(boxes, iou_threshold=0.3 ,score_threshold=0.1):
    if boxes.size == 0:
        return np.empty((0, 3))
    valid = np.where(boxes[:, 4] > score_threshold)
    boxes = boxes[valid]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1) * (y2 - y1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= iou_threshold)]
    pick = pick[0:counter]

    boxes = boxes[pick, :]
    return boxes

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

            img_c = img[h_offset:(h_offset + crop_size[0]), w_offset:(w_offset + crop_size[1]), :]
            crop_imgs.append(img_c)
            offset_imgs.append([h_offset, w_offset])
    return crop_imgs, offset_imgs

def result_offet(result, offset):
    out = {}
    for i in range(len(result)):
        if len(result[i]) != 0:
            mid = result[i] + np.asarray([offset[1], offset[0], offset[1], offset[0], 0.])
            out[i + 1] = mid
    return out

def merge_infer_out(outs):
    result = {}
    for out in outs:
        for label in out:
            if label not in result:
                result[label] = out[label]
            else:
                np.concatenate((result[label], out[label]), axis=0)
    results = {}
    for label in result:
        bboxes = result[label]
        bbox = nms(bboxes)
        results[label] = bbox
    return results



json_gray = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/test.json'
json_color = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/test.json'
path = '/swdata/df/cz_data/data/tile_round1_testA_20201231/testA_imgs'
config_gray = '/swdata/df/cz_data/mmdetection/df_use/gray_crop_base_28/gray.py'
checkpoint_gray = '/swdata/df/cz_data/mmdetection/df_use/gray_crop_base_28/epoch_24.pth'
config_color = '/swdata/df/cz_data/mmdetection/df_use/color_crop_28/color.py'
checkpoint_color = '/swdata/df/cz_data/mmdetection/df_use/color_crop_28/epoch_24.pth'

crop_size = [1500, 2048]
crop_strid = [750, 1024]

with open(json_gray, 'r') as d:
    data_gray = json.load(d)

categories_gray = data_gray['categories']
labels_gray = {}
for categorie in categories_gray:
    id = categorie['id']
    name = categorie['name']
    labels_gray[id] = int(name)

with open(json_color, 'r') as d:
    data_color = json.load(d)

categories_color = data_color['categories']
labels_color = {}
for categorie in categories_color:
    id = categorie['id']
    name = categorie['name']
    labels_color[id] = int(name)

min_score = 1.
out_put = []
files = os.listdir(path)

done_files = []
model_gray = init_detector(config_gray, checkpoint_gray)
model_color = init_detector(config_color, checkpoint_color)

for file in files[50:53]:
    done_files.append(file)
    file_path = os.path.join(path, file)

    if 'CAM3' in file:
        model = model_color
        labels_dic = labels_color
    else:
        model = model_gray
        labels_dic = labels_gray

    crop_imgs, offset_imgs = crop_fun(cv2.imread(file_path))

    outs = []
    for img_id in range(len(crop_imgs)):
        result = inference_detector(model, crop_imgs[img_id])
        result = result_offet(result, offset_imgs[img_id])
        outs.append(result)

    results = merge_infer_out(outs)

    for label in results:
        all_bbox = results[label]
        categorie_label = labels_dic[label]
        for bboxes in all_bbox:
            out_single = {}
            bbox = bboxes[:4]
            score = bboxes[4]
            bbox_single = []
            for bb in bbox:
                bbox_single.append(int(bb))

            out_single['name'] = file
            out_single['category'] = categorie_label
            out_single['bbox'] = bbox_single
            out_single['score'] = np.float(score)
            out_put.append(out_single)

    print((len(done_files) * 1.) / len(files))

result_out = json.dumps(out_put, ensure_ascii=False, indent=4)
with open('result.json', 'w+', encoding='utf-8') as f:
    f.write(result_out)

