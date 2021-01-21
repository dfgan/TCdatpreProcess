# -*- coding: utf-8 -*-
# @time     :21-1-18 下午7:23

import cv2
import shutil
import os
import json
import numpy as np

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

def bbox_in_crop(bbox, cood):
    if cood[0] < bbox[0] < cood[2]:
        x1_insid = True
    else:
        x1_insid = False

    if cood[0] < bbox[2] < cood[2]:
        x2_insid = True
    else:
        x2_insid = False

    if cood[1] < bbox[1] < cood[3]:
        y1_insid = True
    else:
        y1_insid = False

    if cood[1] < bbox[3] < cood[3]:
        y2_insid = True
    else:
        y2_insid = False

    if x1_insid and x2_insid and y1_insid and y2_insid:
        return 1    #inside
    elif not x1_insid and not x2_insid and not y1_insid and not y2_insid:
        return 0    #outside
    else:
        return 2    #cut

def bbox_verify_img(offset, bboxes):
    in_img_bbox = []
    # bbox_cut_flag = False
    img_h, img_w = crop_size
    x1, y1 = offset[1], offset[0]
    x2, y2 = offset[1] + img_w, offset[0] + img_h
    for bbox in bboxes:
        bbox_flag = bbox_in_crop(bbox, [x1, y1, x2, y2])
        if bbox_flag == 1:
            in_img_bbox.append(bbox)
        elif bbox_flag == 0:
            continue
        else:
            return False, in_img_bbox
    if len(in_img_bbox) > 0:
        return True, in_img_bbox
    else:
        return False, in_img_bbox

def bbox_change_offset(offsets, bboxes):
    # bbox = np.asarray(bboxes)
    # out_bbox = bbox[:, :4] - [offsets[1], offsets[0], offsets[1], offsets[0]]
    # # out_bbox = out_bbox[:, 2] - offsets[1]
    # # out_bbox = out_bbox[:, 1] - offsets[0]
    # # out_bbox = out_bbox[:, 3] - offsets[0]
    out_box = []
    for bbox in bboxes:
        x1, y1, x2, y2, label = bbox
        y1 -= offsets[0]
        x1 -= offsets[1]
        y2 -= offsets[0]
        x2 -= offsets[1]
        out_box.append([x1, y1, x2, y2, label])
    return out_box

def main():
    splited_num = 1
    name_info = get_json_info(train_json)
    files = os.listdir(img_dir)
    out_json = []
    for file in files:
        print(splited_num / len(files))
        splited_num += 1
        # print(file)
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)
        img_info = name_info[file]
        bb = img_info['bb']
        crop_imgs, offset_imgs = crop_fun(img)
        for i in range(len(crop_imgs)):
            flag_save, bboxes_inside = bbox_verify_img(offset_imgs[i], bb)
            if not flag_save or crop_imgs[i].shape[0]==0 :
                continue

            new_bbox = bbox_change_offset(offset_imgs[i], bboxes_inside)
            name_split = list(map(str, offset_imgs[i]))
            add_name = '_'.join(name_split)
            new_name = file.split('.')[0] + '_'+ add_name +'.jpg'
            for box in new_bbox:
                json_dict = {}
                json_dict['name'] = new_name
                json_dict['image_height'] = img_info['h']
                json_dict['image_width'] = img_info['w']
                json_dict['category'] = int(box[-1])
                json_dict['bbox'] = box[:4]
                out_json.append(json_dict)

            save_img_path = os.path.join(save_path, new_name)
            cv2.imwrite(save_img_path, crop_imgs[i])
    test_out = json.dumps(out_json, ensure_ascii=False, indent=4)
    with open(save_json, 'w+', encoding='utf-8') as f:
        f.write(test_out)

if __name__ == '__main__':
    img_dir = '/swdata/df/cz_data/data/tile_round1_train_20201231/color/test'
    save_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/color/test_split'
    train_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/all_annos.json'
    save_json = '/swdata/df/cz_data/data/tile_round1_train_20201231/color/test_split.json'
    # 8192, 6000,   4096, 3500
    crop_size = [1500, 2048]
    crop_strid = [750, 1024]
    main()