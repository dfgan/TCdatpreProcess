# -*- coding: utf-8 -*-
# @time     :21-1-12 下午7:31
import os

all_images_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/imgs/'
images_files = os.listdir(all_images_path)
print(len(images_files))

sample_split = {}
out = []
for file in images_files:
    name_ll = file.split('_')
    name = name_ll[:2]
    if name in out:
        sample_split['_'.join(name)].append('_'.join(name_ll))
        continue
    out.append(name)
    sample_split['_'.join(name)] = ['_'.join(name_ll), ]
print(len(out))