# -*- coding: utf-8 -*-
# @time     :21-1-6 上午11:08
import os
import json
import shutil

all_images = '/swdata/df/cz_data/data/tile_round1_train_20201231/imgs'
save_dirt = '/swdata/df/cz_data/data/tile_round1_train_20201231/test_images'
json_file = 'test.json'

with open(json_file, 'r') as r:
    info = json.load(r)

images = info['images']
for info in images:
    name = info['file_name']
    shutil.copyfile(os.path.join(all_images, name), os.path.join(save_dirt, name))
    print(name)