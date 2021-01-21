# -*- coding: utf-8 -*-
# @time     :21-1-19 下午1:27
import json

path = '/swdata/df/cz_data/data/tile_round1_train_20201231/gray/train_split.json'
with open(path, 'r') as r:
    data_info = json.load(r)

for info in data_info:
    info['image_height'] = 1500
    info['image_width'] = 2048

test_out = json.dumps(data_info, ensure_ascii=False, indent=4)
with open('train_split.json', 'w+', encoding='utf-8') as f:
    f.write(test_out)