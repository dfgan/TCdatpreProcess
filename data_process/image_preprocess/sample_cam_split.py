# -*- coding: utf-8 -*-
# @time     :21-1-14 下午2:10
import os
import random
import numpy as np
import shutil

all_data_path = '/swdata/df/cz_data/data/tile_round1_train_20201231/imgs'
save_path = '/swdata/df/cz_data/data/tile_round1_train_20201231'

files = os.listdir(all_data_path)
split_dic = {}
for file in files:
    split_name = file.split('_')[-1].split('.')[0]
    if split_name not in split_dic:
        split_dic[split_name] = [file, ]
    else:
        split_dic[split_name].append(file)

for key in split_dic:
    save_path_split = os.path.join(save_path, key)
    train_path = os.path.join(save_path_split, 'train')
    test_path = os.path.join(save_path_split, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for file in split_dic[key]:
        if np.random.rand() > 0.9:
            shutil.copy(os.path.join(all_data_path, file), os.path.join(test_path, file))
        else:
            shutil.copy(os.path.join(all_data_path, file), os.path.join(train_path, file))