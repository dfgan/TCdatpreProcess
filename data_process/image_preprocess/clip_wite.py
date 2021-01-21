# -*- coding: utf-8 -*-
# @time     :21-1-12 下午7:57
import cv2
import os
import numpy as np

path = './test_image'
files = os.listdir(path)
names = [os.path.join(path, i) for i in files]
for name in names:
    img = cv2.imread(name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img_gray)
    cv2.waitKey(0)