# 1.four_point_transform & resize to (440, 140)
# 2.rgb2gray
# 3.cut to province as patch[0], area as patch[1], letter as patch[2:6]
# & save them to folders named as filename


import json
import cv2
import os
from PIL import Image
import numpy as np
from imutils.perspective import four_point_transform


if not os.path.exists('./preprocessed/train'):
    os.makedirs('./preprocessed/train')
if not os.path.exists('./preprocessed/val'):
    os.makedirs('./preprocessed/val')

height = 90
width = 45
span = 12
x0 = 13
x1 = 13 + width + span
x2 = 13 + width*2 + 15 + span*3
x3 = 13 + width*3 + 15 + span*4
x4 = 13 + width*4 + 15 + span*5
x5 = 13 + width*5 + 15 + span*6
x6 = 13 + width*6 + 15 + span*7
y0 = 25
x = [x0, x1, x2, x3, x4, x5, x6]
y = [y0, y0, y0, y0, y0, y0, y0]


# train data
with open('./LPD_dataset/train/via_region_data.json', 'r', encoding='utf-8') as load_f:
    load_dict_train = json.load(load_f)
    print(len(load_dict_train))
load_dict_train_key = list(load_dict_train.keys())
for i in range(len(load_dict_train_key)):
    filename = load_dict_train[load_dict_train_key[i]]['filename']
    print(filename)
    points = [[load_dict_train[load_dict_train_key[i]]['regions'][0]['shape_attributes']['all_points_x'][j],
               load_dict_train[load_dict_train_key[i]]['regions'][0]['shape_attributes']['all_points_y'][j]] for j in range(4)]
    print(points)

    # 1.four_point_transform via json file & resize to (440, 140)
    correct_path = './preprocessed/train/correct'
    if not os.path.exists(correct_path):
        os.makedirs(correct_path)
    img_path = './LPD_dataset/train/' + filename
    img = Image.open(img_path)
    rect = four_point_transform(cv2.cvtColor(
        np.asarray(img), cv2.COLOR_RGB2BGR), np.array(points))
    rect_resize = cv2.resize(rect, (440, 140))
    rect_Img = Image.fromarray(cv2.cvtColor(rect_resize, cv2.COLOR_BGR2RGB))
    rect_Img.save(correct_path + '/' + filename)

    # 2.rgb2gray
    rgb2gray_path = './preprocessed/train/rgb2gray'
    if not os.path.exists(rgb2gray_path):
        os.makedirs(rgb2gray_path)
    gray = cv2.cvtColor(rect_resize, cv2.COLOR_BGR2GRAY)
    gray_Img = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    gray_Img.save(rgb2gray_path + '/' + filename)

    # 3.crop img to 7 patches
    filepath = './preprocessed/train/crop/'+filename[:-4]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    patch = [[] for n in range(7)]
    for i in range(7):
        patch[i] = gray_Img.crop((x[i], y[i], x[i]+width, y[i]+height))
        patch[i].save(filepath + '/' + str(i) + '_' + filename[i]+'.jpg')


# val data
with open('./LPD_dataset/val/via_region_data.json', 'r', encoding='utf-8') as load_f:
    load_dict_val = json.load(load_f)
    print(len(load_dict_val))
load_dict_val_key = list(load_dict_val.keys())
for i in range(len(load_dict_val_key)):
    filename = load_dict_val[load_dict_val_key[i]]['filename']
    print(filename)
    points = [[load_dict_val[load_dict_val_key[i]]['regions'][0]['shape_attributes']['all_points_x'][j],
               load_dict_val[load_dict_val_key[i]]['regions'][0]['shape_attributes']['all_points_y'][j]] for j in range(4)]
    print(points)

    # 1.four_point_transform via json file & resize to (440, 140)
    correct_path = './preprocessed/val/correct'
    if not os.path.exists(correct_path):
        os.makedirs(correct_path)
    img_path = './LPD_dataset/val/' + filename
    img = Image.open(img_path)
    rect = four_point_transform(cv2.cvtColor(
        np.asarray(img), cv2.COLOR_RGB2BGR), np.array(points))
    rect_resize = cv2.resize(rect, (440, 140))
    rect_Img = Image.fromarray(
        cv2.cvtColor(rect_resize, cv2.COLOR_BGR2RGB))
    rect_Img.save(correct_path + '/' + filename)

    # 2.rgb2gray
    rgb2gray_path = './preprocessed/val/rgb2gray'
    if not os.path.exists(rgb2gray_path):
        os.makedirs(rgb2gray_path)
    gray = cv2.cvtColor(rect_resize, cv2.COLOR_BGR2GRAY)
    # for i in range(len(gray)):
    #     for j in range(len(gray[i])):
    #         if gray[i][j] < 125:
    #             gray[i][j] = 0
    #         else:
    #             gray[i][j] = 255
    gray_Img = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    gray_Img.save(rgb2gray_path + '/' + filename)

    # 3.crop img to 7 patches
    filepath = './preprocessed/val/crop/'+filename[:-4]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    patch = [[] for n in range(7)]
    for i in range(7):
        patch[i] = gray_Img.crop((x[i], y[i], x[i]+width, y[i]+height))
        patch[i].save(filepath + '/' + str(i) + '_' + filename[i]+'.jpg')
