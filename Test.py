# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from Features import *
from tool import *
root ='/home/tas/桌面/机器学习/testCode/车辆寻找和跟踪/carFinding/'

### 调配参数
color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions，颜色特征的图片大小
hist_bins = 32  # Number of histogram bins 颜色直方图的柱大小
spatial_feat = True  # Spatial features on or off 使用颜色特征
hist_feat = True  # Histogram features on or off 使用颜色直方图特征
hog_feat = True  # HOG features on or off 使用方向直方图特征
y_start_stop = [None, None]  # Min and max in y to search in slide_window() 最大和最小搜索框

Y_MIN = 440
THRES_LEN = 32
track_list=[]
n = 0
boxes =[]

def GetData(str):
    # 读取数据
    # glob 通过匹配查找文件
    # images = glob.glob('data_more/*/*/*.png')
    images = glob.glob(root+str)
    notcars = []
    cars = []
    for image in images:
        if 'non-' in image:
            notcars.append(image)
        else:
            cars.append(image)

    print(len(notcars))
    print(len(cars))
    return cars, notcars

def train_test():

    cars, notcars = GetData('data_more/*/*/*.png')
    # 获取特征
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    print('Car samples: ', len(car_features))
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    print('notCar samples: ', len(notcar_features))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # 定义label
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # 切割训练集
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # 使用SVC
    svc = LinearSVC(loss='hinge')
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # 查看训练分数
    print(svc.score(X_test, y_test))
    joblib.dump(X_scaler, root + 'save/scaler.m')
    joblib.dump(svc, root + 'save/model.m')
    return svc, X_scaler

def test_one(img, svc, X_scaler,ifGetOne=True):
    # image = cv2.imread('road.jpeg')
    image = img #cv2.imread('test6.jpg')
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 600],
                           xy_window=(128, 128), xy_overlap=(0.85, 0.85))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    if ifGetOne:
        window_img = GetMultiHeat(image, hot_windows)
    else:
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    return window_img
    #cv2.imshow('result', window_img)
    #cv2.waitKey(0)
    #plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))

def test_oneImg(img, svc, X_scaler,ifGetOne=True):
    global track_list, n, boxes
    draw_image = np.copy(img)
    #hot_windows = find_car(img, 400, 650, 0, img.shape[1], 2, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
    n += 1
    if n%2==0:
        boxes = []
        ystart = 400
        boxes = find_car(img, ystart, 650, 720, 1280, 2.0, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
        boxes += find_car(img, ystart, 500, 720, 1280, 1.5, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
        boxes += find_car(img, ystart, 650, 0, 330, 2.0, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
        boxes += find_car(img, ystart, 500, 0, 330, 1.5, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
        boxes += find_car(img, ystart, 460, 330, 950, 0.75, 3, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
        for track in track_list:
            y_loc = track[1] + track[3]
            lane_w = (y_loc * 2.841 - 1170.0) / 3.0
            if lane_w < 96:
                lane_w = 96
            lane_h = lane_w / 1.2
            lane_w = max(lane_w, track[2])
            xs = track[0] - lane_w
            xf = track[0] + lane_w
            if track[1] < Y_MIN:
                track[1] = Y_MIN
            ys = track[1] - lane_h
            yf = track[1] + lane_h
            if xs < 0: xs = 0
            if xf > 1280: xf = 1280
            if ys < Y_MIN - 40: ys = Y_MIN - 40
            if yf > 720: yf = 720
            size_sq = lane_w / (0.015 * lane_w + 0.3)
            scale = size_sq / 64.0
            # Apply multi scale image windows
            boxes += find_car(img, ys, yf, xs, xf, scale, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
            boxes += find_car(img, ys, yf, xs, xf, scale * 1.25, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
            boxes += find_car(img, ys, yf, xs, xf, scale * 1.5, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)
            boxes += find_car(img, ys, yf, xs, xf, scale * 1.75, 2, pix_per_cell, X_scaler, svc, orient, cell_per_block, spatial_size, hist_bins)

    if ifGetOne:
        window_img, track_list = GetMultiHeat(img, boxes, track_list, Y_MIN, THRES_LEN)
    else:
        window_img = draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)

    return window_img

def train(img,ifGetOne=True):
    try:
        svc = joblib.load(root + 'save/model.m')
        X_scaler = joblib.load(root+'save/scaler.m')
    except:
        svc, X_scaler = train_test()
    #img = cv2.imread(str)
    #return test_one(img, svc, X_scaler)
    return test_oneImg(img, svc, X_scaler,ifGetOne)

if __name__ == '__main__':
    if True:
        cap = cv2.VideoCapture(root+'project_video.mp4')
        while (cap.isOpened()):
            # 读取图片
            ret, img = cap.read()
            finalImg = train(img, True)
            cv2.imshow('Contours', finalImg)
            k = cv2.waitKey(10)
            if k == 27:
                break
    else:
        img = cv2.imread(root + 'test6.jpg')
        #cv2.imshow('1',img)
        #cv2.waitKey(0)
        finalImg = train(img, True)
        cv2.rectangle(finalImg, (720, 380), (1280, 650), (0, 255, 0), 6)
        cv2.rectangle(finalImg, (720, 380), (1280, 500), (255, 0, 0), 6)
        cv2.rectangle(finalImg, (0, 380), (320, 650), (128, 0, 0), 6)
        cv2.rectangle(finalImg, (0, 380), (300, 500), (0, 128, 56), 6)
        cv2.rectangle(finalImg, (330, 380), (950, 460), (0, 128, 56), 6)
        cv2.imshow('Contours', finalImg)
        cv2.waitKey(0)
