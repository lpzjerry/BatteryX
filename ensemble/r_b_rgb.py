import matplotlib.pyplot as plt
import matplotlib
from numpy import *
from sklearn import *
import os
import zipfile
import fnmatch
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, \
                Input, GlobalAveragePooling2D
from keras.preprocessing import image
import keras
import tensorflow
import logging
logging.basicConfig()
import struct
# use channels first representation for images
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.callbacks import TensorBoard
import cv2

# load features
#train_r = np.load('../fin-50_r/featmap/train_predict.npy')
#train_b = np.load('../fin-50_b/featmap/train_predict.npy')
#train_rgb = np.load('../fin-50_rgb/featmap/train_predict.npy')
test_r = np.load('../fin-50_r/featmap/test_predict.npy')
test_b = np.load('../fin-50_b/featmap/test_predict.npy')
test_rgb = np.load('../fin-50_rgb/featmap/test_predict.npy')

# process features
#trainXf = train_r + train_b - 1
predY = []
testXf = test_rgb + test_r + test_b - 1.5
for i in range(457):
    if testXf[i][0] > 0:
        predY.append(1)
    else:
        predY.append(0)
# Extract trainY
#r = open('../list/train.txt').readlines()
#trainY = []
#for line in r:
#	label = int(line.split()[1][0])
#	trainY.append(label)
# Extract trainY
r = open('../list/test.txt').readlines()
testY = []
for line in r:
	label = int(line.split()[1][0])
	testY.append(label)
# visualize data
#print(trainXf.shape)
print(testXf.shape)
#print(len(trainY))
print(len(testY))
fpr, tpr, thresholds = metrics.roc_curve(testY, predY, pos_label=1)
print("SVM AUC =", metrics.auc(fpr, tpr))


