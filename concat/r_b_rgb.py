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
train_r = np.load('../fin-50_r/featmap/train_feature.npy')
train_g = np.load('../fin-50_g/featmap/train_feature.npy')
train_b = np.load('../fin-50_b/featmap/train_feature.npy')
train_rgb = np.load('../fin-50_rgb/featmap/train_feature.npy')
test_r = np.load('../fin-50_r/featmap/test_feature.npy')
test_g = np.load('../fin-50_g/featmap/test_feature.npy')
test_b = np.load('../fin-50_b/featmap/test_feature.npy')
test_rgb = np.load('../fin-50_rgb/featmap/test_feature.npy')

# process features
trainXf = np.concatenate((train_r, train_b, train_rgb),axis=1)
testXf = np.concatenate((test_r, test_b, test_rgb),axis=1)

# Extract trainY
r = open('../list/train.txt').readlines()
trainY = []
for line in r:
	label = int(line.split()[1][0])
	trainY.append(label)
# Extract trainY
r = open('../list/test.txt').readlines()
testY = []
for line in r:
	label = int(line.split()[1][0])
	testY.append(label)
# visualize data
print(trainXf.shape)
print(testXf.shape)
print(len(trainY))
print(len(testY))

# train classifiers
# 1. Random Forest
clf = ensemble.RandomForestClassifier(n_estimators=500, random_state=1234, n_jobs=-1)
clf.fit(trainXf, trainY)
predY = clf.predict(testXf)
#acc = metrics.accuracy_score(testY, predY)
#print("Random Forest accuracy =",acc)
fpr, tpr, thresholds = metrics.roc_curve(testY, predY, pos_label=1)
print("SVM AUC =", metrics.auc(fpr, tpr))

# 2. SVM
clf2 = svm.SVC(probability=True, kernel='poly', gamma=1, C=5)
clf2.fit(trainXf, trainY)
predY = clf2.predict(testXf)
#acc = metrics.accuracy_score(testY, predY)
#print("SVM accuracy =",acc)
fpr, tpr, thresholds = metrics.roc_curve(testY, predY, pos_label=1)
print("SVM AUC =", metrics.auc(fpr, tpr))

# 3. Logistic Regression
cla = linear_model.LogisticRegressionCV(Cs=logspace(-2,2,10), cv=2)
cla.fit(trainXf, trainY)
predY = cla.predict(testXf)
#acc = metrics.accuracy_score(testY, predY)
#print("LR accuracy =",acc)
fpr, tpr, thresholds = metrics.roc_curve(testY, predY, pos_label=1)
print("SVM AUC =", metrics.auc(fpr, tpr))
