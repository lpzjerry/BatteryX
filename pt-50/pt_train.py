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

r = open('../list/train.txt').readlines()
imgdata = []
for i in r:
	e = i.split()
	img = cv2.imread('../data/'+e[0])
	resized_img = cv2.resize(img, (224, 224))
	imgdata.append(resized_img)

for i in range(len(imgdata)):
    im = imgdata[i]
    a = np.asarray(im)
    b = np.transpose(a,(2,0,1))
    c = np.reshape(b,(1,3,224,224))
    if i == 0:
        Xim = c
    else:
        Xim = np.concatenate((Xim,c),axis=0)

print(Xim.shape)

import keras.applications.resnet50 as resnet
from keras.preprocessing import image

model_f = resnet.ResNet50(weights='imagenet', include_top=False, pooling='avg')
Xf = model_f.predict(Xim)
print(Xf.shape)
np.save('trainXf.npy',Xf)
