import cv2
import os
import numpy as np

r1 = open('train.txt').readlines()
r2 = open('test.txt').readlines()
wrtrain = open('train_r.txt','w')
wrtest = open('test_r.txt','w')
wgtrain = open('train_g.txt','w')
wgtest = open('test_g.txt','w')
wbtrain = open('train_b.txt','w')
wbtest = open('test_b.txt','w')
for line in r1:
	e = line.split()
	wrtrain.write(e[0][:-4]+'_r.png '+e[1]+'\n')
	wgtrain.write(e[0][:-4]+'_g.png '+e[1]+'\n')
	wbtrain.write(e[0][:-4]+'_b.png '+e[1]+'\n')
for line in r2:
	e = line.split()
	wrtest.write(e[0][:-4]+'_r.png '+e[1]+'\n')
	wgtest.write(e[0][:-4]+'_g.png '+e[1]+'\n')
	wbtest.write(e[0][:-4]+'_b.png '+e[1]+'\n')
