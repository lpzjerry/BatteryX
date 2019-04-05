import cv2
import os
import numpy as np

r = open('list/all.txt').readlines()
for line in r:
	img_name = line.split()[0]
	img = cv2.imread('data/'+img_name)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	b_image = img[:,:,0]
	g_image = img[:,:,1]
	r_image = img[:,:,2]
	cv2.imwrite('dataset/'+img_name, gray_image)
	cv2.imwrite('dataset/'+img_name[:-4]+'_b.png', b_image)
	cv2.imwrite('dataset/'+img_name[:-4]+'_g.png', g_image)
	cv2.imwrite('dataset/'+img_name[:-4]+'_r.png', r_image)	
