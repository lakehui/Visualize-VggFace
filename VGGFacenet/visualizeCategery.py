#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:35:37 2017

@author: hh
"""

import caffe
import PIL.Image as Image
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()


deployfile = './model/VGG_FACE_deploy_newedit.prototxt'
weightfile = './model/VGG_FACE.caffemodel'


net = caffe.Net(deployfile, weightfile, caffe.TRAIN)


#preprocessing image
averageImg = np.array([129.1863,104.7624,93.5940]).reshape(3,1,1)
img_ori = Image.open('./model/backg.jpg')
img_ori = img_ori.resize((224,224))
img_array_ori = np.array(img_ori)
img_array_ori = img_array_ori.transpose(2,0,1)
img_array_ori = img_array_ori.astype(float)
img_array_ori -= averageImg

img_dst = Image.open('./model/161.jpg')
img_dst = img_dst.resize((224,224))
img_array_dst = np.array(img_dst)
img_array_dst = img_array_dst.transpose(2,0,1)
img_array_dst = img_array_dst.astype(float)
img_array_dst -= averageImg

#feed the image to data layer
net.blobs['data'].data[0,] = img_array_ori
net.blobs['data'].data[1,] = np.ones((3,224,224))
net.blobs['data'].data[2,] = img_array_dst


# optimize 
lr = 0.0001
steps = 500
momentum = 0.75
delta_update = np.zeros(img_array_ori.shape)
for i in range(steps):
    loss = net.forward()
    #for loss_blob, loss_value in loss.iteritems():
    print loss['loss/eucli_relu7_ori']

    diff = net.backward()
    delta_update = momentum*delta_update + lr*diff['data'][1,]
    net.blobs['data'].data[1,] -= delta_update

    

#return an image
img_ = net.blobs['data'].data[1,] + averageImg

img_ = abs(img_/img_.max())*255
img_ = img_.astype('uint8')
img = Image.fromarray(img_.transpose(1,2,0))

