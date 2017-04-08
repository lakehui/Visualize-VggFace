#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:49:24 2017

@author: hh
"""


import caffe

import numpy as np


class MTripLossLayer(caffe.Layer):

    def setup(self, bottom, top):
                    
        if len(bottom) != 3:
            raise Exception("Only need to define two bottoms in MEucliLossLayer.")

        if len(top) != 1:
            raise Exception("not define top in MEucliLossLayer.")
            
        self.batchsize = bottom[0].data.shape[0]

    def reshape(self, bottom, top):
        top[0].reshape(self.batchsize, 1)
           


    def forward(self, bottom, top):
        self.diff1 = np.zeros(bottom[0].data.shape)
        self.diff2 = np.zeros(bottom[2].data.shape)
        #self.mask = bottom[2].data < bottom[2].data.mean()
        #self.mask = bottom[2].data < bottom[2].data.min()
        self.mask = bottom[0].data > bottom[2].data
        #self.mask = bottom[0].data > 0

        print 'mask: ', self.mask.mean()
        
        for i in range(self.batchsize):       
            #tmp_num = bottom[0].data.size / self.batchsize
            
            top[0].data[i,] = np.sum(self.mask[i,] * np.square(bottom[0].data[i,] - bottom[1].data[i,])) \
                    + np.sum((1 -self.mask[i,]) * np.square(bottom[2].data[i,] - bottom[1].data[i,]))
            self.diff1[i] = self.mask * (bottom[0].data[i,] - bottom[1].data[i,])
            self.diff2[i] = (1-self.mask) * (bottom[2].data[i,] - bottom[1].data[i,])
            

    def backward(self, top, propagate_down, bottom):

        if propagate_down[0]:
            bottom[0].diff[...] = self.diff1
            bottom[1].diff[...] = -1. * self.diff1 + -1. * self.diff2
            bottom[2].diff[...] = self.diff2

        


    def check_params(params):
        if len(params) != 0:
            raise Exception("don't define any params")
