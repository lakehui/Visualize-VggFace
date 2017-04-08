#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:53:46 2017

@author: hh

loss function : only calculate the distance when feature[i] > 0  
"""


import caffe

import numpy as np



class MEucliLossLayer(caffe.Layer):

    def setup(self, bottom, top):
                    
        if len(bottom) != 2:
            raise Exception("Only need to define two bottoms in MEucliLossLayer.")

        if len(top) != 1:
            raise Exception("not define top in MEucliLossLayer.")
            
        self.batchsize = bottom[0].data.shape[0]

    def reshape(self, bottom, top):
        top[0].reshape(self.batchsize, 1)
           


    def forward(self, bottom, top):
        self.diff = np.zeros(bottom[0].data.shape)
        
        self.mask = bottom[0].data[...] > bottom[0].data.mean()
        
        for i in range(self.batchsize):       
            #tmp_num = bottom[0].data.size / self.batchsize
            
            top[0].data[i,] = np.sum(self.mask[i,] * np.square(bottom[0].data[i,] - bottom[1].data[i,]))
            self.diff[i] = self.mask * (bottom[0].data[i,] - bottom[1].data[i,])
            
        
        

    def backward(self, top, propagate_down, bottom):

        if propagate_down[0]:
            bottom[0].diff[...] = self.diff
            bottom[1].diff[...] = -1. * self.diff

        


    def check_params(params):
        if len(params) != 0:
            raise Exception("don't define any params")
