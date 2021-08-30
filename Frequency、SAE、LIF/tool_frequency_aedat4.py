#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import struct
import math
import cv2
import numpy as np
import argparse
from dv import AedatFile
from PIL import Image
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    #aedat4使用
    with AedatFile("/home/shon/code/DV_data/dvSave-2021_01_06_02_12_26.aedat4") as f:
    # events will be a named numpy array
        events = np.hstack([packet for packet in f['events'].numpy()])
    
    # Access information of all events by type
        timestamps4, x4, y4, polarities4 = events['timestamp'], events['x'], events['y'], events['polarity']
    
    reT4 = np.array(timestamps4).reshape((-1, 1))#(行，列) (-1,1)代表我不知道要轉成幾行，但是固定要1列，-1就為最大行
    reX4 = np.array(x4).reshape((-1, 1))
    reY4 = np.array(y4).reshape((-1, 1))
    reP4 = np.array(polarities4).reshape((-1, 1))
    
    
    
    
    step_time = 10000 #代表是0.01秒為一張
   
    start_idx = 0
    end_idx = 0
    start_time = reT4[0]
    print('start_time is ',start_time)
    end_time = start_time + step_time
    img_count = 0
    
    while end_time <= reT4[-1]:#-1代表該元素最後一行
       
 
        while reT4[end_idx] < end_time:
            end_idx = end_idx + 1
        
        data_x = np.array(reX4[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(reY4[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)
       
        counter=np.zeros((260,346))
        
        for i in range(0, data.shape[0]):
            counter[data[i,1], data[i,0]]+=1  #計算每個像素點發生的事件次數
        
        
        counter = 255*2*(1/(1+np.exp(-counter))-0.5)  #The normalization formula
        
        
        cv2.imshow('counter',counter)
        
        cv2.waitKey(5)
        wfile='/home/shon/code/PAFBenchmark/hand_frequency/' + str(img_count) + '.jpg'
        cv2.imwrite(wfile,counter)
        
        
        start_time = end_time
        end_time += step_time
        start_idx = end_idx
        img_count += 1
        
    print('  end_time is ',end_time)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

