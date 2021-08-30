#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    print(start_time)
    end_time = start_time + step_time
    img_count = 0
    
    while end_time <= reT4[-1]:
        
 
        while reT4[end_idx] < end_time:
            end_idx = end_idx + 1
        
        data_x = np.array(reX4[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(reY4[start_idx:end_idx]).reshape((-1, 1))
        data_T = np.array(reT4[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)
        
        timestamp=start_time*np.ones((260,346))
        
        for i in range(0, data.shape[0]):
            timestamp[data[i,1], data[i,0]]=data_T[i]
           
        grayscale = np.flip(255*(timestamp-start_time)/step_time, 0).astype(np.uint8)#The normalization formula
        grayscale = np.flip(grayscale,0)
        cv2.imshow('img',grayscale)
       
        cv2.waitKey(5)
        wfile='/home/shon/code/PAFBenchmark/hand_sae/sae_' + str(img_count) + '.jpg'
        cv2.imwrite(wfile,grayscale)

        start_time = end_time
        end_time += step_time
        start_idx = end_idx
        img_count += 1
        
        
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




