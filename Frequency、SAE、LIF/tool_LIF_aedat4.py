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
                         
class SNN():
    """Spiking Neural Network.
    ts: timestamp list of the event stream.
    x: x-coordinate list of the event stream.
    y: y-coordinate list of the event stream.
    pol: polarity list of the event stream.  
    threshold: threshold of neuron firing.
    decay: decay of MP with time.
    margin: margin for lateral inhibition.
    spikeVal: MP increment for each event.
    network: MP of each neuron.
    timenet: firing timestamp for each neuron.
    firing: firing numbers for each neuron.
    image: converted output grayscale image.
    """                    
    def __init__(self): 
        self.ts = []
        self.x = []
        self.y = []
        self.pol = []
        self.threshold = 1.2                                          
        self.decay     = 0.02                                          
        self.margin    = 3                                             
        self.spikeVal  = 1
        self.network   = np.zeros((260, 346), dtype = np.float64)
        self.timenet   = np.zeros((260, 346), dtype = np.int64)    
        self.firing = np.zeros((260, 346), dtype = np.int64)
        self.image = np.zeros((260, 346), dtype = np.int64)
    
    def init_timenet(self, t):
        """initialize the timenet with timestamp of the first event"""
        self.timenet[:] = t

    def spiking(self, data):
        """"main process"""
        count = 0
        img_count = 0   
        startindex = 0

        for line in data:
            self.ts.insert(count, int(line[0]))
            self.x.insert(count, int(line[1]))
            self.y.insert(count, int(line[2]))
            self.pol.insert(count, int(line[3]))

            if count == 0:
                self.init_timenet(self.ts[0])
                starttime = self.ts[0]
               
            self.neuron_update(count, self.spikeVal)
            
            if self.ts[count] - starttime > 10000: #多少時間區間為一張圖
                self.show_image(img_count)
                img_count += 1
                starttime = self.ts[count]
                self.image *= 0
                self.firing *= 0

            count += 1

        print('done')
        
    def clear_neuron(self, position):
        """reset MP value of the fired neuron"""             
        for i in range((-1)*self.margin, self.margin):
            for j in range((-1)*self.margin, self.margin):
                if position[0]+i<0 or position[0]+i>=180 or position[1]+j<0 or position[1]+j>=180:
                    continue
                else:
                    self.network[ position[0]+i ][ position[1]+j ] = 0.0

    def neuron_update(self, i, spike_value):
        """update the MP values in the network"""
        x = self.x[i]
        y = self.y[i]
        escape_time = (self.ts[i]-self.timenet[y][x])/1000.0
        residual = max(self.network[y][x]-self.decay*escape_time, 0)
        self.network[y][x] = residual + spike_value
        self.timenet[y][x] = self.ts[i]
        if self.network[y][x] > self.threshold:
            self.firing[y][x] += 1      # countor + 1
            self.clear_neuron([x,y])

    def show_image(self, img_count):
        """convert to and save grayscale images"""
        #self.image = np.flip(255*2*(1/(1+np.exp(-self.firing))-0.5),0)
        self.image = 255*2*(1/(1+np.exp(-self.firing))-0.5)
        outputfile = '/home/shon/code/PAFBenchmark/hand_LIF/' + str(img_count) + '.jpg'
        cv2.imshow('img', self.image)
        cv2.waitKey(5)
        cv2.imwrite(outputfile, self.image)

    
if __name__ == '__main__':
    # parse the command line argument
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
    
    
    data = np.hstack((reT4, reX4, reY4, reP4))
    print(np.shape(data))
    dvs_snn = SNN()
    dvs_snn.spiking(data)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:




