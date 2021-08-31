# Spiking_Converted_YOLOv4
Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network

We provide three methods: Frequency, SAE, and LIF to convert dynamic vision sensor data into visualization data in ```Frequency、SAE、LIF``` floder

![](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4/blob/master/flowchart/flowchart.png)
## AEDAT4 files to .jpg or .avi
You must clone and install [DV-python](https://gitlab.com/inivation/dv/dv-python)

You can download [AEDAT4 files](https://drive.google.com/file/d/14MbYG6216m2hCdOdjKVSkYRfqeZZ29Fr/view?usp=sharing)

First install PIL and matplotlib
```
cd Frequency、SAE、LIF/
pip3 install scipy
pip3 install matplotlib
pip3 install Pillow
```
Please install it if opencv is not installed
```
pip3 install opencv-python 
```
You can use three methods to convert .jpg
```
python tool_LIF_aedat4.py
python tool_frequency_aedat4.py
python tool_sae_aedat4.py
```
Or you can use jupyter notebook for .avi
```
pip install jupyter notebook
tool_sae_aedat4_avi.ipynb
```

## MNIST-DVS-Detection
You must clone and install [Darknet](https://github.com/AlexeyAB/darknet)
Check the MakeFile and change the following parameters
```
GPU=0					
CUDNN=0					
CUDNN_HALF=0
OPENCV=0				
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0 # ZED SDK 3.0 and above
ZED_CAMERA_v2_8=0 # ZED SDK 2.X 
```


