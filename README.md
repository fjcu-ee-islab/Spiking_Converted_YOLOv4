# Spiking_Converted_YOLOv4
Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network

We provide three methods: Frequency, SAE, and LIF to convert dynamic vision sensor data into visualization data in ```Frequency、SAE、LIF``` floder

We provide object detection trained on the MNIST-DVS dataset label by the auto_labeling algorithm in ```MNIST-DVS-Detection``` floder

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
GPU=1					
CUDNN=1					
CUDNN_HALF=0
OPENCV=1				
AVX=0
OPENMP=0
LIBSO=0
ZED_CAMERA=0 # ZED SDK 3.0 and above
ZED_CAMERA_v2_8=0 # ZED SDK 2.X 
```
Enter the darknet folder and execute
```
cd MNIST-DVS-Detection/darknet/
make
```
Download and use the trained weights for testing [Weights](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view)
Use the pictures that come with darknet for testing
```
./darknet detect /darknet/cfg/yolov4.cfg /yolov4.weights /darknet/data/dog.jpg
```
You can download the training data that we have marked [download](https://drive.google.com/file/d/1X1C-MsoPxtH6S5pBU_F2WlpOiPYIzM2Q/view?usp=sharing)
Enter the train and dev folders respectively and execute the following programs to generate txt files with absolute paths
```
cd train
ls -d "$PWD"/*.jpg > train.txt 
cd dev
ls -d "$PWD"/*.jpg > dev.txt 
```
Change the .cfg file
batch、subdivisions 
```
batch = 64
subdivisions = 16           //Can be adjusted according to the memory
```
Change max_batches = clsss * 2000
```
max_batches = 20000 
```
Change steps = max_batches * 0.8, 0.9
```
steps = 16000, 18000 
```
Change width and height (must be a multiple of 32)
```
width = 416
height = 416 
```
Change the classes of the three [yolo] blocks to the categories that need to be identified
```
classes=10
```
The filter of the previous [convolution] block of the three [yolo] blocks is changed to (classes + 5) x 3, we have 3 categories so it is changed to 24, remember that there are three places to modify
```
filters = 45
```
Add .name file and .data file
.name file is the object type to be recognized
```
0
1
2
3
4
5
6
7
8
9
```
.data file. Store some parameters, the number of object types, and the path (train.txt & dev.txt in the previous step)
```
classes=Number of object classes
train=data/train.txt (the train.txt path of the previous step)
valid=data/dev.txt (dev.txt path in the previous step)
names=data/mask.names (.names file path)
backup=backup/ (Weight storage path)
```
To start training, first download the pre-training weights trained by others [download](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)
```
./darknet detector train /mnist.data /yolov4_MNIST_DVS512.cfg /yolov4.conv.137 
```
Use a single image for testing
```
./darknet detector test /mnist.data /yolov4_MNIST_DVS512.cfg /yolov4_last.weights /images.jpg
```
Use our test program
```
python test.py
```







