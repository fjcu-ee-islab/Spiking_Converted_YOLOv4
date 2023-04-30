# Spiking_Converted_YOLOv4
Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network

We provide three methods: Frequency, SAE, and LIF to convert dynamic vision sensor data into visualization data in ```Frequency、SAE、LIF``` floder

We provide object detection trained on the MNIST-DVS dataset label by the auto_labeling algorithm in ```MNIST-DVS-Detection``` floder

We provide the auto_labeling algorithm program in ```Auto_labeling_algorithm``` floder

We provide object detection trained on the PAFBenchmark dataset label by the auto_labeling algorithm in ```PAFBenchmark``` floder

We provide object detection trained on the FJU_event_pedestrian_detection dataset label by the auto_labeling algorithm in ```fju_event_pedestrian_detection``` floder

We provide object detection based on dynamic vision sensor with spiking neural network trained on the FJU_event_pedestrian_detection dataset label by the auto_labeling algorithm in ```Spiking_converted_YOLOv4``` floder

![](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4/blob/master/flowchart/flowchart.png)

## FJUPD Event Dataset
FJUPD Event Dataset is available at [IEEE DataPort](https://dx.doi.org/10.21227/x8x3-mw77) and [FJUPD Download Link](https://u.pcloud.link/publink/show?code=kZO8ujVZz1BJJUPqzAS7v0h9M1gTJyTnhNW7)

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
## Auto_labeling algorithm
The environmental requirements we use are
```
Window 10
Visual studio 2017
OpenCV - 3.4.11
OpenCV_contrib - 3.4.11
Cmake 3.10.0
```
If you finish installing OpenCV, you can use the following program to test
```
/Auto_labeling_algorithm/test_opencv.cpp
```
The following programs can be used to automatically mark and test OpenCV_contrib, and most of the other programs are used to remove noise
```
/Auto_labeling_algorithm/CSRT_大量存圖_存取影像(可存原圖).cpp
```
## PAFBenchmark
Parameter adjustment and training methods are roughly the same as MNIST-DVS-Detection
You can download relevant training dataset [here](https://drive.google.com/file/d/1rhByl3rk0yGTepXOb9sQAFXmoZ19pghk/view?usp=sharing)
## fju_event_pedestrian_detection
Parameter adjustment and training methods are roughly the same as MNIST-DVS-Detection
You can download relevant training dataset [here](https://drive.google.com/file/d/14MbYG6216m2hCdOdjKVSkYRfqeZZ29Fr/view?usp=sharing)
## Spiking converted YOLOv4
You must clone and install [PyTorch-Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3)
Package version requirements
```
pytorch 1.3
python 3.7
```
If you encounter the following error
```
File "/home/shon/anaconda3/envs/torch1.3/lib/python3.7/site-packages/torch/tensor.py", line 433, in __array__
    return self.numpy()
TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```
Please go to the path listed above to modify the tensor.py
```
return self.numpy()
改成
return self.cpu().numpy()
```
train
```
python train.py --batch-size 32 --cfg /spiking_yolov4.cfg --data /fju_YOLOv4.data --weights ''
```
test
```
python test.py --cfg /spiking_yolov4.cfg --data /fju_YOLOv4.data --weights weights/best.pt --batch-size 32 --img-size 640
```
CNN to SNN
```
python ann_to_snn.py --cfg /spiking_yolov4.cfg --data /fju_YOLOv4.data --weights weights/best.pt --timesteps 32 --batch-size 1
```












