# Spiking_Converted_YOLOv4
Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network

![](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4/blob/master/flowchart/flowchart.png)
## AEDAT4 files to .jpg or .avi
You must clone and install [DV-python](https://gitlab.com/inivation/dv/dv-python)
You can download [AEDAT4 files](https://drive.google.com/file/d/14MbYG6216m2hCdOdjKVSkYRfqeZZ29Fr/view?usp=sharing)
First install PIL and matplotlib
```
cd Frequency、SAE、LIF
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
