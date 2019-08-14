# Installation and Running Guide
## Author: Yun Liu
## Date: 2019/01/01

## 1. Devices, Drivers and System

### 1.1 GPU Devices
We only test this program on Nvidia GTX 1060 6GB and Nvidia GTX 1080 Ti, so make sure you have those devices before running this program.

### 1.2 Drivers
### 1.2.1 CUDA Drivers
Make sure you install CUDA 9.0 as your graphics card driver.
### 1.2.2 Cudnn
Make sure you install Cudnn 7.x on your computer.

### 1.3 System
We support Windows 10 x64 professional, but it should work fine with Ubuntu 16.04 LTS and Ubuntu 18.04 LTS.

## 2. Python Libs
Following packages should be installed before running:

PackageName Version
Click	7.0
Flask	1.0.2
Jinja2	2.10
MarkupSafe	1.1.0
Pillow	5.3.0
PyWavelets	1.0.1
Werkzeug	0.14.1
cloudpickle	0.6.1
cycler	0.10.0
dask	1.0.0
decorator	4.3.0
itsdangerous	1.1.0
kiwisolver	1.0.1
matplotlib	3.0.2
networkx	2.2
numpy	1.15.4
opencv-python	3.4.4.19
pip	10.0.1
pyparsing	2.3.0
python-dateutil	2.7.5
scikit-image	0.14.1
scipy	1.1.0
setuptools	39.0.1
six	1.11.0
toolz	0.9.0
torch	0.4.1
torchvision	0.2.1
tqdm	4.28.1
waitress	1.1.0

## 3.Running
### 3.1 Local
cd $ROOTDIR$ // cd into the root project directory
python app.py // start flask service and you can find this website running at http://localhost:5000/

### 3.2 Remote
You need to comment line 64 and uncomment line 66 in app.py first
cd $ROOTDIR$ // cd into the root project directory
python app.py // start flask service and you can find this website running at http://YOURIPADDR:8080/ and can be accessed remotely.

## 4.Trouble Shooting
First detection will be much slower because system need to load weights and networks into memory.
