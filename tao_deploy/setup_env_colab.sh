#!/bin/sh

#Install python3.8
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.8 -y
apt install python3-pip -y
apt-get install python3.8-distutils
apt-get install python3.8-dev

#Set python3.8 as the default version
rm /usr/bin/python3
ln -sf /usr/bin/python3.8 /usr/bin/python3
ln -sf /usr/bin/python3.8 /usr/local/bin/python

#Install pip dependencies
python3.8 -m pip install --upgrade pip
python3.8 -m pip install cython
python3.8 -m pip install nvidia-ml-py
python3.8 -m pip install nvidia-pyindex
python3.8 -m pip install --upgrade setuptools
python3.8 -m pip install pycuda==2020.1
python3.8 -m pip install nvidia-eff-tao-encryption==0.1.7
python3.8 -m pip install nvidia-eff==0.6.2
python3.8 -m pip install cffi

#Install TensorRT whl files
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/python/tensorrt-*-cp38-none-linux_x86_64.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/onnx_graphsurgeon/onnx_graphsurgeon*.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/graphsurgeon/graphsurgeon*.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/uff/uff*.whl

#Install TAO deploy whl
python3.8 -m pip install nvidia_tao_deploy==4.0.0
