#!/bin/sh

# Install Python 3.6 as the default version
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.6 -y
apt install python3-pip -y
apt-get install python3.6-distutils
apt-get install python3.6-dev

# Set Python 3.6 as the default version
rm /usr/bin/python
ln -sf /usr/bin/python3.6 /usr/bin/python3
ln -sf /usr/bin/python3.6 /usr/local/bin/python

python3.6 -m pip install --upgrade pip
python3.6 -m pip install google-colab
python3.6 -m pip install nvidia-pyindex

# Install Tensorflow
python3.6 -m pip install https://developer.download.nvidia.com/compute/redist/nvidia-horovod/nvidia_horovod-0.20.0+nv20.10-cp36-cp36m-linux_x86_64.whl
python3.6 -m pip install https://developer.download.nvidia.com/compute/redist/nvidia-tensorflow/nvidia_tensorflow-1.15.4+nv20.10-cp36-cp36m-linux_x86_64.whl

# Install Cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

# Install dependencies
python3.6 -m pip install  PATH_TO_GENERAL_WHL/*.whl
python3.6 -m pip install --ignore-installed PyYAML -r PATH_TO_COLAB_NOTEBOOKS/tensorflow/requirements-pip.txt -f https://download.pytorch.org/whl/torch_stable.html --extra-index-url https://developer.download.nvidia.com/compute/redist

# Install code related wheels
python3.6 -m pip install PATH_TO_CODEBASE_WHL/*.whl

