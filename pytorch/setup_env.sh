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

python3.8 -m pip install --upgrade pip
python3.8 -m pip install Cython==0.29.36
python3.8 -m pip install nvidia-pyindex
python3.8 -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

sudo apt-get install build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev libsndfile1 sox libsox-fmt-mp3 ffmpeg libfreetype6 libopenblas-dev libssl-dev -y 

# Install Cmake
sudo mkdir -p /tmp_dir_cmake && sudo chmod -R 777 /tmp_dir_cmake
cd /tmp_dir_cmake
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
sudo ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

#Install tao whls
python3.8 -m pip install nvidia-tao==5.0.0
python3.8 -m pip install nvidia-tao-pyt==5.1.0
python3.8 -m pip install Cython==0.29.36 && pip install --ignore-installed -r PATH_TO_COLAB_NOTEBOOKS/pytorch/requirements-pip.txt
python3.8 -m pip install --ignore-installed --no-deps -r PATH_TO_COLAB_NOTEBOOKS/pytorch/requirements-pip-pytorch.txt

python3.8 -m pip install pytest
python3.8 -m pip install nvidia-ml-py==11.515.75
