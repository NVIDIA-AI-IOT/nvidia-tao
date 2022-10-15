#!/bin/sh

# Update ubuntu to 20.04
sudo apt update && sudo apt upgrade && sudo apt-get install libcudnn8 libcudnn8-dev libnccl-dev libnccl2 --allow-change-held-packages && sudo apt dist-upgrade
sudo apt autoremove
sudo apt install update-manager-core
sudo ln -sf /usr/bin/python3.6 /usr/bin/python3
export DEBIAN_FRONTEND=noninteractive
sudo do-release-upgrade -f DistUpgradeViewNonInteractive

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
python3.6 -m pip install nvidia-pyindex

# Install Tensorflow
sudo apt-get install -y cuda-compat-11-7
python3.6 -m pip install https://developer.download.nvidia.com/compute/redist/nvidia-tensorflow/nvidia_tensorflow-1.15.5+nv22.07-5236135-cp36-cp36m-linux_x86_64.whl

# Install Cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

# Install dependencies
python3.6 -m pip install  /content/drive/MyDrive/tf/general_whl/*.whl
python3.6 -m pip install --ignore-installed PyYAML -r /content/drive/MyDrive/ColabNotebooks/tensorflow/requirements-pip.txt -f https://download.pytorch.org/whl/torch_stable.html --extra-index-url https://developer.download.nvidia.com/compute/redist

# Install code related wheels
python3.6 -m pip install /content/drive/MyDrive/tf/codebase_whl/*.whl
