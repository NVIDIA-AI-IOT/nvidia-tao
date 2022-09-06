#!/bin/sh

apt -qq install -y sox


sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.8 -y
apt install python3-pip -y
apt-get install python3.8-distutils
apt-get install python3.8-dev

rm /usr/bin/python3
ln -sf /usr/bin/python3.8 /usr/local/bin/python
ln -sf /usr/bin/python3.8 /usr/bin/python3

python3.8 -m pip install --upgrade pip
python3.8 -m pip install cython
python3.8 -m pip install zmq requests pandas IPython portpicker google-auth
python3.8 -m pip install --no-deps google-colab 
python3.8 -m pip install nvidia-pyindex

cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

cd /content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset ac3b2fdebe34f94b034854e7ccc4c667ad6447d0
python3.8 -m pip install cython
python3.8 -m pip install git+https://github.com/NVIDIA/NeMo.git@ac3b2fdebe34f94b034854e7ccc4c667ad6447d0#egg=nemo_toolkit[all]

cd /content/drive/MyDrive/ColabNotebooks/pytorch
rm -r kenlm
tar -xzf kenlm.tar.gz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
chmod a+rx kenlm

cd /content/drive/MyDrive/ColabNotebooks/pytorch
apt install libeigen3-dev
# Install KenLM python and pip requirements
rm -r kenlm-master
unzip -q master.zip
cd kenlm-master/
python3.8 -m pip install -e .
python3.8 -m pip install -r /content/drive/MyDrive/pyt/requirements-pip-lm.txt && rm requirements-pip.txt

python3.8 -m pip install /content/drive/MyDrive/pyt/nvidia_tao-0.1.560.dev0-py3-none-any.whl
python3.8 -m pip install /content/drive/MyDrive/pyt/lm_whls/*.whl
python3.8 -m pip install --ignore-installed --no-deps -r /content/drive/MyDrive/pyt/requirements-pip.txt
