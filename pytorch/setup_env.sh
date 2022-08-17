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

cd /content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset ac3b2fdebe34f94b034854e7ccc4c667ad6447d0
python3.8 -m pip install git+https://github.com/NVIDIA/NeMo.git@ac3b2fdebe34f94b034854e7ccc4c667ad6447d0#egg=nemo_toolkit[all]

python3.8 -m pip install /content/drive/MyDrive/pyt/nvidia_tao-0.1.560.dev0-py3-none-any.whl
python3.8 -m pip install /content/drive/MyDrive/pyt/cp38_whls/*.whl
python3.8 -m pip install --ignore-installed --no-deps -r /content/drive/MyDrive/pyt/requirements-pip.txt
python3.8 -m pip install pytorch_lightning==1.6.0
python3.8 -m pip install transformers==4.8.2
python3.8 -m pip install tokenizers==0.10.3
