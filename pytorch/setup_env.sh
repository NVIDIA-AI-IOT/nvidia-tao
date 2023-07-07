#!/bin/sh

apt -qq install -y sox

#Install python3.7
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.7 -y
apt install python3-pip -y
apt-get install python3.7-distutils
apt-get install python3.7-dev

#Set python3.7 as the default version
rm /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/bin/python3
ln -sf /usr/bin/python3.7 /usr/local/bin/python

python3.7 -m pip install --upgrade pip
python3.7 -m pip install cython
python3.7 -m pip install zmq requests pandas IPython portpicker google-auth
python3.7 -m pip install --no-deps google-colab
python3.7 -m pip install nvidia-pyindex
python3.7 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#Install Nemo
cd /content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset --hard e856e9732af79a6ed4bffaa3d709bfa387799587
cp PATH_TO_COLAB_NOTEBOOKS/pytorch/nemo_nlp_model.patch /opt/nemo_nlp_model.patch
git apply /opt/nemo_nlp_model.patch
./reinstall.sh
rm -rf /opt/nemo_nlp_model.patch
python3.7 -m pip uninstall tokenizers -y

#Install Cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

#Install tao whls
python3.7 -m pip install https://files.pythonhosted.org/packages/01/f8/813d0043556caa07623d6e75c820580404caec78a3f2508e8ce4da32e4dd/nvidia_eff_tao_encryption-0.1.6-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python3.7 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-eff==0.6.3
python3.7 -m pip install nvidia-tao==4.0.0
python3.7 -m pip install nvidia-tao-pytorch==4.0.1.dev0

python3.7 -m pip install --ignore-installed --no-deps -r PATH_TO_COLAB_NOTEBOOKS/pytorch/requirements-pip.txt

#Reset some packages to required versions
python3.7 -m pip install pytorch_lightning==1.6.0
python3.7 -m pip install transformers==4.8.2
python3.7 -m pip install tokenizers==0.10.3
python3.7 -m pip install huggingface-hub==0.4.0
python3.7 -m pip install torchmetrics==0.10.3
