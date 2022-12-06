#!/bin/sh

apt -qq install -y sox

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
python3.8 -m pip install cython
python3.8 -m pip install zmq requests pandas IPython portpicker google-auth
python3.8 -m pip install --no-deps google-colab
python3.8 -m pip install nvidia-pyindex
python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#Install Nemo
cd /content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset --hard e856e9732af79a6ed4bffaa3d709bfa387799587
cp PATH_TO_COLAB_NOTEBOOKS/pytorch/nemo_nlp_model.patch /opt/nemo_nlp_model.patch
git apply /opt/nemo_nlp_model.patch
./reinstall.sh
rm -rf /opt/nemo_nlp_model.patch

#Install Cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

#Install KenLM
cd PATH_TO_COLAB_NOTEBOOKS
rm -rf /kenlm
sudo tar -xzf kenlm.tar.gz -C /
sudo mkdir -p /kenlm/build
cd /kenlm/build
cmake ..
make -j2
sudo chmod a+rx /kenlm

cd PATH_TO_COLAB_NOTEBOOKS
apt install libeigen3-dev
# Install KenLM python and pip requirements
rm -rf kenlm-master
python3.8 -m pip install master.zip


#Install tao whls
python3.8 -m pip install nvidia-eff-tao-encryption==0.1.6
python3.8 -m pip install nvidia-eff==0.6.3
python3.8 -m pip install nvidia-tao==4.0.0
python3.8 -m pip install nvidia-tao-pytorch==b0
python3.8 -m pip install --ignore-installed --no-deps -r PATH_TO_COLAB_NOTEBOOKS/pytorch/requirements-pip.txt

#Reset some packages to required versions
python3.8 -m pip install pytorch_lightning==1.6.0
python3.8 -m pip install transformers==4.8.2
python3.8 -m pip install tokenizers==0.10.3
python3.8 -m pip install huggingface-hub==0.4.0
