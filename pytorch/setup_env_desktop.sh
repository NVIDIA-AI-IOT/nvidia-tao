#!/bin/sh

sudo apt -qq install -y sox

python3.7 -m pip install --upgrade pip
python3.7 -m pip install cython
python3.7 -m pip install nvidia-pyindex
python3.7 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

mkdir content 
cd content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset --hard e856e9732af79a6ed4bffaa3d709bfa387799587
sudo cp PATH_TO_COLAB_NOTEBOOKS/pytorch/nemo_nlp_model.patch /opt/nemo_nlp_model.patch
git apply /opt/nemo_nlp_model.patch
./reinstall.sh
sudo rm -rf /opt/nemo_nlp_model.patch

# Install Cmake
sudo mkdir -p /tmp_dir_cmake && sudo chmod -R 777 /tmp_dir_cmake
cd /tmp_dir_cmake
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
sudo ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

#Install KenLM
sudo apt-get install libbz2-dev liblzma-dev -y
sudo apt install build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev -y
cd PATH_TO_COLAB_NOTEBOOKS
sudo rm -rf /kenlm
sudo tar -xzf kenlm.tar.gz -C /
sudo mkdir -p /kenlm/build
cd /kenlm/build
sudo cmake ..
sudo make -j2
sudo chmod a+rx /kenlm

cd PATH_TO_COLAB_NOTEBOOKS
sudo apt install libeigen3-dev -y
# Install KenLM python and pip requirements
rm -rf kenlm-master
python3.7 -m pip install master.zip

#Install tao whls
python3.7 -m pip install https://files.pythonhosted.org/packages/0e/32/6761e35c533d854ce7e5e51908bac3ae1488d94c110949e467345b28334c/nvidia_eff_tao_encryption-0.1.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python3.7 -m pip install nvidia-eff==0.6.3
python3.7 -m pip install nvidia-tao==4.0.0
python3.7 -m pip install nvidia-tao-pytorch==4.0.1.dev0
python3.7 -m pip install --ignore-installed --no-deps -r PATH_TO_COLAB_NOTEBOOKS/pytorch/requirements-pip.txt

#Reset some packages to required versions
python3.7 -m pip install pytorch_lightning==1.6.0
python3.7 -m pip install transformers==4.8.2
python3.7 -m pip install tokenizers==0.10.3
python3.7 -m pip install huggingface-hub==0.4.0
python3.7 -m pip install torchmetrics==0.10.3
