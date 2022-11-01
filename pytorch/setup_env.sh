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
python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

cd /content
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
git reset --hard e856e9732af79a6ed4bffaa3d709bfa387799587
cp /content/drive/MyDrive/ColabNotebooks/pytorch/nemo_nlp_model.patch /opt/nemo_nlp_model.patch
git apply /opt/nemo_nlp_model.patch
./reinstall.sh
rm -rf /opt/nemo_nlp_model.patch

python3.8 -m pip install /content/drive/MyDrive/pyt/nvidia_tao*.whl 
python3.8 -m pip install /content/drive/MyDrive/pyt/cp38_whls/*.whl
python3.8 -m pip install --ignore-installed --no-deps -r /content/drive/MyDrive/ColabNotebooks/pytorch/requirements-pip.txt
python3.8 -m pip install pytorch_lightning==1.6.0
python3.8 -m pip install transformers==4.8.2
python3.8 -m pip install tokenizers==0.10.3
python3.8 -m pip install huggingface-hub==0.4.0

