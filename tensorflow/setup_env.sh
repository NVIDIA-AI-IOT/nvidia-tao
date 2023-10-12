#!/bin/sh

# Install Cuda 12
export DEBIAN_FRONTEND=noninteractive
echo 'keyboard-configuration keyboard-configuration/layout select USA' | sudo debconf-set-selections
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-0 -y

# Install Python 3.8
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.8 -y
apt install python3-pip -y
apt-get install python3.8-distutils
apt-get install python3.8-dev

# Set Python 3.8 as the default version
rm /usr/bin/python
ln -sf /usr/bin/python3.8 /usr/bin/python3
ln -sf /usr/bin/python3.8 /usr/local/bin/python

python3.8 -m pip install --upgrade pip
python3.8 -m pip install nvidia-pyindex

python3.8 -m pip install cython==0.29.35
python3.8 -m pip install pycocotools
python3.8 -m pip install pycocotools-fix==2.0.0.9
python3.8 -m pip install h5py==2.10.0

# Install Tensorflow
python3.8 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-tensorflow==1.15.5+nv23.2

# Install Cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh

# Install dependencies
python3.8 -m pip install nvidia-tao==5.0.0
python3.8 -m pip install --ignore-installed PyYAML -r PATH_TO_COLAB_NOTEBOOKS/tensorflow/requirements-pip.txt -f https://download.pytorch.org/whl/torch_stable.html --extra-index-url https://developer.download.nvidia.com/compute/redist

# Install code related wheels
python3.8 -m pip install /content/drive/MyDrive/ColabNotebooks5.0/colab_notebooks/tensorflow/nvidia_tao_tf1-5.0.0.1-py3-none-any.whl

python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/python/tensorrt-*-cp38-none-linux_x86_64.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/onnx_graphsurgeon/onnx_graphsurgeon*.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/graphsurgeon/graphsurgeon*.whl
python3.8 -m pip install PATH_TO_TRT/TensorRT-TRT_VERSION/uff/uff*.whl

python3.8 -m pip uninstall h5py psutil -y
# Sometimes two versions of package is installed
python3.8 -m pip uninstall h5py -y
python3.8 -m pip uninstall psutil -y
python3.8 -m pip install h5py==2.10.0 psutil
python3.8 -m pip uninstall pycocotools -y
python3.8 -m pip install pycocotools
python3.8 -m pip install -v horovod==0.28.1
python3.8 -m pip install numpy==1.22.2 protobuf==3.20.2

# Add monkey patch lines manually
lines_to_add="
import third_party.keras.mixed_precision
import third_party.keras.tensorflow_backend
third_party.keras.mixed_precision.patch()
third_party.keras.tensorflow_backend.patch()
"
# Check if the __init__.py file exists
init_file="/usr/local/lib/python3.8/dist-packages/keras/__init__.py"
if [ -f "$init_file" ]; then
    # Add the lines to the __init__.py file
    echo "$lines_to_add" | sudo tee -a "$init_file" > /dev/null
    echo "Lines added to $init_file"
else
    echo "Error: $init_file does not exist."
fi