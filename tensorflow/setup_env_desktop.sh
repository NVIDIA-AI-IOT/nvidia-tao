#!/bin/sh

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

# Bash installation
sudo apt-get install libgeos-dev mpich libmpich-dev libhdf5-dev libopenmpi-dev git -y


# Install dependencies
python3.8 -m pip install nvidia-tao==5.0.0
python3.8 -m pip install --ignore-installed PyYAML -r PATH_TO_COLAB_NOTEBOOKS/tensorflow/requirements-pip.txt -f https://download.pytorch.org/whl/torch_stable.html --extra-index-url https://developer.download.nvidia.com/compute/redist

# Build code related wheel
git clone https://github.com/NVIDIA/tao_tensorflow1_backend.git
cd tao_tensorflow1_backend
PYTHONPATH=${PWD} python3.8 release/docker/build_kernels.py
cp -r /tmp/tao_tensorflow1_backend/nvidia_tao_tf1/core/processors/../lib/* /usr/local/lib/python3.8/dist-packages/nvidia_tao_tf1/core/processors/../lib/
PYTHONPATH=${PWD} python3.8 setup.py bdist_wheel
python3.8 -m pip install dist/nvidia_tao_tf1-5.0.0.1-py3-none-any.whl
cd -

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
python3.8 -m pip uninstall numpy protobuf -y
python3.8 -m pip install numpy==1.22.2 protobuf==3.20.2

# Add monkey patch lines manually
sudo apt-get install jq -y
lines_to_add="
import third_party.keras.mixed_precision
import third_party.keras.tensorflow_backend
third_party.keras.mixed_precision.patch()
third_party.keras.tensorflow_backend.patch()
"
active_env_path=$(conda info --json | jq -r '.default_prefix')
init_file="$active_env_path/lib/python3.8/site-packages/keras/__init__.py"

if [ -f "$init_file" ]; then
    # Add the lines to the __init__.py file
    echo "$lines_to_add" | sudo tee -a "$init_file" > /dev/null
    echo "Lines added to $init_file"
else
    echo "Error: $init_file does not exist."
fi

ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$active_env_path/bin/../lib/libstdc++.so.6"