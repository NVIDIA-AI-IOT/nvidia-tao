# Getting started with TAO (Train, Adapt, Optimize) with Google Colab

This repository contains Juptyer notebooks for users to try NVIDIA's TAO product for free on Google Colab before moving onto a scalable cloud instance.
We currently provide support for Conversational AI models based on Pytorch and Computer Vision models based on Tensorflow 1.15 and Pytorch.

## Tutorial ##

All the notebooks available in this repo can be run with Google Colab.
To run a tutorial:
Click the Colab link (see table below).
Follow the cells on the notebook for detailed instructions

| Domain | Title | Github URL  |
| --- | --- | --- |
| Computer Vision | 3D LIDAR Object detection using TAO PointPillars | [Pointpillars](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/cv_notebooks/pointpillars/pointpillars.ipynb) |
| Computer Vision | Action Recognition | [Action Recognition](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/cv_notebooks/action_recognition_net/actionrecognitionnet.ipynb)|
| Computer Vision | Emotion recognition | [Emotion recognition](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/emotionnet/emotionnet.ipynb) |
| Computer Vision | Gesture recognition | [Gesture recognition](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/gesturenet/gesturenet.ipynb) |
| Computer Vision | Heart-rate estimation | [Heart-rate estimation](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/heartratenet/heartratenet.ipynb) |
| Computer Vision | Image Classification | [Image Classification](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/classification/classification.ipynb) |
| Computer Vision | Image Segmentation using MaskRCNN| [MaskRCNN](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/mask_rcnn/maskrcnn.ipynb) |
| Computer Vision | Image Segmentation using Unet| [UNET](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/unet/unet_isbi.ipynb) |
| Computer Vision | License Plate Recogntion | [License Plate recognition](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/lprnet/lprnet.ipynb) |
| Computer Vision | Multi-task Image classification | [Multi-task Image Classification](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/multitask_classification/multitask_classification.ipynb) |
| Computer Vision | Object detection using Deformable DDETR | [Deformable DDETR](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/cv_notebooks/deformable_detr/deformable_detr.ipynb) |
| Computer Vision | Object Detection using DSSD | [DSSD](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/dssd/dssd.ipynb) |
| Computer Vision | Object Detection using EfficientDet | [EfficientDet](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/efficientdet/efficientdet.ipynb) |
| Computer Vision | Object Detection using Retinanet| [Retinanet](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/retinanet/retinanet.ipynb) |
| Computer Vision | Object Detection using SSD | [SSD](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/ssd/ssd.ipynb) |
| Computer Vision | Object Detection using Yolo v3| [Yolo V3](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/yolo_v3/yolo_v3.ipynb) |
| Computer Vision | Object Detection using Yolo v4| [Yolo V4](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/yolo_v4/yolo_v4.ipynb) |
| Computer Vision | Object Detection using Yolo v4 tiny | [Yolo V4 Tiny](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/tensorflow/yolo_v4_tiny/yolo_v4_tiny.ipynb) |
| Computer Vision | Pose Classification | [Pose Classification](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/cv_notebooks/pose_classification_net/poseclassificationnet.ipynb) |
| Conversational AI - ASR | Speech to Text using Jasper | [Speech to Text using Jasper](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/speechtotext/speech-to-text-training.ipynb) |
| Conversational AI - ASR | Speech to Text using Citrinet | [Speech to Text using Citrinet](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/speechtotext_citrinet/speech-to-text-training.ipynb) |
| Conversational AI - ASR | Speech to Text using Conformer | [Speech to Text using Conformer](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/speechtotext_conformer/speech-to-text-training.ipynb) |
| Conversational AI - LM | Language Model using N Gram | [N-Gram](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/ngram_lm/n-gram-training.ipynb) |
| Conversational AI - NLP | Intent Slot Classification | [Intent slot classificatoin](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/intentslotclassification/intent-slot-classification-training.ipynb) |
| Conversational AI - NLP | Question Answering | [Question Answering](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/questionanswering/question-answering-training.ipynb) |
| Conversational AI - NLP | Punctuation and Capitalization | [Punctuation and Capitalization](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/punctuationcapitalization/punctuation-and-capitalization-training.ipynb) |
| Conversational AI - NLP | Text Classification | [Text Classification](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/textclassification/text-classification-training.ipynb) |
| Conversational AI - NLP | Token Classification | [Token Classification](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/tokenclassification/token-classification-training.ipynb) |
| Conversational AI - TTS | Text to Speech | [Text to Speech](https://colab.research.google.com/github/NVIDIA-AI-IOT/nvidia-tao/blob/main/pytorch/convai_notebooks/texttospeech/text-to-speech-finetuning-cvtool.ipynb) |

## Model Restrictions ##

- ConvAI pretrained backbones involving megatron can’t run on Colab/ColabPro instances
  - They need 24 GB of GPU RAM
- The following TAO networks can’t run on Colab until Colab has updated it’s drivers to 515 or above
  - Detectnetv2, BPNET, FasterRCNN, FPNet, Gazenet
