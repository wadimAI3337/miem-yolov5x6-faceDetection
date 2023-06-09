# YOLOv5x6 Face Detection

YOLOv8 Face Detection is an open-source model built and trained on PyTorch. 

It is specifically designed for the task of face detection, ensuring accurate and efficient detection of faces in images.

# Features
YOLOv5 Face Detection is designed for:

* Real-time face detection in images with big quality
* Integration into production-ready applications for face detection tasks
* Research and experimentation in computer vision tasks related to face detection

# Data overview

The YOLOv8 model has been trained on a custom-assembled dataset consisting of images with a resolution of 640x640 pixels. A portion of the images was sourced from publicly available datasets, such as Wider Face and FDDB, while the remainder were manually annotated and obtained from various online sources.

The following steps were undertaken to prepare the dataset for training the YOLOv8 model:

* Collection of images with varying resolutions.
* Annotation of the collected images.
* Resizing of images to a uniform resolution of 640x640 pixels.
* Enhancement of image quality using a generative adversarial network (Real-ESRGAN).

# Model usage

* Google Colab: Open [notebook](https://colab.research.google.com/drive/19Sc75J4GkUtoFsBW1jyIF6wkJtYFL20m?usp=sharing) with model usage example at Google Colab
* Command line: Open cmd on your device and follow next steps:
  1. ```pip install -r requirements.txt
        git clone https://github.com/ultralytics/ultralytics
        %cd ultralytics
        pip install -e ultralytics
        yolo task=detect mode=predict model='Yolo8m_best.pt' conf=0.25 source="1.jpg"

# Contacts

[a.romanov@hse.ru](https://www.hse.ru/staff/a.romanov) - project manager

[dicd@miee.ru](https://miet.ru/person/122101) - project manager

vvshamalo@edu.hse.ru - researcher
