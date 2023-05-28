# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import os
import sys


if os.name != 'nt':
    INPUT_PATH = './test/'
    CACHE_PATH = './test/result/'
else:
    INPUT_PATH = './'
    CACHE_PATH = './result/'

if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from functions import *


if __name__ == '__main__':
    print('CV2 version: {}'.format(cv2.__version__))

    file_list = [
        INPUT_PATH + '1.jpg',
    ]
    output_dir = CACHE_PATH
    # 0 - blur, 1 - black rectangle, 2 - avg rectangle
    hide_type = 0
    # 0 - no logo 1 - logo
    is_logo = True
    # 0 - with videoWriter 1 - with FFMPEG
    save_video_quality = 0
    # 0 - MTCNN,
    # 1 - face_recognition (box),
    # 2 - face_recognition (landmarks)
    # 3 - face_alignment ('sfd'),
    # 4 - face_alignment ('blazeface')
    # 5 - RetinaFace + MobileNet (face-detection)
    # 6 - RetinaFace + ResNet50 (face-detection)
    # 7 - Yolov5-Face (yolov5n, fast)
    # 8 - Yolov5-Face (yolov5l, slow)
    # 9 - all
    # 10 - Face alignment + Yolov5-Face
    # for algorithm_type in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for algorithm_type in [12]:
        print('Algo number: {}'.format(algorithm_type))
        additional_border = 2
        out_files, logs = hide_faces(
            file_list,
            file_list,
            output_dir,
            hide_type,
            is_logo,
            save_video_quality,
            algorithm_type,
            additional_border,
        )
        print('')
