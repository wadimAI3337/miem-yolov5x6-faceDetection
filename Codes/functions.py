# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

# Algo 1: https://github.com/timesler/facenet-pytorch
# Algo 2: https://github.com/ageitgey/face_recognition
# Algo 3: https://github.com/1adrianb/face-alignment
# https://github.com/biubug6/Pytorch_Retinaface
# https://github.com/elliottzheng/face-detection
# https://github.com/deepcam-cn/yolov5-face

# pip install ultralytics

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import os
import traceback
import torch
import cv2
import numpy as np
from subprocess import Popen, PIPE
from PIL import Image
from ultralytics import YOLO
import time
import math
import subprocess
import sys
import pickle
import json

sys.path.insert(0, './yolov5')

DATABASE_JOBS_NAME = 'main_jobs'
DATABASE_CONFIG_NAME = 'main_config'
LIMIT_FRAMES = 1 * 60 * 30
MAX_VIDEO_FRAME_SIZE = 640
MAX_IMAGE_FRAME_SIZE = 1280

ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
if os.name != 'nt':
    MODELS_PATH = ROOT_PATH + 'models/'
else:
    MODELS_PATH = 'models/'
sys.path.insert(0, ROOT_PATH + 'yolov5')


def connect_to_database():
    import pymysql
    global db, cursor

    # Open database connection
    db = pymysql.connect(
        host='localhost',
        user='mvsep_hide_face',
        passwd='Hi.comdb1',
        db='mvsep_hide_face',
        charset='utf8',
        port=3306
    )
    # prepare a cursor object using cursor() method
    cursor = db.cursor(pymysql.cursors.DictCursor)

    return db, cursor


def gen_logo_image(width, height):
    text = 'hide-face.com'

    new_width = 1920
    new_height = int(round((new_width * height / width)))
    center = (new_width - 300, new_height - 30)

    image_in = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    cv2.putText(
        img=image_in,
        text=text,
        org=(center[0], center[1]),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1,
        color=[1, 1, 1],
        lineType=cv2.LINE_AA,
        thickness=8
    )
    cv2.putText(
        img=image_in,
        text=text,
        org=(center[0], center[1]),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1,
        color=[255, 255, 255],
        lineType=cv2.LINE_AA,
        thickness=2
    )
    if width < new_width:
        interpol = cv2.INTER_LINEAR
    else:
        interpol = cv2.INTER_LANCZOS4
    image_in = cv2.resize(image_in, (width, height), interpolation=interpol)
    return image_in


def get_pipe(out_path, save_video_quality, fps):
    crf = 18
    preset = 'fast'
    if save_video_quality > 0:
        crf = save_video_quality
        if save_video_quality == 16:
            preset = 'slow'
        else:
            preset = 'medium'
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(fps), '-i', '-', '-vcodec',
               'h264',
               '-crf', str(crf), '-preset', str(preset), '-r', str(fps), out_path], stdin=PIPE)
    return p


def copy_audio(in_file, out_file, log):
    log += 'Out file path: {}\n'.format(out_file)
    out_file = os.path.abspath(out_file)
    log += 'Out file path canonical: {}\n'.format(out_file)

    # ffprobe -i INPUT -show_streams -select_streams a -loglevel error
    ret = subprocess.run(
        ['ffprobe', '-i', in_file, '-show_streams', '-select_streams', 'a', '-loglevel', 'error'],
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True
    )
    # print(ret.stdout)
    if '[STREAM]' in ret.stdout:
        log += 'Audio found in input video\n'
        in_audio = out_file + '.aac'
        # print(in_audio)
        ret = subprocess.run(
            ['ffmpeg', '-y', '-i', in_file, '-vn', '-acodec', 'copy', in_audio],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True
        )
        if os.path.isfile(in_audio):
            # ffmpeg -i video.mp4 -i audio.m4a -c copy -map 0:v -map 1:a output.mp4
            new_video_file = out_file + '.mp4'
            ret = subprocess.run(
                ['ffmpeg', '-y', '-i', out_file, '-i', in_audio, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0',
                 '-map', '1:a:0', '-bsf:a', 'aac_adtstoasc', new_video_file],
                stdout=PIPE,
                stderr=PIPE,
                universal_newlines=True
            )
            print(ret.stdout)
            print(ret.stderr)
            if 'failed' in ret.stderr:
                log += 'Fail adding audio. Check problems below:\n' + str(ret.stderr) + '\n' + str(ret.stdout)
                os.remove(new_video_file)
            elif os.path.isfile(new_video_file):
                os.remove(out_file)
                os.rename(new_video_file, out_file)
                log += 'Audio added to final video\n'
            else:
                log += 'Problem with merging video and audio without re-encode\n' + str(ret.stderr) + '\n' + str(
                    ret.stdout)
            os.remove(in_audio)
        else:
            log += 'Problem with audio extraction\n'
    else:
        log += 'No audio found\n'
    return log


def rescale_frame(frame, max_frame_size):
    image = frame.copy()
    scale = 1
    if frame.shape[0] > frame.shape[1]:
        if frame.shape[0] > max_frame_size:
            scale = frame.shape[0] / max_frame_size
    if frame.shape[0] <= frame.shape[1]:
        if frame.shape[1] > max_frame_size:
            scale = frame.shape[1] / max_frame_size

    if scale == 1:
        return image, scale
    newx = int(round(frame.shape[1] / scale))
    newy = int(round(frame.shape[0] / scale))
    image = cv2.resize(image, (newx, newy), interpolation=cv2.INTER_LINEAR)
    return image, scale


def proc_single_frame_mtcnn(
        mtcnn,
        frame,
        log,
        hide_type,
        logo,
        additional_border,
):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    # Detect faces
    image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = int(round(scale * b[0])), int(round(scale * b[1])), int(round(scale * b[2])), int(
                round(scale * b[3]))
            if additional_border > 0:
                x1 = max(x1 - additional_border, 0)
                y1 = max(y1 - additional_border, 0)
                x2 = min(x2 + additional_border, frame.shape[1])
                y2 = min(y2 + additional_border, frame.shape[0])
            if hide_type == 0:
                mask = np.zeros_like(frame)
                mask[y1:y2 + 1, x1:x2 + 1] = 1
                frame[mask != 0] = blured_frame[mask != 0]
            elif hide_type == 1:
                frame[y1:y2 + 1, x1:x2 + 1] = 0
            elif hide_type == 2:
                frame[y1:y2 + 1, x1:x2 + 1, 0] = frame[y1:y2 + 1, x1:x2 + 1, 0].mean()
                frame[y1:y2 + 1, x1:x2 + 1, 1] = frame[y1:y2 + 1, x1:x2 + 1, 1].mean()
                frame[y1:y2 + 1, x1:x2 + 1, 2] = frame[y1:y2 + 1, x1:x2 + 1, 2].mean()

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, log, boxes


def update_percent_complete(id, current_frames, max_frames):
    global db, cursor

    try:
        if current_frames % 10 == 1:
            complete_fraction = 0.0
            if max_frames > 0:
                complete_fraction = 100 * current_frames / max_frames
            sql = "UPDATE {} SET complete_fraction = '{}' WHERE id = '{}'".format(DATABASE_JOBS_NAME, complete_fraction,
                                                                                  id)
            cursor.execute(sql)
            db.commit()
    except Exception as e:
        print('Update DB error during percent of complete execution: {}'.format(str(e)))


# algorithm 1

def hide_face_for_single_file_mtcnn(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
):
    from facenet_pytorch import MTCNN
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_name = None
    log = 'Running algorithm "facenet_pytorch (MTCNN)"\n'
    log += 'Running on device: {}\n'.format(device)
    log += 'Proc file: {}\n'.format(in_file)
    mtcnn = MTCNN(keep_all=True, device=device)

    media_params = dict()
    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        # image = face_recognition.load_image_file(in_file)
        # face_locations = face_recognition.face_locations(image)
        # print(face_locations)
        frame = cv2.imread(in_file)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])
        frame, log, boxes = proc_single_frame_mtcnn(
            mtcnn,
            frame,
            log,
            hide_type,
            logo,
            additional_border,
        )
        if boxes is not None:
            log += 'Found boxes: {}\n'.format(len(boxes))
        else:
            log += 'Not found boxes\n'
        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame, log, boxes = proc_single_frame_mtcnn(
                    mtcnn,
                    frame,
                    log,
                    hide_type,
                    logo,
                    additional_border,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                if boxes is not None:
                    total_boxes += len(boxes)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found boxes: {} Avg boxes per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def proc_single_frame_face_rec(
        boxes,
        frame,
        log,
        hide_type,
        logo,
        scale,
        additional_border,
):
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    # Detect faces
    for b in boxes:
        # (top, right, bottom, left) order
        y1, x2, y2, x1 = int(round(scale * b[0])), int(round(scale * b[1])), int(round(scale * b[2])), int(
            round(scale * b[3]))
        if additional_border > 0:
            x1 = max(x1 - additional_border, 0)
            y1 = max(y1 - additional_border, 0)
            x2 = min(x2 + additional_border, frame.shape[1])
            y2 = min(y2 + additional_border, frame.shape[0])
        if hide_type == 0:
            mask = np.zeros_like(frame)
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[y1:y2 + 1, x1:x2 + 1] = 0
        elif hide_type == 2:
            frame[y1:y2 + 1, x1:x2 + 1, 0] = frame[y1:y2 + 1, x1:x2 + 1, 0].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 1] = frame[y1:y2 + 1, x1:x2 + 1, 1].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 2] = frame[y1:y2 + 1, x1:x2 + 1, 2].mean()

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def proc_single_frame_face_rec_landmarks(
        landmarks,
        frame,
        log,
        hide_type,
        logo,
        scale,
        additional_border,
):
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    for landmark in landmarks:
        points = []
        for el in landmark:
            points += landmark[el]
        scaled_points = np.round((scale * np.array(points))).astype(np.int32)
        convexHull = cv2.convexHull(scaled_points)
        mask = np.zeros_like(frame)
        cv2.drawContours(mask, [convexHull], -1, (255, 255, 255), -1)
        if additional_border > 0:
            mask = cv2.dilate(mask, np.ones((additional_border + 2, additional_border + 2)))
        if hide_type == 0:
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[mask != 0] = 0
        elif hide_type == 2:
            frame[:, :, 0][mask[:, :, 0] != 0] = frame[:, :, 0][mask[:, :, 0] != 0].mean()
            frame[:, :, 1][mask[:, :, 1] != 0] = frame[:, :, 1][mask[:, :, 1] != 0].mean()
            frame[:, :, 2][mask[:, :, 2] != 0] = frame[:, :, 2][mask[:, :, 2] != 0].mean()

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log


# algorithm 2

def hide_face_for_single_file_face_rec(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
        use_landmarks=False
):
    import face_recognition
    start_time = time.time()
    out_name = None
    log = 'Running algorithm "face_recognition"\n'
    log += 'Proc file: {}\n'.format(in_file)
    log += 'Use landmarks: {}\n'.format(use_landmarks)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        # image = face_recognition.load_image_file(in_file)
        # face_locations = face_recognition.face_locations(image)
        # print(face_locations)
        image = face_recognition.load_image_file(in_file)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])
        if not use_landmarks:
            face_locations = face_recognition.face_locations(image)
            log += 'Found boxes: {}\n'.format(len(face_locations))
            if len(face_locations) > 0:
                frame, log = proc_single_frame_face_rec(
                    face_locations,
                    frame,
                    log,
                    hide_type,
                    logo,
                    scale,
                    additional_border,
                )
            else:
                if logo is not None:
                    frame[logo != 0] = logo[logo != 0]
        else:
            face_landmarks_list = face_recognition.face_landmarks(image)
            log += 'Found landmarks: {}\n'.format(len(face_landmarks_list))
            if len(face_landmarks_list) > 0:
                frame, log = proc_single_frame_face_rec_landmarks(
                    face_landmarks_list,
                    frame,
                    log,
                    hide_type,
                    logo,
                    scale,
                    additional_border,
                )
            else:
                if logo is not None:
                    frame[logo != 0] = logo[logo != 0]
        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                image, scale = rescale_frame(image, MAX_VIDEO_FRAME_SIZE)
                if not use_landmarks:
                    face_locations = face_recognition.face_locations(image)
                    # log += 'Found boxes: {}\n'.format(len(face_locations))
                    if len(face_locations) > 0:
                        frame, log = proc_single_frame_face_rec(
                            face_locations,
                            frame,
                            log,
                            hide_type,
                            logo,
                            scale,
                            additional_border,
                        )
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                    total_boxes += len(face_locations)
                else:
                    face_landmarks_list = face_recognition.face_landmarks(image)
                    # log += 'Found landmarks: {}\n'.format(len(face_landmarks_list))
                    if len(face_landmarks_list) > 0:
                        frame, log = proc_single_frame_face_rec_landmarks(
                            face_landmarks_list,
                            frame,
                            log,
                            hide_type,
                            logo,
                            scale,
                            additional_border,
                        )
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                    total_boxes += len(face_landmarks_list)

                frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)

                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def proc_single_frame_face_alignment(
        landmarks,
        frame,
        log,
        hide_type,
        logo,
        scale,
        additional_border=0,
):
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    for points in landmarks:
        points = np.round((scale * points)).astype(np.int32)
        convexHull = cv2.convexHull(points)
        mask = np.zeros_like(frame)
        cv2.drawContours(mask, [convexHull], -1, (255, 255, 255), -1)
        if additional_border > 0:
            mask = cv2.dilate(mask, np.ones((additional_border + 2, additional_border + 2)))
        if hide_type == 0:
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[mask != 0] = 0
        elif hide_type == 2:
            frame[:, :, 0][mask[:, :, 0] != 0] = frame[:, :, 0][mask[:, :, 0] != 0].mean()
            frame[:, :, 1][mask[:, :, 1] != 0] = frame[:, :, 1][mask[:, :, 1] != 0].mean()
            frame[:, :, 2][mask[:, :, 2] != 0] = frame[:, :, 2][mask[:, :, 2] != 0].mean()

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log


# algorithm 3

def hide_face_for_single_file_face_alignement(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
        face_detector='sfd'
):
    import face_alignment
    from skimage import io

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        face_detector=face_detector,
    )
    start_time = time.time()
    out_name = None
    log = 'Running algorithm "face_alignment [+{}]"\n'.format(face_detector)
    log += 'Proc file: {}\n'.format(in_file)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        # image = face_recognition.load_image_file(in_file)
        # face_locations = face_recognition.face_locations(image)
        # print(face_locations)
        image = cv2.imread(in_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])

        preds = fa.get_landmarks(image)
        if preds:
            log += 'Found boxes: {}\n'.format(len(preds))
            if len(preds) > 0:
                frame, log = proc_single_frame_face_alignment(
                    preds,
                    frame,
                    log,
                    hide_type,
                    logo,
                    scale,
                    additional_border,
                )
            else:
                if logo is not None:
                    frame[logo != 0] = logo[logo != 0]
        else:
            if logo is not None:
                frame[logo != 0] = logo[logo != 0]
            log += 'No boxes found\n'

        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
                # image = frame.copy()
                # scale = 1
                preds = fa.get_landmarks(image)
                # log += 'Found boxes: {}\n'.format(len(face_locations))
                if preds:
                    if len(preds) > 0:
                        frame, log = proc_single_frame_face_alignment(
                            preds,
                            frame,
                            log,
                            hide_type,
                            logo,
                            scale,
                            additional_border,
                        )
                        total_boxes += len(preds)
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                else:
                    if logo is not None:
                        frame[logo != 0] = logo[logo != 0]

                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def proc_single_frame_face_retinaface(
        faces,
        frame,
        log,
        hide_type,
        logo,
        scale,
        additional_border,
):
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    # Detect faces
    total_used_boxes = 0
    for box, landmarks, score in faces:
        if score < 0.5:
            continue
        # (top, right, bottom, left) order
        x1, y1, x2, y2 = int(round(scale * box[0])), int(round(scale * box[1])), int(round(scale * box[2])), int(
            round(scale * box[3]))
        if additional_border > 0:
            x1 = max(x1 - additional_border, 0)
            y1 = max(y1 - additional_border, 0)
            x2 = min(x2 + additional_border, frame.shape[1])
            y2 = min(y2 + additional_border, frame.shape[0])
        if hide_type == 0:
            mask = np.zeros_like(frame)
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[y1:y2 + 1, x1:x2 + 1] = 0
        elif hide_type == 2:
            frame[y1:y2 + 1, x1:x2 + 1, 0] = frame[y1:y2 + 1, x1:x2 + 1, 0].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 1] = frame[y1:y2 + 1, x1:x2 + 1, 1].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 2] = frame[y1:y2 + 1, x1:x2 + 1, 2].mean()

        if 0:
            for i in range(len(landmarks)):
                x1 = int(landmarks[i][0])
                y1 = int(landmarks[i][1])
                frame[y1 - 5:y1 + 5, x1 - 5:x1 + 5, :] = 0
        total_used_boxes += 1

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log, total_used_boxes


# algorithm 4

def hide_face_for_single_file_retinaface(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
        network='mobilenet'
):
    from face_detection import RetinaFace

    if network == 'mobilenet':
        detector = RetinaFace()
    else:
        detector = RetinaFace(
            model_path=MODELS_PATH + "Resnet50_Final.pth",
            network="resnet50",
        )
    start_time = time.time()
    out_name = None
    log = 'Running algorithm "face_detector [RetinaFace]"\n'
    log += 'Network: {}\n'.format(network)
    log += 'Proc file: {}\n'.format(in_file)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        image = cv2.imread(in_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])

        preds = detector(image)
        if preds:
            log += 'Found boxes: {}\n'.format(len(preds))
            if len(preds) > 0:
                frame, log, total_used_boxes = proc_single_frame_face_retinaface(
                    preds,
                    frame,
                    log,
                    hide_type,
                    logo,
                    scale,
                    additional_border,
                )
                log += 'Used boxes: {}\n'.format(total_used_boxes)
            else:
                if logo is not None:
                    frame[logo != 0] = logo[logo != 0]
        else:
            if logo is not None:
                frame[logo != 0] = logo[logo != 0]
            log += 'No boxes found\n'

        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = frame.copy()
                image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
                preds = detector(image)
                # log += 'Found boxes: {}\n'.format(len(face_locations))
                if preds:
                    if len(preds) > 0:
                        frame, log, total_used_boxes = proc_single_frame_face_retinaface(
                            preds,
                            frame,
                            log,
                            hide_type,
                            logo,
                            scale,
                            additional_border,
                        )
                        total_boxes += total_used_boxes
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                else:
                    if logo is not None:
                        frame[logo != 0] = logo[logo != 0]

                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def dynamic_resize(shape, stride=64):
    max_size = max(shape[0], shape[1])
    if max_size % stride != 0:
        max_size = (int(max_size / stride) + 1) * stride
    return max_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    import torchvision

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[:, 5:15], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def detect(model, device, img0):
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    if imgsz <= 0:  # original size
        imgsz = dynamic_resize(img0.shape)
    imgsz = check_img_size(imgsz, s=64)  # check img_size
    img = letterbox(img0, imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression_face(pred, 0.02, 0.5)[0]
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
    gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
    boxes = []
    landmarks = []
    h, w, c = img0.shape
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred[:, 5:15] = scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
        for j in range(pred.size()[0]):
            xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
            xywh = xywh.data.cpu().numpy()
            conf = pred[j, 4].cpu().numpy()
            landmark = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
            class_num = pred[j, 15].cpu().numpy()
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
            boxes.append([x1, y1, x2, y2, conf])
            landmarks.append(landmark)
    return boxes, landmarks


def proc_single_frame_face_yolo_face(
        faces,
        landmarks,
        frame,
        log,
        hide_type,
        logo,
        scale,
        additional_border,
):
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    # Detect faces
    total_used_boxes = 0
    for j, box in enumerate(faces):
        score = box[4]
        if score < 0.5:
            continue
        # (top, right, bottom, left) order
        x1, y1, x2, y2 = int(round(scale * box[0])), int(round(scale * box[1])), int(round(scale * box[2])), int(
            round(scale * box[3]))
        if additional_border > 0:
            x1 = max(x1 - additional_border, 0)
            y1 = max(y1 - additional_border, 0)
            x2 = min(x2 + additional_border, frame.shape[1])
            y2 = min(y2 + additional_border, frame.shape[0])
        if hide_type == 0:
            mask = np.zeros_like(frame)
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[y1:y2 + 1, x1:x2 + 1] = 0
        elif hide_type == 2:
            frame[y1:y2 + 1, x1:x2 + 1, 0] = frame[y1:y2 + 1, x1:x2 + 1, 0].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 1] = frame[y1:y2 + 1, x1:x2 + 1, 1].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 2] = frame[y1:y2 + 1, x1:x2 + 1, 2].mean()

        if 0:
            landmark = landmarks[j]
            for i in range(0, len(landmark), 2):
                x1 = int(scale * frame.shape[1] * landmark[i])
                y1 = int(scale * frame.shape[0] * landmark[i + 1])
                frame[y1 - 5:y1 + 5, x1 - 5:x1 + 5, :] = 0
        total_used_boxes += 1

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log, total_used_boxes


# algorithm 5 (Yolov5-Face)

def hide_face_for_single_file_yolo_face(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
        network='n'
):
    device = torch.device('cuda:0')
    if network == 'n':
        m = torch.load(MODELS_PATH + "yolov5n-face.pt", map_location=device)
        model = m['model'].float().fuse().eval()
    else:
        model = torch.load(MODELS_PATH + "yolov5l-face.pt", map_location=device)['model'].float().fuse().eval()
    start_time = time.time()
    out_name = None
    log = 'Running algorithm "face_detector [Yolov5-Face]"\n'
    log += 'Network: {}\n'.format(network)
    log += 'Proc file: {}\n'.format(in_file)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        image = cv2.imread(in_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])

        preds, landmarks = detect(model, device, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if preds:
            log += 'Found boxes: {}\n'.format(len(preds))
            if len(preds) > 0:
                frame, log, total_used_boxes = proc_single_frame_face_yolo_face(
                    preds,
                    landmarks,
                    frame,
                    log,
                    hide_type,
                    logo,
                    scale,
                    additional_border,
                )
                log += 'Used boxes: {}\n'.format(total_used_boxes)
        else:
            if logo is not None:
                frame[logo != 0] = logo[logo != 0]
            log += 'No boxes found\n'

        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {} SQ: {}\n'.format(length, width, height, fps,
                                                                             save_video_quality)
            print('Length: {} Width: {} Height: {} FPS: {} SQ: {}\n'.format(length, width, height, fps,
                                                                            save_video_quality))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name
            out_path = os.path.abspath(out_path)

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)
            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = frame.copy()
                image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
                preds, landmarks = detect(model, device, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # log += 'Found boxes: {}\n'.format(len(face_locations))
                if preds:
                    if len(preds) > 0:
                        frame, log, total_used_boxes = proc_single_frame_face_yolo_face(
                            preds,
                            landmarks,
                            frame,
                            log,
                            hide_type,
                            logo,
                            scale,
                            additional_border,
                        )
                        total_boxes += total_used_boxes
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                else:
                    if logo is not None:
                        frame[logo != 0] = logo[logo != 0]

                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))

            if not os.path.isfile(out_path):
                log += 'File was not created by FFMPEG! Check problem: {}\n'.format(out_path)

            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def get_overall_preds(preds1, preds2, preds3, preds4, scale, image_width, image_height):
    from ensemble_boxes import weighted_boxes_fusion

    boxes_list = []
    scores_list = []
    labels_list = []

    # Algo 1 (Face landmarks)
    boxes = []
    scores = []
    labels = []
    if preds1 is not None:
        for points in preds1:
            # Get box
            x1, x2, y1, y2 = points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()
            x1 = scale * x1 / image_width
            x2 = scale * x2 / image_width
            y1 = scale * y1 / image_height
            y2 = scale * y2 / image_height
            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)
            labels.append(0)

    boxes_list.append(boxes.copy())
    scores_list.append(scores.copy())
    labels_list.append(labels.copy())

    # Algo 2 (Yolo)
    boxes = []
    scores = []
    labels = []
    for points in preds2:
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
        x1 = scale * x1 / image_width
        x2 = scale * x2 / image_width
        y1 = scale * y1 / image_height
        y2 = scale * y2 / image_height
        boxes.append([x1, y1, x2, y2])
        scores.append(points[4])
        labels.append(0)
    boxes_list.append(boxes.copy())
    scores_list.append(scores.copy())
    labels_list.append(labels.copy())

    # Algo 3 (MTCNN)
    boxes = []
    scores = []
    labels = []
    if preds3[0] is not None:
        for i in range(len(preds3[1])):
            points, prob = preds3[0][i], preds3[1][i]
            x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
            x1 = scale * x1 / image_width
            x2 = scale * x2 / image_width
            y1 = scale * y1 / image_height
            y2 = scale * y2 / image_height
            boxes.append([x1, y1, x2, y2])
            scores.append(prob)
            labels.append(0)
    boxes_list.append(boxes.copy())
    scores_list.append(scores.copy())
    labels_list.append(labels.copy())

    # Algo 4 (Retina)
    boxes = []
    scores = []
    labels = []
    for points, _, prob in preds4:
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
        x1 = scale * x1 / image_width
        x2 = scale * x2 / image_width
        y1 = scale * y1 / image_height
        y2 = scale * y2 / image_height
        boxes.append([x1, y1, x2, y2])
        scores.append(prob)
        labels.append(0)
    boxes_list.append(boxes.copy())
    scores_list.append(scores.copy())
    labels_list.append(labels.copy())

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        [1, 1, 1, 1],
        iou_thr=0.4,
        conf_type='box_and_model_avg',
    )

    # Get coords back
    boxes = np.array(boxes)
    scores = np.array(scores)
    boxes[:, 0] *= image_width
    boxes[:, 1] *= image_height
    boxes[:, 2] *= image_width
    boxes[:, 3] *= image_height
    boxes = boxes.astype(np.int32)

    return boxes, scores


def proc_single_frame_face_four_algos(
        faces,
        scores,
        frame,
        log,
        hide_type,
        logo,
        additional_border,
):
    thr = 0.1
    if hide_type == 0:
        ksize = (frame.shape[0] // 16, frame.shape[1] // 16)
        blured_frame = cv2.blur(frame.copy(), ksize)

    # Detect faces
    total_used_boxes = 0
    for j, box in enumerate(faces):
        score = scores[j]
        if score < thr:
            continue
        # (top, right, bottom, left) order
        x1, y1, x2, y2 = box
        if additional_border > 0:
            x1 = max(x1 - additional_border, 0)
            y1 = max(y1 - additional_border, 0)
            x2 = min(x2 + additional_border, frame.shape[1])
            y2 = min(y2 + additional_border, frame.shape[0])

        if hide_type == 0:
            mask = np.zeros_like(frame)
            mask[y1:y2 + 1, x1:x2 + 1] = 1
            frame[mask != 0] = blured_frame[mask != 0]
        elif hide_type == 1:
            frame[y1:y2 + 1, x1:x2 + 1] = 0
        elif hide_type == 2:
            frame[y1:y2 + 1, x1:x2 + 1, 0] = frame[y1:y2 + 1, x1:x2 + 1, 0].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 1] = frame[y1:y2 + 1, x1:x2 + 1, 1].mean()
            frame[y1:y2 + 1, x1:x2 + 1, 2] = frame[y1:y2 + 1, x1:x2 + 1, 2].mean()

        if 0:
            landmark = landmarks[j]
            for i in range(0, len(landmark), 2):
                x1 = int(scale * frame.shape[1] * landmark[i])
                y1 = int(scale * frame.shape[0] * landmark[i + 1])
                frame[y1 - 5:y1 + 5, x1 - 5:x1 + 5, :] = 0
        total_used_boxes += 1

    if logo is not None:
        frame[logo != 0] = logo[logo != 0]
    return frame, log, total_used_boxes


# algorithm all at once (Face Align + SFD -> Yolov5-Face (Large) -> MTCNN -> RetinaFace + ResNet50)

def hide_face_for_single_file_all_algos(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
):
    import face_alignment
    from facenet_pytorch import MTCNN

    # pip install git+https://github.com/elliottzheng/face-detection.git@master
    from face_detection import RetinaFace

    # Face Align + SFD
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        face_detector='sfd',
    )

    # Yolo model
    device = torch.device('cuda:0')
    model_yolo = torch.load(MODELS_PATH + "yolov5l-face.pt", map_location=device)['model'].float().fuse().eval()

    # MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)

    # RetinaFace
    detector_retina = RetinaFace(
        model_path=MODELS_PATH + "Resnet50_Final.pth",
        network="resnet50",
    )

    start_time = time.time()
    out_name = None
    log = 'Running algorithm "Four algos at once"\n'
    log += 'Proc file: {}\n'.format(in_file)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        image = cv2.imread(in_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])

        preds1 = fa.get_landmarks(image.copy())
        preds2, _ = detect(model_yolo, device, cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR))
        preds3 = mtcnn.detect(image.copy())
        preds4 = detector_retina(image.copy())
        print(preds1)
        print(preds2)
        print(preds3)
        print(preds4)
        preds, scores = get_overall_preds(preds1, preds2, preds3, preds4, scale, frame.shape[1], frame.shape[0])

        if len(scores) > 0:
            log += 'Found boxes: {}\n'.format(len(scores))
            frame, log, total_used_boxes = proc_single_frame_face_four_algos(
                preds,
                scores,
                frame,
                log,
                hide_type,
                logo,
                additional_border,
            )
            log += 'Used boxes: {}\n'.format(total_used_boxes)
        else:
            if logo is not None:
                frame[logo != 0] = logo[logo != 0]
            log += 'No boxes found\n'

        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
                preds1 = fa.get_landmarks(image.copy())
                preds2, _ = detect(model_yolo, device, cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR))
                preds3 = mtcnn.detect(image.copy())
                preds4 = detector_retina(image.copy())
                preds, scores = get_overall_preds(preds1, preds2, preds3, preds4, scale, frame.shape[1], frame.shape[0])
                if len(scores) > 0:
                    frame, log, total_used_boxes = proc_single_frame_face_four_algos(
                        preds,
                        scores,
                        frame,
                        log,
                        hide_type,
                        logo,
                        additional_border,
                    )
                    total_boxes += total_used_boxes
                else:
                    if logo is not None:
                        frame[logo != 0] = logo[logo != 0]

                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


# algorithm 10

def hide_face_for_single_file_face_alignement_yolo(
        id,
        in_file,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        additional_border,
):
    import face_alignment
    from skimage import io
    face_detector = 'sfd'

    # Yolov5
    device = torch.device('cuda:0')
    m = torch.load(MODELS_PATH + "yolov5l-face.pt", map_location=device)
    model = m['model'].float().fuse().eval()

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False,
        face_detector=face_detector,
    )
    start_time = time.time()
    out_name = None
    log = 'Running algorithm "face_alignment [+{}]"\n'.format('Yolov5-face')
    log += 'Proc file: {}\n'.format(in_file)

    logo = None
    # Image
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        log += 'Type: image\n'
        # image = face_recognition.load_image_file(in_file)
        # face_locations = face_recognition.face_locations(image)
        # print(face_locations)
        image = cv2.imread(in_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image.copy()
        image, scale = rescale_frame(image, MAX_IMAGE_FRAME_SIZE)
        if is_logo:
            logo = gen_logo_image(frame.shape[1], frame.shape[0])

        preds_yolo, _ = detect(model, device, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        preds = fa.get_landmarks(image, detected_faces=preds_yolo)
        if preds:
            log += 'Found boxes: {}\n'.format(len(preds))
            if len(preds) > 0:
                frame, log = proc_single_frame_face_alignment(preds, frame, log, hide_type, logo, scale,
                                                              additional_border)
            else:
                if logo is not None:
                    frame[logo != 0] = logo[logo != 0]
        else:
            if logo is not None:
                frame[logo != 0] = logo[logo != 0]
            log += 'No boxes found\n'

        out_name = os.path.basename(in_file)
        out_path = output_dir + out_name
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, frame)
    # Video
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        log += 'Type: video [Q: {}]\n'.format(save_video_quality)
        batch_size = 16

        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            log += 'Problem with opening video file!\n'
        else:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log += 'Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps)
            print('Length: {} Width: {} Height: {} FPS: {}\n'.format(length, width, height, fps))

            out_name = os.path.basename(in_file)
            out_path = output_dir + out_name

            if is_logo:
                logo = gen_logo_image(width, height)

            p = get_pipe(out_path, save_video_quality, fps)

            total_boxes = 0
            total_frames = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
                if total_frames > LIMIT_FRAMES:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image, scale = rescale_frame(frame, MAX_VIDEO_FRAME_SIZE)
                # image = frame.copy()
                # scale = 1
                preds_yolo, _ = detect(model, device, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                preds = fa.get_landmarks(image, detected_faces=preds_yolo)
                # log += 'Found boxes: {}\n'.format(len(face_locations))
                if preds:
                    if len(preds) > 0:
                        frame, log = proc_single_frame_face_alignment(preds, frame, log, hide_type, logo, scale,
                                                                      additional_border)
                        total_boxes += len(preds)
                    else:
                        if logo is not None:
                            frame[logo != 0] = logo[logo != 0]
                else:
                    if logo is not None:
                        frame[logo != 0] = logo[logo != 0]

                im = Image.fromarray(frame)
                im.save(p.stdin, 'JPEG', quality=100)
                update_percent_complete(id, total_frames, min(length, LIMIT_FRAMES))
                total_frames += 1

            avg_box_per_frame = 0
            if total_frames > 0:
                avg_box_per_frame = total_boxes / total_frames
            log += 'Found locations: {} Avg locations per frame: {:.1f}\n'.format(total_boxes, avg_box_per_frame)
            cap.release()
            cv2.destroyAllWindows()
            p.stdin.close()
            p.wait()
            if total_frames > 0:
                log += 'Average FPS: {:.2f}\n'.format(total_frames / (time.time() - start_time))
            log = copy_audio(in_file, out_path, log)
    else:
        log += 'Type: unknown source type\n'
        raise ValueError('Type: unknown source type!')

    log += 'Proc time: {:.2f} sec\n'.format(time.time() - start_time)
    return log, out_name


def update_media_params(in_file, id, type):
    from pymysql import escape_string
    global db, cursor

    media_params = dict()
    if in_file[-4:] == '.jpg' or in_file[-4:] == '.png':
        frame = cv2.imread(in_file)
        media_params['type'] = 'image'
        media_params['extension'] = in_file[-3:]
        media_params['width'] = frame.shape[1]
        media_params['height'] = frame.shape[0]
    elif in_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov']:
        cap = cv2.VideoCapture(in_file)
        if cap.isOpened():
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            media_params['type'] = 'video'
            media_params['extension'] = in_file[-3:]
            media_params['length'] = length
            media_params['width'] = width
            media_params['height'] = height
            media_params['fps'] = fps

    js = escape_string(str(json.dumps(media_params)))
    try:
        if type == 0:
            sql = "UPDATE {} SET media_input_params = '{}' WHERE id = '{}'".format(DATABASE_JOBS_NAME, js, id)
        else:
            sql = "UPDATE {} SET media_output_params = '{}' WHERE id = '{}'".format(DATABASE_JOBS_NAME, js, id)
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print('Media param error: {}'.format(str(e)))


# Algo Yolov5x6
# Algo Yolov8m

import argparse
import os
import platform
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from common import DetectMultiBackend
from dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                     increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from plots import Annotator, colors, save_one_box
from torch_utils import select_device, smart_inference_mode

#
# def blur_faces_v5(image, detections):
#     for *xyxy, conf, cls in reversed(detections):
#         # Extract the face area
#         x1, y1, x2, y2 = map(int, xyxy)
#         face = image[y1:y2, x1:x2]
#
#         # Apply a gaussian blur to this area
#         blurred_face = cv2.GaussianBlur(face, (99, 99), 10)
#
#         # Merge this blurry face to our final image
#         image[y1:y2, x1:x2] = blurred_face
#
#     return image

def blur_faces_v5(image, detections, option=0):
    for *xyxy, conf, cls in reversed(detections):
        # Extract the face area
        x1, y1, x2, y2 = map(int, xyxy)
        face = image[y1:y2, x1:x2]

        if option == 0:
            # Apply a gaussian blur to this area
            blurred_face = cv2.GaussianBlur(face, (99, 99), 10)
        elif option == 1:
            # Create a black rectangle with the same size as the face
            blurred_face = np.zeros_like(face)
        elif option == 2:
            # Calculate the average color of the face
            avg_color = np.mean(face, axis=(0, 1))
            # Create a rectangle filled with the average color
            blurred_face = np.full_like(face, avg_color, dtype=np.uint8)

        # Merge this blurry face to our final image
        image[y1:y2, x1:x2] = blurred_face

    return image

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'result',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        log=None,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    out_name = None
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                im0 = blur_faces_v5(im0, det, option=0)
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        out_name = im0
        # Print time (inference-only)
        detections = f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
        LOGGER.info(detections)
        log += detections
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    info = f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t
    LOGGER.info(info)
    log += info
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        res = f"Results saved to {colorstr('bold', save_dir)}{s}"
        LOGGER.info(res)
        log += res
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    return log, out_name


def blur_faces(image, results, option=0):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            b = [int(coord) for coord in b]  # Convert box coordinates to integers
            x1, y1, x2, y2 = b
            face = image[y1:y2, x1:x2]  # Extract face from the image

            if option == 0:
                blurred_face = cv2.GaussianBlur(face, (99, 99), 30)  # Apply blur
            elif option == 1:
                blurred_face = np.zeros_like(face)  # Create black rectangle
            elif option == 2:
                avg_color = np.mean(face, axis=(0, 1))  # Calculate average color
                blurred_face = np.full_like(face, avg_color, dtype=np.uint8)  # Create rectangle with average color

            image[y1:y2, x1:x2] = blurred_face  # Replace original face with modified face
    return image


def blur_faces_in_media(yolo, file_path, output_dir, log):
    os.makedirs(output_dir, exist_ok=True)
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    out_name = None
    if ext in ['.jpg', '.jpeg', '.png']:  # If the file is an image
        image = cv2.imread(file_path)
        results = yolo(image, show_labels=False, line_width=-1, show_conf=False)

        result = results[0]
        boxes = result.boxes
        orig_shape = result.orig_shape
        speed = result.speed

        if boxes is not None and len(boxes) > 0:
            face_info = "1 Face"
            output_string = f"0: {orig_shape[1]}x{orig_shape[0]} {face_info}, {speed['inference']}ms"
        else:
            output_string = f"0: {orig_shape[1]}x{orig_shape[0]} (no detections), {speed['inference']}ms"
        speed_string = f"Speed: {speed['preprocess']}ms preprocess, {speed['inference']}ms inference, {speed['postprocess']}ms postprocess per image at shape (1, 3, {orig_shape[1]}, {orig_shape[0]})"
        log += output_string
        log += speed_string

        blurred_image = blur_faces(image, results, option=2)
        cv2.imwrite(f'{output_dir}/blurred_' + os.path.basename(file_path), blurred_image)
        out_name = f'{output_dir}/blurred_' + os.path.basename(file_path)
    elif ext in ['.avi', '.mp4']:  # If the file is a video
        cap = cv2.VideoCapture(file_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_dir}/blurred_' + os.path.basename(file_path), fourcc, 20.0,
                              (frame_width, frame_height))
        out_name = f'{output_dir}/blurred_' + os.path.basename(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, show_labels=False, line_width=-1, show_conf=False)

            result = results[0]
            boxes = result.boxes
            orig_shape = result.orig_shape
            speed = result.speed

            if boxes is not None and len(boxes) > 0:
                face_info = "1 Face"
                output_string = f"0: {orig_shape[1]}x{orig_shape[0]} {face_info}, {speed['inference']}ms"
            else:
                output_string = f"0: {orig_shape[1]}x{orig_shape[0]} (no detections), {speed['inference']}ms"
            speed_string = f"Speed: {speed['preprocess']}ms preprocess, {speed['inference']}ms inference, {speed['postprocess']}ms postprocess per image at shape (1, 3, {orig_shape[1]}, {orig_shape[0]})"
            log += output_string
            log += speed_string

            blurred_frame = blur_faces(frame, results)
            out.write(blurred_frame)

        cap.release()
        out.release()

    else:
        print("Unsupported file format")

    return log, out_name


def two_yolo(id, f, output_dir, hide_type, is_logo, save_video_quality, additional_border, network):
    # git clone https://github.com/ultralytics/yolov5
    if network == "Yolov5x6":
        log = 'Running algorithm "[{}]"\n'.format('Yolov5x6-face')
        log += 'Proc file: {}\n'.format(id)
        path = MODELS_PATH + "Yolov5x6-face.pt"
        log, out_name = run(weights=path, source=f, project=output_dir, log=log)
    elif network == "Yolov8m":
        from ultralytics import YOLO
        log = 'Running algorithm "[{}]"\n'.format('Yolov8m-face')
        log += 'Proc file: {}\n'.format(id)
        path = MODELS_PATH + "Yolov8m-face.pt"
        yolo = YOLO(path)
        log, out_name = blur_faces_in_media(yolo=yolo, file_path=f, output_dir=output_dir, log=log)
    return log, out_name


def hide_faces(
        ids_list,
        file_list,
        output_dir,
        hide_type,
        is_logo,
        save_video_quality,
        algorithm_type,
        additional_border,
):
    out_files = []
    logs = []
    for i, f in enumerate(file_list):
        try:
            id = ids_list[i]
            if not os.path.isfile(f):
                log = 'File doesnt exists: {}\n'.format(f)
                out_file = None
            else:
                if algorithm_type == 0:
                    log, out_file = hide_face_for_single_file_mtcnn(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                    )
                elif algorithm_type == 1:
                    log, out_file = hide_face_for_single_file_face_rec(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        use_landmarks=False,
                    )
                elif algorithm_type == 2:
                    log, out_file = hide_face_for_single_file_face_rec(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        use_landmarks=True,
                    )
                elif algorithm_type == 3:
                    log, out_file = hide_face_for_single_file_face_alignement(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        face_detector='sfd',
                    )
                elif algorithm_type == 4:
                    log, out_file = hide_face_for_single_file_face_alignement(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        face_detector='blazeface',
                    )
                elif algorithm_type == 5:
                    log, out_file = hide_face_for_single_file_retinaface(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network='mobilenet',
                    )
                elif algorithm_type == 6:
                    log, out_file = hide_face_for_single_file_retinaface(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network='resnet50',
                    )
                elif algorithm_type == 7:
                    log, out_file = hide_face_for_single_file_yolo_face(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network='n',
                    )
                elif algorithm_type == 8:
                    log, out_file = hide_face_for_single_file_yolo_face(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network='l',
                    )
                elif algorithm_type == 9:
                    log, out_file = hide_face_for_single_file_all_algos(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                    )
                elif algorithm_type == 10:
                    log, out_file = hide_face_for_single_file_face_alignement_yolo(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                    )
                elif algorithm_type == 11:
                    log, out_file = two_yolo(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network="Yolov5x6",
                    )
                elif algorithm_type == 12:
                    log, out_file = two_yolo(
                        id,
                        f,
                        output_dir,
                        hide_type,
                        is_logo,
                        save_video_quality,
                        additional_border,
                        network="Yolov8m",
                    )
        except Exception as e:
            out_file = None
            log = traceback.format_exc()
        print(log)
        out_files.append(out_file)
        logs.append(log)
    return out_files, logs
