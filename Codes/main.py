from ultralytics import YOLO
import cv2
yolo_path = 'models/Yolov8m-face.pt'

yolo = YOLO(yolo_path)

# yolo.predict(source="1.jpg")
#
# results = yolo("1.jpg", show_labels=False, line_width=-1, show_conf=False, save=True)
# print(results['boxes'].xyxy)

from ultralytics import YOLO
import cv2
import numpy as np


# yolo_path = 'models/Yolov8m-face.pt'
# yolo = YOLO(yolo_path)
from ultralytics.yolo.utils.plotting import Annotator
# Predict bounding boxes
# results = yolo.predict(source="1.jpg", return_outputs=True)

#
# from ultralytics import YOLO
# import cv2
#
# def blur_faces(image, results):
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
#             # Convert box coordinates to integers
#             b = [int(coord) for coord in b]
#             x1, y1, x2, y2 = b
#             face = image[y1:y2, x1:x2]  # Extract face from the image
#             blurred_face = cv2.GaussianBlur(face, (99, 99), 30)  # Apply blur
#             image[y1:y2, x1:x2] = blurred_face  # Replace original face with blurred face
#     return image
#
# yolo_path = 'models/Yolov8m-face.pt'
# yolo = YOLO(yolo_path)
#
# # Predict bounding boxes
# results = yolo("1.jpg", show_labels=False, line_width=-1, show_conf=False)
#
# # Load the image
# image = cv2.imread('1.jpg')
#
# # Blur faces in the image
# blurred_image = blur_faces(image, results)
#
# # Save the result
# cv2.imwrite('blurred_faces.jpg', blurred_image)

import os
import cv2
from ultralytics import YOLO

# Initialize the YOLO model
yolo_path = 'models/Yolov8m-face.pt'
yolo = YOLO(yolo_path)

# Define the function to blur faces in an image
def blur_faces(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            b = [int(coord) for coord in b]  # Convert box coordinates to integers
            x1, y1, x2, y2 = b
            face = image[y1:y2, x1:x2]  # Extract face from the image
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)  # Apply blur
            image[y1:y2, x1:x2] = blurred_face  # Replace original face with blurred face
    return image

def blur_faces_in_media(file_path):
    os.makedirs('results', exist_ok=True)
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in ['.jpg', '.jpeg', '.png']:  # If the file is an image
        image = cv2.imread(file_path)
        results = yolo(image, show_labels=False, line_width=-1, show_conf=False)
        blurred_image = blur_faces(image, results)

        cv2.imwrite('results/blurred_' + os.path.basename(file_path), blurred_image)
    elif ext in ['.avi', '.mp4']:  # If the file is a video
        cap = cv2.VideoCapture(file_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('results/blurred_' + os.path.basename(file_path), fourcc, 20.0, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, show_labels=False, line_width=-1, show_conf=False)
            blurred_frame = blur_faces(frame, results)
            out.write(blurred_frame)

        cap.release()
        out.release()

    else:
        print("Unsupported file format")

# Example usage:
# blur_faces_in_media('4.avi')
blur_faces_in_media('1.jpg')