import os
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

RED = (255, 0, 0)


def read_video(filepath):
    frames = []

    assert os.path.isfile(filepath) and "Can't find input filename"
    cap = cv2.VideoCapture(filepath)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    assert frames and "Empty frames list"
    return frames, cap.get(cv2.CAP_PROP_FPS)


def annotate_frames(frames, object_boxes_per_frame, cenroid_ids_per_frame):
    new_frames = []

    for i in range(len(frames)):
        new_frame = write_on_frame(frames[i], object_boxes_per_frame[i], cenroid_ids_per_frame[i])
        new_frames.append(new_frame)

    return new_frames


def write_on_frame(frame, boxes, centroid_ids):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    for box in boxes:
        rect = [box['x1'], box['y1'], box['x2'], box['y2']]
        draw.rectangle(xy=rect, outline=RED)

    font = ImageFont.truetype("Montserrat-Bold.otf", 30)
    if type(centroid_ids) == OrderedDict:
        for (objectID, centroid) in centroid_ids.items():
            draw.text(centroid, text=f"{objectID}", font=font, fill=RED)
    elif type(centroid_ids) == list:
        for (centroid, objectID) in centroid_ids:
            draw.text(centroid, text=f"{objectID}", font=font, fill=RED)

    return np.array(img)


def frames2video(frames, fps, filepath):
    height, width, layers = frames[0].shape

    out = cv2.VideoWriter(filepath,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps,
                          (width, height))

    for i in range(len(frames)):
        out.write(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
    out.release()
