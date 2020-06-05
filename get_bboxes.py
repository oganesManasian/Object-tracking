import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import numpy as np
import random
import argparse
import json

import torch.nn.functional as F
import torchvision.transforms as transforms

from YOLOv3.models import *
from YOLOv3.utils.utils import rescale_boxes


def torch_format(img):
    return transforms.ToTensor()(img)


def resize(image, size=416):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def get_bboxes(original_frame, bboxes):
    # Draw bounding boxes and labels of detections
    if bboxes is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(bboxes, 416, original_frame.shape[:2])

    return detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--config", type=str)

    args = parser.parse_args()
    return args


def main(args):
    # creating model on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model ...")
    model = Darknet(args.config, img_size=416)
    model.load_darknet_weights(args.weights)
    model.eval()
    model.to(device)

    # loading frames from video
    video_name = args.video
    print("Loading video ...")
    cap = cv2.VideoCapture(video_name)

    # infer bboxes on video frames
    frame_number = 0
    while cap.isOpened():
        if frame_number % 100 == 0:
            print(f'Processing frame number {frame_number}')

        ret, original_frame = cap.read()

        if not ret:
            break

        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        frame = torch_format(original_frame)
        frame, pad = pad_to_square(frame, 0)
        frame = resize(frame).to(device)

        detections = model(frame.unsqueeze(0))
        detections = non_max_suppression(detections, nms_thres=0.3, conf_thres=0.5)
        detections = detections[0]

        filename = f"bboxes/{video_name.split('/')[-1]}_frame_{frame_number}"

        if detections is None:
            with open(f"{filename}.json", 'w') as f:
                json.dump({'children': []}, f)
            frame_number += 1
            continue

        # Taking only needed predictions
        detections = detections[detections[:, -1] == 0]  # Pedestrian class
        detections = detections[detections[:, -2] > 0.7]
        detections = detections[detections[:, -3] > 0.7]

        # print('save...')
        get_bboxes(original_frame, detections).cpu().numpy()
        # np.save(filename, get_bboxes(original_frame, detections).cpu().numpy())

        # Davit's code
        bboxes = [{'x1': x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item(), 'confidence': cls_conf.item()}
                  for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections]

        # Filter not valid boxes
        bboxes = [box for box in bboxes if abs(box['x2'] - box['x1']) > 0 and abs(box['y2'] - box['y1']) > 0]
        bboxes = [{key: max(float(value), 0) for key, value in box.items()} for box in bboxes]

        with open(f"{filename}.json", 'w') as f:
            json.dump({'children': bboxes}, f)

        frame_number += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
