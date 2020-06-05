import torchreid
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import argparse
import json
import os
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from IPython.display import clear_output
# from matplotlib.ticker import NullLocator

from video import read_video, annotate_frames, write_on_frame, frames2video


def extract_features(model, inputs):
    fm = model.featuremaps(inputs)
    fm = model.global_avgpool(fm)
    return fm.view(fm.size(0), -1)

def torch_format(img):
    return transforms.ToTensor()(img)

def resize(image, size=416):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_patch(image, bbox):
    # bbox = bbox.astype('int32')
    # return image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

    return image[int(bbox['y1']):int(bbox['y2']), int(bbox['x1']):int(bbox['x2']), :]

def normalize(image):
    norm_mean = [0.485, 0.456, 0.406] # imagenet mean
    norm_std = [0.229, 0.224, 0.225] # imagenet std
    return transforms.Normalize(mean=norm_mean, std=norm_std)(image)

def patch_to_features(image, bbox, model):
    
    # prepare path for network
    patch = get_patch(image, bbox)
    patch = torch_format(patch)
    patch = resize(patch, [256, 128])
    
    # forward pass to get features
    patch = normalize(patch).cuda()
    patch_features = extract_features(model, patch.unsqueeze(0))
    patch_features = F.normalize(patch_features, p=2, dim=1)
    
    return patch_features

def build_features(image, bboxes, model):
    return torch.cat([patch_to_features(image, bbox, model) for bbox in bboxes], dim=0)

def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat

def update_gallery(gallery, features, threshold=0.006):
    
    dist = euclidean_squared_distance(gallery, features)
    new_features = features[torch.all(dist > threshold, dim=0)]
    
    return torch.cat([gallery, new_features], dim=0)


def parse_args():
    # parse script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--weights", type=str)
    
    args = parser.parse_args()
    return args
    
    
def main(args):
    # Load model
    model = torchreid.models.build_model(
                        name='resnet50',
                        num_classes=1041,
                        loss='softmax',
                        pretrained=False)
    
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    
    # Read video
    video_name = args.video
    frames, fps = read_video(video_name)

    # Read bounding boxes
    object_boxes_per_frame = []
    for frame_number in range(len(frames)):
        filename = f"bboxes/{video_name.split('/')[-1]}_frame_{frame_number}.json"
        with open(filename) as f:
            data = json.load(f)
            bboxes = data['children']
            object_boxes_per_frame.append(bboxes)
            
    # Run object tracking
    centroid_ids_per_frame = []
        
    for frame_number, frame in enumerate(frames):
        if frame_number % 100 == 0:
            print(f'Processing frame number {frame_number}')

        image = frames[frame_number]
        bboxes = object_boxes_per_frame[frame_number]
        features = build_features(image, bboxes, model)
        
        if frame_number == 0: # Form gallery at the first frame
            gallery = features
            indexes = range(len(gallery))
        else:
            if frame_number % 30 == 0: # Update gallery every 30 frames
                gallery = update_gallery(gallery, features)
                
            dist_matrix = euclidean_squared_distance(gallery, features)
            indexes = torch.argmin(dist_matrix, dim=0).cpu().numpy()
        
        centroid_ids_per_frame.append([[((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2), index] for box, index in zip(bboxes, indexes)])
        
    print(f"Processed {len(centroid_ids_per_frame)} frames")
    
    annotated_frames = annotate_frames(frames, object_boxes_per_frame, centroid_ids_per_frame)
    frames2video(annotated_frames, fps=28, filepath="result.avi")
    print("Created output video")

if __name__ == "__main__":
    args = parse_args()
    main(args)