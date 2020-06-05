import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms


# def horisontal_flip(images, targets):
#     images = torch.flip(images, [-1])
#     targets[:, 2] = 1 - targets[:, 2]
#     return images, targets


def color_jitter(images, targets):
    data_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    ])
    images = data_transforms(images)
    return images, targets


def horizontal_flip(images, targets):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
    ])
    images = data_transforms(images)
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

