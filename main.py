import cv2
import os
import json
import numpy as np
import argparse

from video import read_video, annotate_frames, frames2video
from centroid_tracker import CentroidTracker
from sort import Sort


# from detector import GroundTruthDetections

def main(args):
    # Read video
    frames, fps = read_video(args.video_path)
    print(f"Read {len(frames)} frames (fps: {fps})")

    # Read bboxes of each frame
    json_files = sorted(os.listdir(args.bbox_path), key=lambda x: int(x.split(".")[0]))
    object_boxes_per_frame = []

    for file in json_files:
        with open(os.path.join(args.bbox_path, file)) as f:
            data = json.load(f)
            bboxes = data['children'].copy()
            object_boxes_per_frame.append(bboxes)
    print(f"Read {len(object_boxes_per_frame)} bbox files")

    # Run object tracking
    centroid_ids_per_frame = []

    if args.method == "centroid":
        ct = CentroidTracker(maxDisappeared=50)

        for ind in range(len(frames)):
            rects = [[obj['x1'], obj['y1'], obj['x2'], obj['y2']] for obj in object_boxes_per_frame[ind]]
            centroid_ids = ct.update(rects)
            centroid_ids_per_frame.append(centroid_ids.copy())

    elif args.method == "kalman":
        tracker = Sort(max_age=50, min_hits=3)

        for ind in range(len(frames)):
            detections = np.array([[obj['x1'], obj['y1'], obj['x2'], obj['y2'], obj['confidence']]
                                   for obj in object_boxes_per_frame[ind]])
            trackers = tracker.update(detections, None)
            centroid_ids = [[((track[0] + track[2]) / 2, (track[1] + track[3]) / 2), int(track[4])]
                            for track in trackers]
            centroid_ids_per_frame.append(centroid_ids)
    else:
        raise NotImplementedError
    print(f"Processed {len(centroid_ids_per_frame)} frames")

    # Create output video
    annotated_frames = annotate_frames(frames, object_boxes_per_frame, centroid_ids_per_frame)
    frames2video(annotated_frames, fps=28, filepath=args.save_path)
    print("Created output video")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object tracking')
    parser.add_argument('--method', default='centroid', help='Which method to use', choices=["centroid", "kalman"])
    parser.add_argument('--video_path', default='MOT16-11-raw.webm', help='Path to the input video')
    parser.add_argument('--bbox_path', default='MOT16-11-bb', help='Path to the bboxes of each video frame')
    parser.add_argument('--save_path', default='MOT16-11-raw-output.avi', help='Path for saving annotated video')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
