# Person Detection and Tracking On a Video

For object detection we used YOLOv3, which we had already trained on EuroCity Persons (ECP) Dataset. For object tracking, however, three different approaches have been tested, namely centroid tracking algorithm, kalman filter tracker and person re-identification model.

To see the results of each model for the three given videos refer to this [link](https://drive.google.com/open?id=1d-IUrzjbMIyvn1Ah_lp6yvNy_AWPYnEC).
Our model weights can be found here [link](https://drive.google.com/open?id=1LU9k_kdO5ahfW_Xh-ULH6IkU4a69oik-).

We have used Google Colab for running the codes in order to avoid package dependency conflicts. 

## Requirements:
* torch
* numpy
* filterpy
* numba

## Run
### To do person detection on the video run from the project's folder:
```bash
python get_bboxes.py
```
Use mandatory argument
```
--video_path PATH
```
with PATH to define path to the input video.

Use mandatory argument
```
--weights PATH
```
with PATH to define path to the YoloV3 model weights.

Use mandatory argument
```
--config PATH
```
with PATH to define path to the YoloV3 model config.


### To do object tracking on video with centroid or kalman methods run from the project's folder:

```bash
python main.py
```
Use optional argument
```
--method METHOD_NAME
```
with METHOD_NAME to be either "centroid" or "kalman". 

Use mandatory argument
```
--video_path PATH
```
with PATH to define path to the input video.
    
Use mandatory argument
```
--bbox_path PATH
```
with PATH to define path to the bounding boxes of each video frame.

Use mandatory argument
```
--save_path PATH
```
with PATH to define path for saving annotated video.

### To do object tracking on video with reid neural network methods run from the project's folder:

```bash
python get_ids.py
```
Use optional argument
```
--video PATH
```
with PATH to define path to the input video.

Use mandatory argument
```
--weights PATH
```
with PATH to define path to the reid model weights.
