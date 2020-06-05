# Object-tracking
Object tracking with centroid tracking algorithm, kalman filter tracker and person re-identification model.

To see the results of each for the three given videos refer to this [linke](https://drive.google.com/open?id=1d-IUrzjbMIyvn1Ah_lp6yvNy_AWPYnEC).


## Requirements:
* filterpy
* numba

## Run
To do object tracking on video run from the project's folder:

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
