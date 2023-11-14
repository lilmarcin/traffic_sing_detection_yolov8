# Road Sign Detection with YOLOv8

This project demonstrates road sign detection using YOLOv8, trained on the Road Sign Detection dataset available on Kaggle.

## Overview

The goal of this project is to develop a model capable of detecting and localizing road signs, including stop signs, crosswalks, traffic lights, and speed limit signs. The YOLOv8 architecture is used for its efficiency and effectiveness in object detection tasks.

## Dataset

The dataset used for training and evaluation is the [Road Sign Detection dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) from Kaggle. It contains annotated images with various road signs, making it suitable for training a road sign detection model.

## YOLOv8

YOLO (You Only Look Once) is a real-time object detection algorithm. YOLOv8, an improved version, is used in this project. It divides an image into a grid and predicts bounding boxes and class probabilities for each grid cell.

## Usage

### Installation

0. If you want to train on GPU, make sure you have access to GPU. Type `nvidia-smi` to check if graphic card is detected and has proper Driver version  and CUDA version.

1. Install ultralytics [Ultralytics yolov8 quickstart](https://docs.ultralytics.com/quickstart/):

   ```bash
   pip install ultralytics
   ```

2. Install dependencies (torch, opencv, matplotlib, pillow, seaborn, etc.)
    ```bash
    pip install -U -r requirements.txt
    ```

3. Prepare dataset (split data for model's train, validation and test)
Organize your dataset with the following structure:
    ```bash
    datasets/
    ├─── road_sign/
    |   ├── annotations/
    |   │   ├── train/
    |   │   ├── val/
    |   │   └── test/
    |   ├── images/
    |   │   ├── train/
    |   │   ├── val/
    |   │   └── test/
    |   |   labels/
    |   │   ├── train/
    |   │   ├── val/
    |   │   └── test/
    |─── road_sign.yaml
    ```
- annotations: contains a set of .xml files that store, among others: annotations the name of the detected object and bboxes
- images: set of png images
- labels: converted data from annotations folder to yolo format (format .txt with type and bboxes of detected object)
- road_sign.yaml - configuration file to train yolov8.
```bash
train: ../road_sign/images/train/ 
val:  ../road_sign/images/val/
test: ../road_sign/images/test/

# number of classes
nc: 4

# class names
names: ["trafficlight","stop", "speedlimit","crosswalk"]
```
4. Train dataset using yolov8
```python
model = YOLO("yolov8n.pt")
model.train(data="datasets/road_sign.yaml", epochs=40)
```

5. Validation

- Confusion matrix

<img src="runs\detect\train\confusion_matrix_normalized.png" alt="matrix" width="800"/>

- Results

<img src="runs\detect\train\results.png" alt="results" width="800"/>

6. Test model on images
```python
# Load a model
model = YOLO('runs/detect/train/weights/best.pt')  # best trained YOLOv8n model

# Run batched inference on a list of images
results = model(['examples/example1.png', 'examples/example2.png', 'examples/example3.png', 'examples/example4.png'], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
```
- test dataset
<img src="runs\detect\train\val_batch0_pred.jpg" alt="Val0" width="500"/>
<img src="runs\detect\train\val_batch1_pred.jpg" alt="Val1" width="500"/>
<img src="runs\detect\train\val_batch2_pred.jpg" alt="Val2" width="500"/>

- random images
<img src="examples\example2.png" alt="Example2" width="500"/>
<img src="examples\example3_result.png" alt="Example3" width="500"/>
<img src="examples\example4.png" alt="Example4" width="500"/>

7. Test model video
```Python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Open the video file
video_path = "examples/example1.mp4"
cap = cv2.VideoCapture(video_path)

#codec = cv2.VideoWriter_fourcc(*"MJPG")
#out = cv2.VideoWriter('./processed.avi' , codec, 30, (800, 600))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (800, 600)) 

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        #out.write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```

<img src="examples\example_yolov8_2.gif"/>

<img src="examples\poniatowski.gif"/>