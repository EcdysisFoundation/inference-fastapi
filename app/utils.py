import cv2
import yaml
import numpy as np

from fastapi import HTTPException


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_object(data):
    if isinstance(data, dict):
        return Config(**data)
    return data


def format_yolo_output(detections):

    # Extract bounding box data
    print(detections[0].boxes.xyxy)
    print(detections[0].boxes.conf)
    print(detections[0].boxes.cls)
    boxes = detections[0].boxes.xyxy.cpu().numpy()
    scores = detections[0].boxes.conf.cpu().numpy()
    classes = detections[0].boxes.cls.cpu().numpy()

    # Format the results as a list of dictionaries
    results = []
    for box, score, cls in zip(boxes, scores, classes):
        results.append({
            'x1': box[0],
            'y1': box[1],
            'x2': box[2],
            'y2': box[3],
            'confidence': score,
            'class': int(cls)
        })
    return results


def prepare_img_foryolo(image_data):

    # Convert the file contents to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode the numpy array as an image using cv2.imdecode
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    # we need it in RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize image to get the desired size (640,640) for inference
    img = cv2.resize(img,(640, 640))
    #img_height, img_width = img.shape[:2]
    #print('image size after resize...')
    #print(img_height)
    #print(img_width)

    # change the order of image dimension from (640,640,3) to (3,640,640)
    # img = img.transpose(2, 0, 1)
    # add an extra dimension at index 0 (indicates a single image)
    # img = img.reshape(1, 3, 640, 640)
    # scale pixels to 0-1 from 0-255
    # img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img
