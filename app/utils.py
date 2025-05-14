import yaml

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
