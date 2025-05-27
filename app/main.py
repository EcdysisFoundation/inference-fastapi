import onnxruntime as ort
import os
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile

from .utils import read_yaml, dict_to_object
from .metaformer.handler import MetaformerHandler
from .yolo_handler.processing import (
    preprocess, postprocess, draw_detections, get_detections, YAML)
from . import constants


app = FastAPI(redirect_slashes=False)


METAFORMER_ENABLED = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_METAFORMER]):
    METAFORMER_HANDLER = MetaformerHandler()
    METAFORMER_HANDLER.initialize()
    METAFORMER_CONFIG = read_yaml('..' + constants.PATH_METAFORMER_CONFIG)
    METAFORMER_ENABLED = True
    print('metaformer_enabled')


YOLO_ENABLED = False
YOLO_SAVE_IMG_LOCAL = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_YOLO]):
    YOLO_SESSION = ort.InferenceSession(
        constants.PATH_YOLO_ONNX,
        providers=["CPUExecutionProvider"])
    yolo_model_inputs = YOLO_SESSION.get_inputs()
    YOLO_MODEL_NAME = yolo_model_inputs[0].name
    YOLO_INPUT_SHAPE = yolo_model_inputs[0].shape
    YOLO_INPUT_WIDTH = YOLO_INPUT_SHAPE[2]
    YOLO_INPUT_HEIGHT = YOLO_INPUT_SHAPE[3]
    YOLO_ENABLED = True
    print('yolo_enabled')


@app.get("/")
def read_root():
    return {"message": "Welcome to Inference-FastAPI!"}


@app.get("/metaformer")
def results():
    response = {"model": "metaformer", "version": None}
    if METAFORMER_CONFIG:
        config = dict_to_object(METAFORMER_CONFIG)
        response["version"] = config.VERSION
        print(config)
        return response
    else:
        return response


@app.post('/metaformer-predict')
async def predict_img(file: UploadFile = File(...)):
    # Test this with
    # curl -X POST -F "file=@app/test/diabrotica.JPG" http://localhost:8070/metaformer-predict
    image_data = await file.read()

    if isinstance(image_data, (bytearray, bytes)):
        if METAFORMER_ENABLED:
            output = METAFORMER_HANDLER.handle(image_data)
        else:
            return {'message': 'MetaFormer is not enabled.'}
    else:
        return {'message': 'Warning: Image is not a bytearray.'}

    return output


@app.post('/yolo-predict')
async def yolo_predict_img(file: UploadFile = File(...)):
    image_data = await file.read()
    result = {'message': 'Yolo is not enabled'}
    if YOLO_ENABLED:
        input_image = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image_shape = input_image.shape[:2]  # img_height, img_width
        image_data, pad = preprocess(
            input_image,
            YOLO_INPUT_WIDTH,
            YOLO_INPUT_HEIGHT)
        outputs = YOLO_SESSION.run(None, {YOLO_MODEL_NAME: image_data})
        postprocesed = postprocess(
            outputs,
            pad,
            YOLO_INPUT_HEIGHT,
            YOLO_INPUT_WIDTH,
            image_shape)
        if YOLO_SAVE_IMG_LOCAL:
            img = input_image
            for i in postprocesed:
                img = draw_detections(
                    img,
                    i['box'],
                    i['score'],
                    i['class_id'])
            img_w_boxes = Image.fromarray(img)
            img_w_boxes.save('app/images/{0}'.format(file.filename))
        result = get_detections(postprocesed, image_shape)
    return result
