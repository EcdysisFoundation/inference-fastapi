import onnxruntime as ort
import os
import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile

from .utils import read_yaml, dict_to_object
from .metaformer.handler import MetaformerHandler
from .yolo_handler.processing import (
    preprocess, postprocess, draw_detections, get_detections)
from . import constants


app = FastAPI(redirect_slashes=False)


metaformer_enabled = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_METAFORMER]):
    metaformer_handler = MetaformerHandler()
    metaformer_handler.initialize()
    metaformer_enabled = True
    print('metaformer_enabled')


yolo_enabled = False
YOLO_SAVE_IMG_LOCAL = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_YOLO]):
    yolo_session = ort.InferenceSession(
        constants.PATH_YOLO_ONNX,
        providers=["CPUExecutionProvider"])
    yolo_model_inputs = yolo_session.get_inputs()
    yolo_model_name = yolo_model_inputs[0].name
    yolo_input_shape = yolo_model_inputs[0].shape
    yolo_input_width = yolo_input_shape[2]
    yolo_input_height = yolo_input_shape[3]
    yolo_enabled = True
    print('yolo_enabled')


@app.get("/")
def read_root():
    return {"message": "Welcome to Inference-FastAPI!"}


@app.get("/metaformer")
def results():
    response = {"model": "metaformer", "version": None}
    config = read_yaml('../models/metaformer_config.yaml')
    if config:
        config = dict_to_object(config)
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
        if metaformer_enabled:
            output = metaformer_handler.handle(image_data)
        else:
            return {'message': 'MetaFormer is not enabled.'}
    else:
        return {'message': 'Warning: Image is not a bytearray.'}

    return output


@app.post('/yolo-predict')
async def yolo_predict_img(file: UploadFile = File(...)):
    image_data = await file.read()
    result = {'message': 'Yolo is not enabled'}
    if yolo_enabled:
        input_image = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image_shape = input_image.shape[:2]  # img_height, img_width
        image_data, pad = preprocess(
            input_image,
            yolo_input_width,
            yolo_input_height)
        outputs = yolo_session.run(None, {yolo_model_name: image_data})
        postprocesed = postprocess(
            outputs,
            pad,
            yolo_input_height,
            yolo_input_width,
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
