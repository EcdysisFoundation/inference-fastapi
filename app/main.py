#import onnxruntime
import os
#import psutil

from fastapi import FastAPI, File, UploadFile

from .utils import read_yaml, dict_to_object, prepare_img_foryolo
from .metaformer.handler import MetaformerHandler
from .yolo_handler.onnx_object_detection import OnnxObjectDetection
from .yolo_handler.images_input import Images
from . import constants


app = FastAPI(redirect_slashes=False)


metaformer_enabled = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_METAFORMER]):
    metaformer_handler = MetaformerHandler()
    metaformer_handler.initialize()
    metaformer_enabled = True
    print('metaformer_enabled')


yolo_enabled = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_YOLO]):
   #sess_options = onnxruntime.SessionOptions()
    #sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    #yolo_net = onnxruntime.InferenceSession(
    #    constants.PATH_YOLO_ONNX,
    #    providers=(['CPUExecutionProvider']),
    #    sess_options=sess_options)
    yolo_handler = OnnxObjectDetection(weight_path=constants.PATH_YOLO_ONNX)
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
    # curl -X POST -F "file=@test/diabrotica.JPG" http://localhost:8070/metaformer-predict
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
    # file.file: A SpooledTemporaryFile object, which is a file-like object providing methods for reading and interacting with the file's content.
    #image_data = await file.read()
    #print(file.filename)
    image_data = await file.read()
    image_data = Images(images=Images.read_from_upload_file(file.filename, image_data))
    if yolo_enabled:
        for i, batch in enumerate(image_data.create_batch(batch_size=1)):
            raw_out = yolo_handler.predict_object_detection(
                input_data=batch.to_onnx_input(image_size=yolo_handler.input_size))
            batch.init_detected_objects(raw_out)

            print(batch[0])

        #img = prepare_img_foryolo(image_data)
        #out = yolo_net.run(None, {'input': {'images': [img]}})
        #print(out.shape)
        #results = out[0]
        #results = results.transpose()
        #print(results.shape)



    return {'message': 'working on it'}
