import os
from typing import Union

from fastapi import FastAPI, File, UploadFile

from .utils import read_yaml, dict_to_object
from .metaformer.handler import MetaformerHandler

from . import constants


app = FastAPI(redirect_slashes=False)


metaformer_enabled = False
if all([os.path.exists(v) for v in constants.REQUIRED_PATHS_METAFORMER]):
    metaformer_handler = MetaformerHandler()
    metaformer_handler.initialize()
    metaformer_enabled = True


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


# Use ../models/metaformer_config.yaml to get metaformer detials
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
