# inference-fastapi

This is a https://github.com/fastapi/fastapi app to deploy AI models for image classification and object detection, intended to be ran on a private network or localhost. It is configured for two types of models. Once setup, images can be processed through the docs page, http://localhost:8090/docs or with curl, `curl -X POST  -F "file=@app/test/diabrotica.JPG" http://localhost:8070/metaformer-predict`

## Metaformer model deployment

This section refers to deployment of the model generated from https://github.com/EcdysisFoundation/metaformer_ecdysis

In main.py, METAFORMER_ENABLED is True when the following files are present.

To get the model .pth file on the Docker container, a Docker named volume is used. To add new files to this volume, get the CONTAINER_ID

    docker compose ps -q web

copy these files to the CONTAINER_ID.

    docker cp ./models/metaformer_best.pth CONTAINER_ID:/models
    docker cp ./models/metaformer_morphospecies_map.csv CONTAINER_ID:/models
    docker cp ./models/metaformer_config.yaml  CONTAINER_ID:/models

Restart the docker container.

## Yolo model deployment

This section refers to deployment of an Ultralytics object detection model ( https://github.com/ultralytics/ultralytics ). The model must be exported to .onnx format first. The Ultraltics library conflicts with other libraries in use here therefore, it should not be included in the environment here. The files needed to deploy the Yolo model are identified in constants.py, see `constants.REQUIRED_PATHS_YOLO`. These files need to be moved to the /models folder then the Docker volume in the same way as explained above.
