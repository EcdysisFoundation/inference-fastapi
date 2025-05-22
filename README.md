# inference-fastapi

## Test Inference

To perform inference with the app running and from the home directory ...

    curl -X POST  -F "file=@app/test/diabrotica.JPG" http://localhost:8070/metaformer-predict

## Development

The Docker compose volume defined as `./app:/code/app` copies the app directory to the Docker image at `docker compose up`. To get code changes to appear on the app, bring the container down then back up after editing code changes so a new copy of the code is moved to the container. As a result, only code within the /app directory is available in the continaer, unless moved there another way.

## Metaformer model deployment

This section refers to deployment of the model generated from https://github.com/EcdysisFoundation/metaformer_ecdysis

To get the model .pth file on the Docker container, a Docker named volume is used for better performance due to the large size of the .ppth file. Additional model config files are stored in the same location for clearer syncronicity when updating the model.  Use scp to place these files into the local /models folder. These files are .gitignored.

Bring the container up to copy files to the volume, then restart it

Bring the container up, without attaching the terminal to it.

    docker compose up -d

Check the logs

    docker compose logs --tail=1000 --follow

To add new files to this volume, get the CONTAINER_ID

    docker compose ps -q web

copy these files to the CONTAINER_ID.

    docker cp ./models/metaformer_best.pth CONTAINER_ID:/models
    docker cp ./models/metaformer_morphospecies_map.csv CONTAINER_ID:/models
    docker cp ./models/metaformer_config.yaml  CONTAINER_ID:/models

Restart the docker container.

## Yolo model deployment

This section refers to deployment of the model generated from https://github.com/EcdysisFoundation/ultralytics . The model must be exported to .onnx format first. The Ultraltics library conflicts with other libraries in use here therefore, it should not be included in the environment here. The files needed to deploy the Yolo model are identified in constants.py, see `constants.REQUIRED_PATHS_YOLO`. These files need to be moved to the /models folder then the Docker volume in the same way as explained above.
