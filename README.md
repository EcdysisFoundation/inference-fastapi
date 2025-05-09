# inference-fastapi

## Test Inference

To perform inference with the app running and from the home directory ...

    curl -X POST  -F "file=@app/test/diabrotica.JPG" http://localhost:8070/metaformer-predict

## Development

The Docker compose volume defined as `./app:/code/app` copies the app directory to the Docker image at `docker compose up`. To get code changes to appear on the app, bring the container down then back up after editing code changes so a new copy of the code is moved to the container. As a result, only code within the /app directory is available in the continaer, unless moved there another way.

## Model deployment

To get the model .pth file on the Docker container, a Docker named volume is used for better performance due to the large size of the .ppth file. Additional model config files are stored in the same location for clearer syncronicity when updating the model. These files are .gitignored

To add new files to this volume, get the CONTAINER_ID

    docker compose ps -q web

copy these files to the CONTAINER_ID.

    docker cp ./models/metaformer_best.pth CONTAINER_ID:/models
    docker cp ./models/metaformer_morphospecies_map.csv CONTAINER_ID:/models
    docker cp ./models/metaformer_config.yaml  CONTAINER_ID:/models
